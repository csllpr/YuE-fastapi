"""Queued HTTP service around a single YuE2Pipeline.

The torch pipeline does not support concurrent generation, so the app owns one
worker thread that holds one pipeline and drains a FIFO queue. HTTP handlers
only enqueue jobs, poll state, and serve saved artifacts from disk.
"""
from __future__ import annotations
import json
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .protocol import SongRequest, GenerationConfig, resolve_sampling

JOB_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
SAMPLING_KEYS = ("abc_sampling", "semantic_sampling")
REQUEST_KEYS = ("style", "tags", "lyrics", "cot", "seed", "abc", "cfg_scale", "id")


@dataclass
class ServiceConfig:
    model: str = "m-a-p/YuE2-3B"
    vae: str = "m-a-p/YuE2-Vae"
    revision: str | None = None
    vae_revision: str | None = None
    device: str = "auto"
    budget: float = 24
    backend: str = "torch"
    quantization: str = "none"
    offload_ar: bool = False
    offline: bool = False
    output_dir: Path = Path("runs/service")
    max_queue: int = 64
    generation_config: dict | None = None


@dataclass
class Job:
    id: str
    kind: str  # "song" or "plan"
    kwargs: dict
    output_dir: Path
    status: str = "queued"  # queued, running, complete, failed, cancelled
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    receipt: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def public(self):
        data = {"job_id": self.id, "kind": self.kind, "status": self.status,
                "created_at": self.created_at, "finished_at": self.finished_at,
                "error": self.error}
        if self.receipt:
            data["result"] = self.receipt
        return data


def validate_request(data, kind):
    """Return normalized pipeline kwargs or raise ValueError with the reason."""
    unknown = set(data) - set(REQUEST_KEYS) - set(SAMPLING_KEYS)
    if unknown:
        raise ValueError(f"Unknown request fields: {sorted(unknown)}")
    kwargs = {k: v for k, v in data.items() if k in REQUEST_KEYS and v is not None}
    SongRequest(style=kwargs.get("style", kwargs.get("tags")),
                lyrics=kwargs.get("lyrics"),
                cot=kwargs.get("cot", "full"), seed=kwargs.get("seed", 831001),
                abc=kwargs.get("abc"), cfg_scale=kwargs.get("cfg_scale"),
                id=kwargs.get("id", "song"))
    defaults = GenerationConfig()
    for key, default in (("abc_sampling", defaults.abc), ("semantic_sampling", defaults.semantic)):
        if key in data and data[key] is not None:
            if kind == "plan" and key == "semantic_sampling":
                raise ValueError("Plan jobs do not accept semantic_sampling")
            kwargs[key] = resolve_sampling(data[key], default)
    return kwargs


class JobRunner:
    """One worker thread owning one pipeline; jobs are strictly serialized."""

    def __init__(self, config, pipeline_factory=None):
        self.config = config
        self.pipeline_factory = pipeline_factory or self._default_factory
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.queue: queue.Queue = queue.Queue(maxsize=config.max_queue)
        self.pipe = None
        self.thread = threading.Thread(target=self._run, name="yue2-worker", daemon=True)

    def _default_factory(self):
        from .pipeline import YuE2Pipeline
        c = self.config
        return YuE2Pipeline.from_pretrained(
            c.model, vae=c.vae, revision=c.revision, vae_revision=c.vae_revision,
            device=c.device, memory_budget_gib=c.budget, backend=c.backend,
            quantization=c.quantization, offload_ar=c.offload_ar,
            local_files_only=c.offline, progress=True,
            generation_config=GenerationConfig.from_dict(c.generation_config)
            if c.generation_config else None)

    def start(self):
        self.thread.start()

    def stop(self):
        self.queue.put(None)
        self.thread.join(timeout=30)
        if self.pipe is not None:
            self.pipe.close()
            self.pipe = None

    def submit(self, kind, kwargs):
        job_id = uuid.uuid4().hex
        job = Job(id=job_id, kind=kind, kwargs=kwargs,
                  output_dir=Path(self.config.output_dir) / job_id)
        try:
            self.queue.put_nowait(job)
        except queue.Full:
            raise
        with self.lock:
            self.jobs[job.id] = job
        return job

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def cancel(self, job):
        job.cancel_event.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = time.time()
        return job.status != "queued"  # True if a running job must unwind

    def _run(self):
        while True:
            job = self.queue.get()
            if job is None:
                return
            if job.status == "cancelled":
                continue
            job.status = "running"
            try:
                if self.pipe is None:
                    self.pipe = self.pipeline_factory()
                cancelled = job.cancel_event.is_set
                if job.kind == "plan":
                    plan = self.pipe.plan(cancelled=cancelled, **job.kwargs)
                    plan.save(job.output_dir)
                    job.receipt = {"truncated": {"abc": plan.truncated},
                                   "timing": plan.timing,
                                   "artifacts": sorted(p.name for p in job.output_dir.iterdir())}
                else:
                    result = self.pipe(cancelled=cancelled, **job.kwargs)
                    receipt = result.save_artifacts(job.output_dir)
                    job.receipt = {"identity": receipt["identity"], "truncated": receipt["truncated"],
                                   "audio_seconds": receipt["audio_seconds"],
                                   "e2e_seconds": receipt["timing"]["e2e_seconds"],
                                   "artifacts": sorted(receipt["artifacts"])}
                job.status = "complete" if not job.cancel_event.is_set() else "cancelled"
            except InterruptedError:
                job.status = "cancelled"
            except Exception as exc:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                (job.output_dir).mkdir(parents=True, exist_ok=True)
                try:
                    (job.output_dir / "failure.json").write_text(
                        json.dumps({"status": "failed", "type": type(exc).__name__, "reason": str(exc)}))
                except OSError:
                    pass
            finally:
                job.finished_at = time.time()


def create_app(config=None, pipeline_factory=None):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, PlainTextResponse

    config = config or ServiceConfig()
    runner = JobRunner(config, pipeline_factory)

    @asynccontextmanager
    async def lifespan(app):
        runner.start()
        yield
        runner.stop()

    app = FastAPI(title="YuE2 song generation API", version="0.1.0", lifespan=lifespan)

    def job_or_404(job_id):
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(404, f"Unknown job {job_id}")
        return job

    def submit(data, kind):
        try:
            kwargs = validate_request(data, kind)
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc))
        try:
            return runner.submit(kind, kwargs)
        except queue.Full:
            raise HTTPException(429, f"Queue full ({config.max_queue} pending jobs)")

    @app.get("/health")
    def health():
        return {"status": "ok", "worker_alive": runner.thread.is_alive(),
                "queued": runner.queue.qsize()}

    @app.get("/v1/model")
    def model_info():
        c = config
        return {"model": c.model, "vae": c.vae, "revision": c.revision,
                "vae_revision": c.vae_revision, "device": c.device, "budget_gib": c.budget,
                "backend": c.backend, "quantization": c.quantization}

    @app.post("/v1/songs", status_code=202)
    def create_song(data: dict = Body(...)):
        return submit(data, "song").public()

    @app.post("/v1/plans", status_code=202)
    def create_plan(data: dict = Body(...)):
        return submit(data, "plan").public()

    @app.get("/v1/jobs")
    def list_jobs():
        with runner.lock:
            jobs = sorted(runner.jobs.values(), key=lambda j: j.created_at)
        return [j.public() for j in jobs]

    @app.get("/v1/jobs/{job_id}")
    def job_status(job_id: str):
        return job_or_404(job_id).public()

    @app.delete("/v1/jobs/{job_id}")
    def cancel_job(job_id: str):
        job = job_or_404(job_id)
        if job.status in {"complete", "failed", "cancelled"}:
            return job.public()
        runner.cancel(job)
        return job.public()

    def artifact_path(job, name):
        if not JOB_NAME.fullmatch(name) or name in {".", ".."}:
            raise HTTPException(400, "Invalid artifact name")
        path = job.output_dir / name
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, f"No artifact {name}")
        return path

    @app.get("/v1/jobs/{job_id}/audio")
    def audio(job_id: str):
        job = job_or_404(job_id)
        return FileResponse(artifact_path(job, "audio.flac"), media_type="audio/flac",
                            filename=f"{job_id}.flac")

    @app.get("/v1/jobs/{job_id}/score")
    def score(job_id: str):
        job = job_or_404(job_id)
        return PlainTextResponse(artifact_path(job, "score.abc").read_text(encoding="utf-8"),
                                 media_type="text/vnd.abc")

    @app.get("/v1/jobs/{job_id}/artifacts")
    def artifacts(job_id: str):
        job = job_or_404(job_id)
        if not job.output_dir.is_dir():
            raise HTTPException(404, "Job has no artifacts")
        return sorted(p.name for p in job.output_dir.iterdir() if p.is_file())

    @app.get("/v1/jobs/{job_id}/artifacts/{name}")
    def artifact(job_id: str, name: str):
        return FileResponse(artifact_path(job_or_404(job_id), name))

    return app
