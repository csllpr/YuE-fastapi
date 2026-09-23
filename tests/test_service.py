"""HTTP service tests with a fake pipeline; no GPU or model download needed."""
import json
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from yue2.service import ServiceConfig, create_app


class FakePlan:
    truncated = False
    timing = {"seconds": 0.01, "output_tokens": 3}

    def __init__(self, abc):
        self.abc = abc

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "score.abc").write_text(self.abc, encoding="utf-8")
        (directory / "plan.json").write_text(json.dumps({"abc": self.abc}))


class FakeResult:
    truncated = {"abc": False, "semantic": False}

    def __init__(self, kwargs):
        self.kwargs = kwargs

    def save_artifacts(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "audio.flac").write_bytes(b"fLaC-fake-audio")
        (directory / "score.abc").write_text("X:1\nK:C\nC8|", encoding="utf-8")
        (directory / "result.json").write_text("{}")
        return {"identity": "fake-identity", "truncated": self.truncated,
                "audio_seconds": 1.5, "timing": {"e2e_seconds": 0.5},
                "artifacts": {"audio.flac": "sha", "score.abc": "sha", "result.json": "sha"}}


class FakePipe:
    def __init__(self, block=None):
        self.block = block
        self.calls = []
        self.closed = False

    def plan(self, cancelled=None, **kwargs):
        self.calls.append(("plan", kwargs))
        return FakePlan(kwargs.get("abc") or "X:1\nK:C\nC8|")

    def __call__(self, cancelled=None, **kwargs):
        self.calls.append(("song", kwargs))
        if self.block is not None:
            self.block.wait(timeout=30)
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before VAE")
        return FakeResult(kwargs)

    def close(self):
        self.closed = True


def make_client(tmp_path, pipe=None):
    config = ServiceConfig(output_dir=tmp_path)
    app = create_app(config, pipeline_factory=lambda: pipe or FakePipe())
    return TestClient(app)


def wait_for(client, job_id, want=("complete", "failed", "cancelled"), timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/v1/jobs/{job_id}").json()["status"]
        if status in want:
            return client.get(f"/v1/jobs/{job_id}").json()
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still {status}")


def song_body(**overrides):
    body = {"style": "piano pop", "lyrics": "[Verse]\nhello world"}
    body.update(overrides)
    return body


def test_song_job_end_to_end(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/v1/songs", json=song_body(seed=7))
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        done = wait_for(client, job_id)
        assert done["status"] == "complete"
        assert done["result"]["truncated"] == {"abc": False, "semantic": False}
        assert done["result"]["audio_seconds"] == 1.5

        audio = client.get(f"/v1/jobs/{job_id}/audio")
        assert audio.status_code == 200 and audio.content == b"fLaC-fake-audio"
        assert audio.headers["content-type"].startswith("audio/flac")

        score = client.get(f"/v1/jobs/{job_id}/score")
        assert score.status_code == 200 and score.text.startswith("X:1")

        names = client.get(f"/v1/jobs/{job_id}/artifacts").json()
        assert "audio.flac" in names and "result.json" in names
        result = client.get(f"/v1/jobs/{job_id}/artifacts/result.json")
        assert result.status_code == 200

        listing = client.get("/v1/jobs").json()
        assert [j["job_id"] for j in listing] == [job_id]


def test_plan_job_saves_only_notation(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/v1/plans", json=song_body())
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        done = wait_for(client, job_id)
        assert done["status"] == "complete"
        assert client.get(f"/v1/jobs/{job_id}/score").status_code == 200
        assert client.get(f"/v1/jobs/{job_id}/audio").status_code == 404


def test_validation_rejects_bad_requests(tmp_path):
    with make_client(tmp_path) as client:
        assert client.post("/v1/songs", json={"lyrics": "x"}).status_code == 422
        assert client.post("/v1/songs", json=song_body(cot="off", abc="X:1\nK:C\n")).status_code == 422
        assert client.post("/v1/songs", json=song_body(nope=1)).status_code == 422
        assert client.post("/v1/songs", json=song_body(seed=-1)).status_code == 422
        assert client.post("/v1/plans", json=song_body(semantic_sampling={"top_k": 5})).status_code == 422


def test_unknown_job_and_artifact_paths(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/v1/jobs/deadbeef").status_code == 404
        assert client.delete("/v1/jobs/deadbeef").status_code == 404
        job_id = client.post("/v1/songs", json=song_body()).json()["job_id"]
        wait_for(client, job_id)
        assert client.get(f"/v1/jobs/{job_id}/artifacts/../result.json").status_code in {400, 404}
        assert client.get(f"/v1/jobs/{job_id}/artifacts/missing.bin").status_code == 404


def test_cancel_queued_job(tmp_path):
    block = threading.Event()
    pipe = FakePipe(block=block)
    config = ServiceConfig(output_dir=tmp_path)
    app = create_app(config, pipeline_factory=lambda: pipe)
    with TestClient(app) as client:
        first = client.post("/v1/songs", json=song_body(id="first")).json()["job_id"]
        second = client.post("/v1/songs", json=song_body(id="second")).json()["job_id"]
        deadline = time.time() + 5
        while client.get(f"/v1/jobs/{first}").json()["status"] != "running" and time.time() < deadline:
            time.sleep(0.02)
        cancelled = client.delete(f"/v1/jobs/{second}")
        assert cancelled.json()["status"] == "cancelled"
        block.set()
        assert wait_for(client, first)["status"] == "complete"
        assert wait_for(client, second)["status"] == "cancelled"


def test_model_info_and_health(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/health").json()["worker_alive"] is True
        info = client.get("/v1/model").json()
        assert info["model"] == "m-a-p/YuE2-3B" and info["vae"] == "m-a-p/YuE2-Vae"
