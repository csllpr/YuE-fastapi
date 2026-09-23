# HTTP API service

The `serve` subcommand wraps one `YuE2Pipeline` in a FastAPI app. Generation is
strictly serialized: a single worker thread owns the pipeline and drains a FIFO
queue, so HTTP handlers never touch the GPU directly. Requires the `serve` extra:

```bash
pip install 'yue2-infer[serve]'
yue2 serve --host 127.0.0.1 --port 8000 --output runs/service
# same model flags as generate: --model --vae --revision --budget --backend
# --quantization --offload-ar --offline --config
```

Interactive OpenAPI docs are at `http://127.0.0.1:8000/docs`.

## Workflow

Submitting returns `202` with a `job_id`; poll the job, then fetch artifacts.
Request bodies are the same fields as the Python API: `style`, `lyrics`, `cot`
(`full` | `melody` | `off`), `seed`, `abc`, `cfg_scale`, `id`, plus optional
`abc_sampling` / `semantic_sampling` override dicts. Native validation applies
(e.g. `abc` requires `cot` full/melody; violations return `422`).

```bash
# Submit a song (planning + audio) or a notation-only plan job
curl -X POST localhost:8000/v1/songs -H 'Content-Type: application/json' -d '{
  "style": "English, upbeat synth pop, male vocal",
  "lyrics": "[Chorus]\nWe are running through the night",
  "cot": "off", "seed": 42}'
curl -X POST localhost:8000/v1/plans -H 'Content-Type: application/json' -d '{
  "style": "warm piano pop", "lyrics": "[Verse]\nNeon fades along the lane"}'

curl localhost:8000/v1/jobs/<job_id>          # queued → running → complete/failed/cancelled
curl -X DELETE localhost:8000/v1/jobs/<job_id>  # cancel queued or running job
curl localhost:8000/v1/jobs/<job_id>/audio -o song.flac   # 48 kHz stereo FLAC
curl localhost:8000/v1/jobs/<job_id>/score                # generated ABC notation
curl localhost:8000/v1/jobs/<job_id>/artifacts            # artifact name list
curl localhost:8000/v1/jobs/<job_id>/artifacts/result.json
```

Status payloads include `truncated`, `audio_seconds`, `e2e_seconds`, and the
request identity hash on completion. Artifacts are the native `save_artifacts`
set (`audio.flac`, `score.abc`, `plan.json`, `result.json`, token/latent `.npy`)
under `<output_dir>/<job_id>/`. Job registry is in-memory: finished job metadata
does not survive a restart, but artifact directories do.

Operational notes:

- One process = one GPU worker. The pipeline rejects concurrent generation, so
  scale out with one server process per GPU behind a load balancer, not threads.
- `/health` reports worker liveness and queue depth; a full queue (default 64)
  returns `429`.
- The first job on a fresh server includes model load time; later jobs reuse
  the resident pipeline.
