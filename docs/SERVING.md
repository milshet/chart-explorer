# Serving stack (task 3.0f)

**Decision, 2026-11-19: serve through `transformers.generate` behind FastAPI. The vLLM claim is dropped.**
This is the fallback task 3.0f named in advance; the smoke test below is what confirms it works.

## Why not vLLM

An attempt to install vLLM into a separate `~/venv-serve` on VM03 failed with `[Errno 122] Disk quota exceeded`
on the last of 201 packages. The reasons it is not retried:

1. **Quota.** Home and `/storage` are one filesystem with a per-user quota of 32 GB soft / 35 GB hard. After removing
   the failed venv and purging pip's cache, 17.9 GB is in use, leaving about 14-17 GB. vLLM pulls a complete
   second torch stack (CUDA 13 build, cuDNN, NCCL, FlashInfer, Triton), on the order of 10 GB installed; the merged
   bf16 checkpoint needs about 7 GB more; and Week 3 still has four training runs to write. All three do not fit.
2. **A second torch.** The vLLM release that resolved pins a CUDA 13 torch, while every measured number ran on
   torch 2.14.1+cu126 from the shared runtime. Serving on a different torch is a different numerical stack from the
   one the model card describes, and a CUDA 13 build needs a newer driver than the cu126 stack, which was not verified.
3. **Scale.** The served model is 3B on one L40S. vLLM's throughput advantage matters under concurrent load; this
   deployment's job is one request at a time with a measured latency, which `transformers.generate` delivers.

What it costs: no continuous batching or paged attention, so throughput under concurrent load is lower; task 3.8
reports the measured single-request latency and tokens/s on this stack instead of quoting vLLM figures. The report
says "served with transformers behind FastAPI because vLLM's dependencies did not fit the VM's quota", not that a
vLLM deployment was built.

What it gains: the served model runs on **exactly** the locked stack every Week 1-3 number was measured on
(`requirements-vm03.lock`, torch from the shared runtime at `/storage/tracka/runtime/venv`), so there is no second
environment to pin: `requirements-serve.lock` is not needed. Week 4 adds only FastAPI and uvicorn on top; check after
installing them that the lock check still passes (`python scripts/evaluate.py --split val --targets answer_only
--dry-run` prints any drift).

## Smoke test on VM03

From `~/venv`, with nothing installed. The merged checkpoint goes on `/storage` (one filesystem with home, but it
keeps large artefacts in one place to clean up):

```bash
cd ~/chart-explorer && git pull && mkdir -p logs
M=/storage/tracka/users/learner01/merged/run_a_r16_smoke
# merge Run A into bf16 weights and send one request (one job)
SERVE_VENV=$HOME/venv sbatch slurm/serve_smoke.sbatch --merge ~/runs/run_a_r16/adapter --merged $M --engine transformers
# later, one request against the existing merge
SERVE_VENV=$HOME/venv sbatch slurm/serve_smoke.sbatch --model $M --engine transformers
```

Pass: the log ends `SERVE_SMOKE_OK`, meaning the answer to `val-two_col_83529-542` is JSON-valid, schema-valid and
correct (Runs A and B both answer it correctly on val, so a failure is the stack, not the model). Copy
`logs/serve_smoke-<JOB_ID>.json` to `docs/serve_smoke_log.json` and commit it.

The merged checkpoint is a smoke artefact: task 4.1 does the verified merge. Delete
`/storage/tracka/users/learner01/merged/run_a_r16_smoke` once the smoke log is committed if the quota is tight.

## Verified merge on VM03 (task 4.1)

Every Week 4 number comes from this checkpoint, so it is checked against the model that was measured before
anything else uses it. `scripts/serve/merge.py` merges on CPU in bf16 (never 4-bit), writes the checkpoint, then runs
three checks:

| Check | What fails it | On failure |
|---|---|---|
| **Base** | `adapter_config.json` names another base, or the base revision in `run_summary.json` differs from the local snapshot. DoRA, rank/alpha patterns or `modules_to_save`, which the weight check can't reproduce | raises before merging (`MERGE_REFUSED`) |
| **Weights** | The merged checkpoint, read back from disk, isn't base + (alpha/r)·B@A computed from the adapter file (not through PEFT) in exactly the modules the adapter names. Or a tensor outside them changed, or a dtype changed | raises (`MERGE_REFUSED`), manifest stays `UNVERIFIED` |
| **Answers** | Merged vs unmerged base+adapter on 100 val items (seed 41001), through `scripts/serve/generation.py`, compared on parsed answers. More than 3 disagree, or more than 1 schema-valid answer lost. Tolerance fixed in code before the run | `MERGE_FAILED_ANSWER_CHECK`, manifest `FAILED_ANSWER_CHECK` |

```bash
cd ~/chart-explorer && git pull && mkdir -p logs && quota -s       # the merge writes ~7 GB
A=/storage/tracka/users/learner01/runs/run_a_r16/adapter            # or your copy, with run_summary.json beside adapter/
python scripts/serve/merge.py --adapter $A --dry-run                # login shell: base check, sample, provenance
sbatch slurm/merge.sbatch --adapter $A                              # -> ~/merged/run_a_r16 + results/serve/merge_run_a_r16.json
sbatch slurm/merge.sbatch --adapter $A --verify-only                # re-check an existing merge, no new weights
```

Pass: the log ends `MERGE_VERIFIED`. Commit `results/serve/merge_run_a_r16.json`. The checkpoint stays in
`~/merged/run_a_r16` as the served model. `merge_manifest.json` beside its weights records the status, the adapter
and weight hashes and both checks, and `/health` reports it (4.4). If you copied only `adapter/`, copy the run's
`run_summary.json` next to it too; otherwise the base revision can't be verified and the job needs
`--allow-unrecorded-revision`, which the manifest records. The 3.0f smoke merge
(`/storage/tracka/users/learner01/merged/run_a_r16_smoke`) is unverified and is not the served model; delete it
once this one passes if the quota is tight.

The report also shows how often the batch-1 reference matches the committed batch-4 val evaluation of the same
adapter (`reference_vs_committed_eval`). That figure is for information only: batching and padding change greedy
decoding slightly, so it is not a gate.

## The endpoint (task 4.3)

FastAPI in front of `transformers.generate`, serving the verified merge (`~/merged/run_a_r16`). vLLM is not used
(decision 12 above); the latency quoted is the measured one.

| Piece | File | Role |
|---|---|---|
| Request path | `scripts/serve/service.py` | parsing, image limits, refusals, prompt, counters, `/health`; no web framework |
| Validation and retry | `scripts/serve/decoding.py` | `scoring.contract_errors()`, re-ask with the errors quoted, fallback |
| Engine | `scripts/serve/generation.py` | the one generation path (4.1 verified it) |
| HTTP | `scripts/serve/app.py` | FastAPI routes only |
| Client, bench | `scripts/serve/client.py`, `scripts/serve/http_bench.py` | standard-library client; the 3.14 requests over HTTP |

**API.** `POST /v1/answer` with `{"question": str, "image_base64": str}` returns an envelope:

```json
{"result": {"evidence": [...], "rationale": "...", "answer_type": "number", "unit": null, "answer": 38},
 "degraded": false, "attempts": 1, "retry_errors": [],
 "model": {"run": "run_a_r16", "status": "VERIFIED", "adapter_sha256": "..."},
 "usage": {"prompt_tokens": 702, "new_tokens": 70}, "timing": {"generate_s": 1.18, "total_s": 1.22}}
```

- **Contract.** `scoring.contract_errors()` is stricter than `schema_valid`, which stays unchanged so no
  reported number moves. It requires exactly the contract's fields, and the cross-field rule both ways: the
  answer is null if and only if `answer_type` is `unanswerable`. Every gold answer in the repo and every
  schema-valid Run A output satisfies it.
- **Retry.** A reply that fails the contract goes back as the assistant turn, the errors are quoted in a new user
  turn, and the model answers again, up to 3 attempts in all.
- **Fallback.** After 3 failures the caller gets a contract-valid abstention with `"degraded": true` in the
  envelope, never inside `result`, so it can't be mistaken for the model's own refusal.
- **Counters.** `GET /metrics` reports requests, first-attempt-valid, retried, recovered, degraded, the model's own
  abstentions, refusals by code and engine errors. The report quotes them.
- **Refused, not ignored.** Any field that would put caller text into the prompt (`context`, `table`, `messages`,
  `system`, `prompt`, ...) returns 400 `caller_text_refused`. Unknown fields are refused too. Images must be
  PNG/JPEG/WebP, at most 8 MB and 40 MP (checked from the header before decoding), and at least 28 px a side.
  They're resized exactly as in training. Line breaks in the question become spaces.
- **Health.** `GET /health` reports the merge manifest: status, adapter and weight hashes, base revision. The app
  refuses to start on a checkpoint whose manifest isn't `VERIFIED`.

**Install FastAPI and uvicorn once, without moving the locked stack.** The lockfile is passed as a constraints
file, so pip either keeps every pinned package at its pinned version or fails. It never upgrades one:

```bash
source ~/.tracka_environment && source ~/venv/bin/activate && cd ~/chart-explorer
pip install --dry-run fastapi uvicorn -c requirements-vm03.lock     # read the "Would install" line
pip install fastapi uvicorn -c requirements-vm03.lock
python scripts/evaluate.py --split val --targets answer_only --dry-run | grep -i warning   # must print nothing
pip freeze | grep -iE "^(fastapi|uvicorn|starlette)=="              # these get added to the lockfile
```

**Measure served latency through HTTP** (the same 50 requests as 3.14):

```bash
python scripts/serve/http_bench.py --dry-run
sbatch slurm/serve_http_bench.sbatch          # -> results/serve/run_a_r16_http.json; ends HTTP_BENCH_OK
```

**Run the endpoint** (for the demo; it holds the GPU, so `scancel` it when done):

```bash
sbatch slurm/serve.sbatch
grep -m1 "Uvicorn running" logs/chartqa-serve-<JOB_ID>.out
curl -s localhost:8000/health
python scripts/serve/client.py data/v2.0/images/val/two_col_24100.png "What is the sum all the blue bar?"
ssh -N -L 8000:127.0.0.1:8000 -p 1026 learner01@<vm03>     # from a laptop: http://localhost:8000/docs
```

## Decision record

| Date | Engine | Environment | Smoke log | Decision |
|---|---|---|---|---|
| 2026-11-19 | vLLM (attempted) | `~/venv-serve` | — | install failed: `Disk quota exceeded`; not retried (quota, second torch, scale) |
| 2026-11-19 | transformers | `~/venv`, locked stack | `docs/serve_smoke_log.json` (job 85: SERVE_SMOKE_OK) | adopted; vLLM claim dropped |
