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

## Decision record

| Date | Engine | Environment | Smoke log | Decision |
|---|---|---|---|---|
| 2026-11-19 | vLLM (attempted) | `~/venv-serve` | — | install failed: `Disk quota exceeded`; not retried (quota, second torch, scale) |
| 2026-11-19 | transformers | `~/venv`, locked stack | pending | adopted; vLLM claim dropped |
