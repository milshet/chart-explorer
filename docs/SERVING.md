# Serving stack: resolve, pin, smoke (task 3.0f)

**Status: prepared, not yet run.** The scripts below exist and are tested on the laptop; the resolution itself
has to happen on VM03. Until `requirements-serve.lock` is committed, Week 4 tasks 4.1–4.4 have no stack.

## Why a separate venv

Training and every reported number ran on torch 2.14.1+cu126 / transformers 5.22.1 (`requirements-vm03.lock`).
vLLM pins torch. Installing it into `~/venv` either moves torch under the measured runs or fails. So the
serving stack lives in `~/venv-serve` (the `~/venv-report` precedent) and never touches `~/venv`.

## Procedure on VM03

```bash
cd ~/chart-explorer && git pull && mkdir -p logs
python3 -m venv ~/venv-serve
~/venv-serve/bin/pip install --upgrade pip
~/venv-serve/bin/pip install vllm fastapi uvicorn pydantic peft pillow   # let pip resolve torch for vLLM
~/venv-serve/bin/pip freeze > requirements-serve.lock                    # commit this file

# 1. smoke merge of Run A (bf16; task 4.1 does the verified merge)
sbatch slurm/serve_smoke.sbatch --merge ~/runs/run_a_r16/adapter --merged ~/merged/run_a_r16_smoke
# 2. one multimodal request through vLLM, scored by scripts/scoring.py
sbatch slurm/serve_smoke.sbatch --model ~/merged/run_a_r16_smoke
```

Pass: the log ends `SERVE_SMOKE_OK` (the answer to `val-two_col_83529-542` is JSON-valid, schema-valid and
correct; Run A and Run B both answer it correctly on val, so a failure is the stack, not the model). Commit
`requirements-serve.lock` and the log `logs/serve_smoke-<job>.json` as `docs/serve_smoke_log.json`.

## If vLLM will not resolve or will not load the checkpoint

That is a decision, not a blocker, and it is taken this week:

1. **Fallback engine.** `sbatch slurm/serve_smoke.sbatch --model ~/merged/run_a_r16_smoke --engine transformers`.
   If that passes, Week 4 serves through `transformers.generate` behind FastAPI and the report drops the vLLM
   claim. Record the vLLM error text below.
2. **AWQ/GPTQ exporter.** Add `autoawq` (or `llmcompressor` for GPTQ) to `~/venv-serve` and re-freeze. If no
   exporter accepts the VL checkpoint, task 4.2's verdict is `reject` and Week 4 serves fp16 (named in advance).

## Decision record

| Date | Engine | Lock committed | Smoke log | Decision |
|---|---|---|---|---|
| — | — | — | — | not yet run |
