# Training runs: folder layout and rules

What `scripts/train.py` writes, and what any later step (evaluation #19, model selection #7, the final
test #8, the write-up #9) may rely on. If a field or file is not listed here, do not depend on it.

## Where runs live

Each run is one folder on VM03: `~/runs/<run_name>/` (or `--out`). Runs are never overwritten:
`train.py` refuses an existing folder unless `--resume` is given.

**Naming.** Everyone submits as `learner01`, so the run name is the only record of who ran what and why:

```
run_a_r16                     # planned runs: configs/<name>.yaml as committed
run_a_seed1_parul             # a member's rerun (#20): --set run_name=... --set train.seed=1
pilot                         # the pilot (#3), via --out ~/runs/pilot
```

The same name is used in Weights & Biases (project `chartqa-week2`).

## Folder layout

| Path | Written | What it is |
|---|---|---|
| `provenance.json` | at start, before the GPU is used | Code, environment and data the run started with (below) |
| `config.json` | at start | The full merged config (YAML + defaults + `--set`) |
| `metrics.jsonl` | every step and validation | One JSON object per line: `{"kind": "train" or "val", "step": ..., ...}` |
| `checkpoints/step-NNNNN/` | every `save.every_steps`, at a time limit or `--max-steps` | Resumable state. Only the newest `save.keep` are kept |
| `checkpoints/epoch-N/` | end of each epoch | Same content as a step checkpoint, **never deleted**. Used to compare epochs |
| `adapter/` | end of the run only | **The result.** Final LoRA adapter, processor, `train_config.json` |
| `run_summary.json` | when the job ends (any status) | Status, provenance, timing, memory, validation history |
| `wandb/`, `wandb_id.txt` | when W&B logging is on | Local W&B files; the ID lets `--resume` continue the same W&B run |

A checkpoint folder holds `adapter_model.safetensors`, `adapter_config.json` and `trainer_state.pt`
(optimizer, scheduler, step position, RNG state). It appears only once complete (written to `*.tmp`, then
renamed).

## `provenance.json` and the reproducibility gate

Before loading the model, `train.py` checks three things and **refuses to start** if any fails:

| Check | Fails when | Override |
|---|---|---|
| Data | a split file differs from `data/<version>/MANIFEST.txt` | none; the data must match |
| Code | tracked files have uncommitted changes (`git status`) | `--allow-unclean` |
| Environment | an installed package differs from `requirements-vm03.lock` | `--allow-unclean` |

`--allow-unclean` is for throwaway tests only. The run is then recorded with `"clean": false` and its
results must not be reported. After an intentional upgrade, regenerate the lock on VM03
(`python -m pip freeze`, keeping the header) and commit it in the same change.

```json
{
  "git_commit": "full hash",
  "git_dirty": [],
  "versions": {"python": "3.12.3", "torch": "...", "transformers": "...", "peft": "...", "...": "..."},
  "env_drift": {},
  "data": {
    "data_dir": "data/v2.0",
    "train_file": "data/v2.0/image_only/train.jsonl", "train_sha256": "...",
    "val_file": "data/v2.0/image_only/val.jsonl", "val_sha256": "...",
    "images": 941, "images_sha256": "hash over every image the two splits use"
  },
  "clean": true
}
```

`--resume` compares the data fingerprint with the run's own `provenance.json` and stops if it differs.
A different code commit on resume is allowed (e.g. a bug fix) but printed as a warning; the commits of
every job are visible through `slurm_jobs` and the logs.

## `run_summary.json`

Written when the job ends, whatever the outcome. Fields later steps may use:

| Field | Meaning |
|---|---|
| `status` | `completed`, `stopped_at_max_steps`, `paused_time_limit` (resume it), `failed_nonfinite_loss_at_step_N` |
| `run_name`, `config` | As in `config.json` |
| `model`, `model_revision` | Base model and the exact Hugging Face snapshot |
| `git_commit`, `git_dirty`, `clean`, `versions`, `env_drift` | From `provenance.json` (this job) |
| `data` | Fingerprint above plus `train_examples`, `val_examples`, `limit_train` |
| `planned_steps`, `steps_per_epoch`, `steps_done` | Optimizer steps (one step = `grad_accum` examples) |
| `val_history` | `[{"step", "val_loss", "at"}]`, every validation in order |
| `final_val_loss`, `best_val_loss` | Completed runs only |
| `adapter_dir`, `adapter_mib`, `epoch_checkpoints` | Completed runs only: paths of the result and the epoch snapshots |
| `peak_gpu_memory_mib`, `seconds_per_step`, `minutes_this_job`, `slurm_jobs` | Cost and speed |
| `wandb_url` | Link to the run, if W&B was on |

## Rules for evaluation and comparison

1. **Evaluate adapters, not runs.** Load the base model at `model_revision` and attach
   `adapter/` or `checkpoints/epoch-N/` (`PeftModel.from_pretrained`). Use the `max_pixels`, `variant` and
   `targets` from `adapter/train_config.json`, so prompts match training.
2. **Compare only runs with the same data fingerprint** (`data.*_sha256`). A data change is a new version
   (`data/v2.1`, new tag) and is never compared with runs on the old one.
3. **Report only clean runs** (`"clean": true`, `"status": "completed"`).
4. **Model selection uses the validation split only; test is never used to choose or tune.** Every
   configuration is fixed and the final model named on val first. Then each compared model (the planned
   runs and the ablation, #8) is scored on test once, and nothing is changed afterwards. The course brief asks
   for a held-out test set "the team never trains on" and a comparison of runs, not a single run.

## Evaluations

`scripts/evaluate.py` (submit with `slurm/eval.sbatch`) applies rules 1-4: it reads `variant`, `targets` and
`max_pixels` from the adapter's run, refuses a conflicting `--targets`, and scores an adapter on test only with
`--final-test`. It uses the same reproducibility gate as training. One folder per evaluation,
`~/evals/<model>_<split>_<variant>_<style>_<shots>shot_<precision>/`, never overwritten:

| File | What it is |
|---|---|
| `predictions.jsonl` | Per item: `raw` output, `parsed` JSON, `pred`, `gold`, `correct`, `json_valid`, `schema_valid`, `question_type`, `answer_type`, `new_tokens`, `truncated` (hit `max_new_tokens`), `seconds`, `evidence_n`/`evidence_ok` |
| `summary.json` | `overall` (relaxed accuracy, schema validity, refusal and answerable accuracy, evidence grounding, truncated, tokens/s), `by` question type, answer type, source and chart type, `settings`, `provenance` (incl. `adapter_sha256` and the run's training data), `reportable` |

"Correct" is defined once, in `scripts/scoring.py`, which the Week 1 notebook also imports. Reported
evaluations are copied into `results/evals/`.

## Week 3 GPU jobs (tasks 3.0h, 3.0j, 3.0c)

Queue these together early in the week: same data fingerprint, so they are comparable to Run A. Each member
who has not yet trained submits one of them, which satisfies the template's "every member runs a job" rule.

| Job | Command | Who |
|---|---|---|
| `run_a_r16_table` (3.0h, §9 decision 6) | `sbatch slurm/train.sbatch configs/run_a_table.yaml` | member 1 |
| `run_a_r16_vision` (3.0j) | `sbatch slurm/train.sbatch configs/run_a_vision.yaml` | member 2 |
| seed rerun (3.0h) | `sbatch slurm/train.sbatch configs/run_a.yaml --set run_name=run_a_seed1_<name> --set train.seed=1` | member 3 |
| seed rerun (3.0h) | same, `train.seed=2`, `run_a_seed2_<name>` | member 4 |

`configs/run_a_table.yaml` and `configs/run_a_vision.yaml` differ from `configs/run_a.yaml` in exactly one
setting each (`variant`, `lora.scope`); `tests/test_train.py` fails if that ever stops being true.

**Vision run (3.0j).** `lora.scope: decoder+vision` adapts the vision blocks' attention and MLP and the merger,
selected by module *path*. `train.py` exits if a scope adapts nothing, if `decoder` reaches the vision encoder,
or if `decoder+vision` misses the tower or the merger. Every run writes `targets.json` (in the run folder and
in `adapter/`) listing the modules actually adapted, with counts per part: check it before reading the result.
Expect more memory and slower steps; do **not** change `max_pixels` or the batch.

**Then evaluate** each new adapter on val, and only after selection on test and Eval-C, once each:

```bash
sbatch slurm/eval.sbatch --adapter ~/runs/<run>/adapter --split val
sbatch slurm/eval.sbatch --adapter ~/runs/<run>/adapter --split test --final-test
sbatch slurm/eval.sbatch --adapter ~/runs/<run>/adapter --split eval_c --final-test
```

Eval-C also needs the existing models scored once each (`run_a_r16`, `run_b_r64`, `ablation_answer_only`, and
the few-shot base: `--split eval_c --targets answer_only --shots 2 --precision 4bit --max-new-tokens 384`).
Copy every `~/evals/<name>/` into `results/evals/` and run `python scripts/taxonomy.py --all --write`.
