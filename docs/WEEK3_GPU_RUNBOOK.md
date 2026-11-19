# Week 3 GPU runbook (VM03)

Every pending Week 3 job, in the order that keeps results comparable. Each job refuses to start unless the data
matches its manifest, the tracked code is committed and `~/venv` matches `requirements-vm03.lock`, so never
`pip install` anything into `~/venv` and never edit tracked files on VM03 while jobs are queued.

`slurm/*.sbatch` request `--exclusive`, so jobs run one at a time in submission order: submit a whole phase, then
watch it. `squeue -u $USER` shows the queue; `tail -f logs/<job-name>-<JOB_ID>.out` follows one job.

## 0. Prepare (login shell, once)

```bash
cd ~/chart-explorer
git fetch && git checkout week3-carry-forward && git pull      # main, once the PR is merged
git status --short                                              # must print nothing tracked
mkdir -p logs
python -m pytest tests -q                                       # 145 passed (pip install pytest if missing:
                                                                # it pins nothing the lock checks)
```

## 1. Serving stack (task 3.0f) — first, it is the one task that cannot be cut

Decided 2026-11-19 (`docs/SERVING.md`): serve with `transformers.generate` from `~/venv`; vLLM did not fit the
quota. Nothing to install. The merge writes about 7 GB, so check `quota -s` first (35 GB hard limit per user).

```bash
M=/storage/tracka/users/learner01/merged/run_a_r16_smoke
SERVE_VENV=$HOME/venv sbatch slurm/serve_smoke.sbatch --merge ~/runs/run_a_r16/adapter --merged $M --engine transformers
```

Pass = the log ends `SERVE_SMOKE_OK`. Copy `logs/serve_smoke-<JOB_ID>.json` to `docs/serve_smoke_log.json`, fill in
the "pending" cell of the decision record in `docs/SERVING.md`, commit both, push, and `git pull`. Do that **before**
submitting step 3, because jobs refuse to start while a tracked file has uncommitted changes. If the quota is tight
for the training runs, delete `$M` once the log is committed.

## 2. Embedding retriever against the lexical floor (task 3.3) — before any RAG evaluation

Retrieval only, no training stack: use its own venv so `~/venv` stays locked. CPU is enough.

```bash
python3 -m venv ~/venv-embed && ~/venv-embed/bin/pip install sentence-transformers pyyaml
~/venv-embed/bin/python scripts/rag/metrics.py --retrievers tfidf bm25 embed --write
git diff --stat results/rag/
```

- If `results/rag/retrieval.json` still says `"chosen": ... "retriever": "bm25"`, only the selection table gained the
  embedding rows: commit `results/rag/` and go on.
- If an embedding configuration was chosen, the RAG prompt sets must be rebuilt **before** step 4:
  `~/venv-embed/bin/python scripts/rag/prompts.py --write`, then commit `results/rag/` and `data/rag/` together.
  (The committed tests check that `data/rag/` matches the chosen configuration.)

Then `git pull` on VM03 so the clean-tree check sees the commit.

## 3. Training runs (tasks 3.0h, 3.0j) — every member who has not trained submits one

Dry-run each config on the login shell first; it checks config and data without a GPU.

```bash
python scripts/train.py configs/run_a_table.yaml --dry-run
python scripts/train.py configs/run_a_vision.yaml --dry-run

sbatch slurm/train.sbatch configs/run_a_table.yaml                                       # member 1
sbatch slurm/train.sbatch configs/run_a_vision.yaml                                      # member 2
sbatch slurm/train.sbatch configs/run_a.yaml --set run_name=run_a_seed1_<name> --set train.seed=1   # member 3
sbatch slurm/train.sbatch configs/run_a.yaml --set run_name=run_a_seed2_<name> --set train.seed=2   # member 4
```

A run that hits the 4-hour limit saves a checkpoint and exits with code 75: resubmit the same command with `--resume`.

**Check the vision run before reading its result:** `~/runs/run_a_r16_vision/targets.json` must show non-zero
`vision_tower` and `merger` counts. If the run exited at start-up with a `lora.scope` message, the module paths in
this transformers version differ from the pattern; report the message, do not edit the pattern on VM03.
Expect more memory and slower steps; do **not** change `max_pixels` or the batch.

## 4. Evaluations — after step 3 has finished

**Select on val first.** Compare each new adapter with `run_a_r16` on val before touching any reported set:

```bash
sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16_table/adapter  --split val
sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16_table/adapter  --split rag_val
sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16_vision/adapter --split val
sbatch slurm/eval.sbatch --adapter ~/runs/run_a_seed1_<name>/adapter --split val       # optional, seed spread
sbatch slurm/eval.sbatch --adapter ~/runs/run_a_seed2_<name>/adapter --split val
```

**Then reported sets, once per model** (`--final-test` is required, and means it: never rerun to get a better number).

Eval-C (task 3.0c), the existing models and the new ones:

```bash
sbatch slurm/eval.sbatch --split eval_c --targets answer_only --shots 2 --precision 4bit --max-new-tokens 384   # few-shot base
sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16/adapter            --split eval_c --final-test
sbatch slurm/eval.sbatch --adapter ~/runs/run_b_r64/adapter            --split eval_c --final-test
sbatch slurm/eval.sbatch --adapter ~/runs/ablation_answer_only/adapter --split eval_c --final-test
sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16_vision/adapter     --split eval_c --final-test
```

Vision ablation on test (task 3.0j):

```bash
sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16_vision/adapter --split test --final-test
```

RAG conditions (tasks 3.5, 3.6):

```bash
sbatch slurm/eval.sbatch --split rag_test          --targets answer_only --shots 2 --precision 4bit --max-new-tokens 384
sbatch slurm/eval.sbatch --split rag_test_withheld --targets answer_only --shots 2 --precision 4bit --max-new-tokens 384
sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16/adapter       --split rag_test          --final-test
sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16_table/adapter --split rag_test          --final-test
sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16_table/adapter --split rag_test_withheld --final-test
# optional, the floor end of the bracket
sbatch slurm/eval.sbatch --split rag_test_qo --targets answer_only --shots 2 --precision 4bit --max-new-tokens 384
sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16_table/adapter --split rag_test_qo --final-test
```

Each writes `~/evals/<name>/`; a name that already exists is refused, which is the point.

## 5. Bring the results back and regenerate (once every job in step 4 is done)

```bash
cd ~/chart-explorer
for d in ~/evals/*eval_c* ~/evals/*rag_* ~/evals/run_a_r16_table-* ~/evals/run_a_r16_vision-* ~/evals/run_a_seed*; do
  [ -e "results/evals/$(basename "$d")" ] || cp -r "$d" results/evals/
done
grep -L '"reportable": true' results/evals/*/summary.json          # must print nothing
python scripts/rescore.py                                            # every line OK
python scripts/taxonomy.py --all --write
python scripts/compare.py --split eval_c --reference base_eval_c_image_only_answer_only_2shot_4bit
python scripts/threeway.py --write
python scripts/report_week3.py --write
python -m pytest tests -q
git add results/ docs/report_draft_week3.md && git commit -m "Week 3 GPU results: Eval-C, RAG conditions, vision and table arms"
git push
```

Read the "What Week 3 has not yet established" list at the end of `docs/report_draft_week3.md`. After this runbook,
it should hold only the human tasks: the audit κ (3.0i) and the judge with its spot-check (3.7, which runs from a
machine with Anthropic API credentials, not VM03).

## If something refuses to start

| Message | Meaning | Fix |
|---|---|---|
| `uncommitted changes in N tracked file(s)` | VM03 checkout differs from a commit | `git status`; restore or commit on the laptop, `git pull` |
| `package(s) differ from requirements-vm03.lock` | something was installed into `~/venv` | reinstall from the lock; use a separate venv for extras |
| `does not match MANIFEST.txt` | a data file changed or has CRLF endings | `git checkout -- data/` |
| `add --final-test` | an adapter on a reported set | only after selection on val, once |
| `lora.scope ... selected only` | the vision pattern adapted no vision module | report it; do not loosen the guard |
| `... already exists: pass --resume` | run folder already there | `--resume`, or a new `run_name` |
