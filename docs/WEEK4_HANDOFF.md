# Week 4 handoff

Written 2026-11-20, at the end of Week 3; updated 2026-11-21 with the results of tasks 3.10–3.16. Week 4 is in `docs/Capstone_Project_4WeekPlan.md` (§6, Week 4, tasks
4.1–4.8). This note covers what that plan does not: where things stand, what has already been decided, and
what VM03 will do to you.

## 1. Where Week 3 left things

Everything is on `main`, and 164 tests pass on the laptop.

| Area | Status | Where |
|---|---|---|
| Final model | **Run A, r16** (§9 decision 4); it stays the model to serve, see §3 | `~/runs/run_a_r16/adapter` on learner01 |
| Serving engine | **`transformers.generate` behind FastAPI; no vLLM** (§9 decision 12) | `docs/SERVING.md` |
| Serving smoke test | Passed: merged bf16, JSON-valid, schema-valid, correct, 1.67 s for one request | `docs/serve_smoke_log.json` |
| Week 3 numbers | All machine-run results measured and committed | `docs/report_draft_week3.md`, `results/evals/`, `results/rag/`, `results/judge/`, `results/serve/` |
| RAG decisions | BM25 over whole tables, chart-metadata queries; no embedding model or reranker adopted | `docs/RAG.md` (generated) |
| Served latency (input for 4.3) | Run A p50 1.256 s, p95 1.81 s, first token 0.086 s, ~60 tok/s, 7.3 GB peak | `results/serve/run_a_r16.json` |
| Experiment tracking (deliverable) | W&B Weeks 2–3 report: all five runs' curves plus the Week 3 results | link in `README.md`; `scripts/wandb_report_week3.py` |
| Red-team charts (input for 4.5) | 21 charts, 7 families (3 abstain, 4 answer), seeds 700000+ | `data/redteam/`, `scripts/redteam_charts.py` |
| Error taxonomy (input for 4.6) | `taxonomy.json` in every eval folder | `scripts/taxonomy.py` |
| Abstention pair and Eval-C (input for 4.6) | True and false abstention in every `summary.json`; 50 Eval-C probes | `scripts/scoring.py`, `data/eval_c/` |

Tasks 3.10–3.16 closed the template gaps in items 5 and 6 and the tracking deliverable. All of them ran on VM03
and their results are committed:

| Task | Outcome |
|---|---|
| 3.10 | Four embedding models (MiniLM, bge-small, e5-small, bge-base) against BM25. **None adopted**: the closest, bge-base, reached recall@1 0.925 against BM25's 0.933 |
| 3.11 | Two cross-encoder rerankers over BM25's top 10. **Neither adopted**: recall@1 0.867 (ms-marco-MiniLM) and 0.892 (bge-reranker-base), though BM25's top 10 held the right table for 98.3% of rows |
| 3.12 | `docs/RAG.md`, the generated decision record. Since nothing was adopted, serving needs no embedding model and no RAG number changed |
| 3.13 | The LLM judge, run locally with `arena-education/Llama-3.1-8B-Instruct-bnb-8bit` (gpt-oss-20b did not fit; plan §9 decision 13). Its abstention verdict matches the exact answer on 63/80 rows; all 17 misses are answerable rows it marked "no" |
| 3.14 | Served latency and tokens/sec for Run A and the answer-only ablation (`scripts/serve_bench.py`), see §2 |
| 3.15 | The Week 3 draft and this handoff regenerated and updated |
| 3.16 | W&B report covering Weeks 2–3, generated from the committed results (`scripts/wandb_report_week3.py`) |

Still open, all of it **human** work, owned by the Week 3 team:

- **3.7, judge spot-check κ.** Grader 1's grades are committed in
  `results/judge/run_a_r16/judge_spot_check.jsonl`; grader 2 is pending. Graders use `audit/judge_review.html`,
  which writes their grades into that file. Then `python scripts/judge.py agree --name run_a_r16`, then
  `python scripts/report_week3.py --write`, then rerun `scripts/wandb_report_week3.py` so the dashboard shows the κ
  (it makes a new report; update the README link).
- **3.0i, audit κ:** two or three reviewers fill in `audit/audit_v2.xlsx`, then run `scripts/audit_kappa.py`. The
  model card's data-provenance section needs this, and renders `NOT MEASURED` without it.
- **3.0h, member training runs: unconfirmed.** `docs/RUNS.md` assigns two seed reruns (`run_a_seed1_<name>`,
  `run_a_seed2_<name>`) to members 3 and 4, and neither is in the repo. The template requires every member to run a
  training job, so either they run (about 35 min each) or the gap is recorded as a decision.

**Served latency in 4.3:** rerun the same 50 requests through FastAPI by pointing `serve_bench.py`'s request loop at
the endpoint. The difference from `results/serve/run_a_r16.json` (p50 1.256 s without HTTP) is the HTTP cost. The
1.67 s smoke figure is a single request, not the 3.8 number.

## 2. Headline results, for the model card and the report

All comparisons are paired exact McNemar tests on identical items (`scripts/compare.py`, `scripts/threeway.py`).
Quote them as such, never as differences of percentages.

- **Fine-tuning is the effect.** Run A scores 83.9% on test against 57.2% for the few-shot base (p = 3.5e-11).
- **Retrieval adds nothing on top of fine-tuning.** The table-trained model with RAG scores 84.4% against Run
  A's 83.9% (9 vs 8 discordant items, p = 1).
- **The prompt format costs accuracy.** Giving Run A a retrieved table it was not trained on drops it to 76.1%
  (p = 0.0066). That is why `run_a_r16_table` exists.
- **When retrieval fails** (task 3.6):
  - Table model: lower, not significant (77.6% vs 83.5%, p = 0.23).
  - Base model: harmful, and it abstains on questions it should answer (21.2% vs 54.1%, p = 2.5e-07).
- **The vision LoRA (3.0j) helps abstention without a significant accuracy cost.**
  - Eval-C true abstention: 42/50, against 31/50 for Run A. The vision arm wins on 11 probes and loses on none
    (p = 0.001).
  - Test accuracy: 80.0% against 83.9%, 3 vs 10 discordant items (p = 0.092), so not significant. Say so; don't
    hide the 4 points.
- **Eval-C true abstention:** few-shot base 24/50, Run A 31/50, Run B 31/50, answer-only ablation 30/50, vision
  42/50. Many abstentions don't say why (`refusal_reason_unstated` in `taxonomy.json`).
- **Retrieval:** BM25 over whole-table chunks. Test recall@1 is 0.24 with question-only queries and 0.90 with
  chart-metadata queries. Four embedding models and two rerankers were measured against it on a train selection
  set, with the adoption rule fixed in advance; none beat it (`docs/RAG.md`).
- **LLM judge:** Llama-3.1-8B 8-bit, three axes on 80 rows (correctness 47/58, grounding 57/72, abstention
  57/80). Quote it with its measured reliability: 63/80 against the exact abstention answer, it graded
  correctness on 7 rows where the rubric says n/a, and the human κ once 3.7 lands. Don't quote its axis counts
  alone.
- **Served latency:** Run A p50 1.256 s, p95 1.81 s; the answer-only ablation p50 0.442 s, p95 0.538 s. First
  token (0.086 s) and decode rate (~60 tok/s) are the same for both, so the 2.8× gap is entirely Run A's evidence
  and rationale (72.4 vs 21.8 tokens per answer). Both 98% schema-valid on these 50 requests.

## 3. Decisions already made, so don't reopen them

- **Serve Run A, not the vision arm.** The model was selected on val, where the vision arm scored 79.0% against
  Run A's 81.5%. Switching after seeing test and Eval-C would be selecting on the reported sets. The vision
  arm's abstention gain is a finding for the report, not a reason to change the served model.
- **No vLLM.** The plan's 4.3 row now says `transformers.generate` (§9 decision 12, `docs/SERVING.md`). The
  template names vLLM, so the report, the model card and the demo state the deviation and its reason up front,
  and quote the measured 3.14 latency, never vLLM figures. The cost is no continuous batching.
- **No vector store to containerise.** Retrieval is BM25 built in-process from `data/v2.0/tables/`; the plan's
  4.4 row was corrected to match.
- **No `schema.json`.** Task 4.3 enforces the contract with validate-and-retry, using `scripts/scoring.py`'s
  check, and reports the retry and fallback counters.
- **Code goes in `scripts/`, not `src/`.** Read the plan's `src/serve/...` paths as roles. Runs are Slurm jobs
  driven by committed scripts and YAML (§6.0).

## 4. VM03 rules that will bite you

- **Two jobs per user, maximum.** Slurm rejects a third submission (`QOSGrpSubmitJobsLimit`). To queue a longer
  list, use a helper that submits the next job whenever a slot frees up. learner01 has one at `~/submit_queue.sh`
  (reads one `sbatch ...` command per line from a file); copy it.
- **Every job refuses to start on a dirty tree.** Don't edit, commit or `git pull` in the checkout while jobs are
  queued.
- **Every job refuses to start if `~/venv` drifts from `requirements-vm03.lock`.** Before installing FastAPI and
  uvicorn into `~/venv`, run `pip install --dry-run fastapi uvicorn` and check it upgrades nothing that the
  lock pins. Then confirm that `python scripts/evaluate.py --split val --targets answer_only --dry-run` prints
  no drift. AWQ/GPTQ tooling (4.2) almost certainly pins its own torch, so put it in a separate venv. To make
  that venv reuse the shared torch instead of downloading a second one, copy `tracka_shared_runtime.pth` from
  `~/venv/lib/python3.12/site-packages/` into it, and check with `pip install --dry-run` that `torch` is not
  in the install list.
- **Quota: 35 GB hard per user,** with home and `/storage` on one filesystem. A merged bf16 checkpoint is about
  7 GB; a quantised copy adds more. Check `quota -s` before every merge or export, and delete smoke artefacts
  once their logs are committed.
- **No `python` until you activate:** `source ~/.tracka_environment && source ~/venv/bin/activate`, on every
  login.
- **Pushing from VM03 needs a GitHub fine-grained token** (Contents: read and write, this repo only) or an SSH
  key. Passwords are rejected. Don't store the token on disk; `git config --global credential.helper 'cache
  --timeout=28800'` keeps it in memory.
- **Slurm's `sacct -S today-1` is rejected;** use `-S now-1days`.
- **Timings on the L40S:** a LoRA training run takes about 35 min; an evaluation 1–4 min; the serving smoke job
  32 s including the merge.

## 5. Adapters live in learner01's space

The adapters are under `/storage/tracka/users/learner01/runs/`:

| Adapter | Path |
|---|---|
| Final model | `run_a_r16/adapter` |
| Others | `run_b_r64`, `ablation_answer_only`, `run_a_r16_table`, `run_a_r16_vision` |

Before your first job, check that your account can read them. If it can't, ask learner01 to copy
`run_a_r16/adapter` (114 MB) to you, or to grant read access. Every evaluation records `adapter_sha256`, so a
copy is fine as long as its hash matches.

## 6. Where to start

1. **4.1, verified merge.** The smoke test's `scripts/serve_smoke.py::merge()` is unverified. 4.1 adds the two
   checks: a base-model mismatch raises, and the merged model is scored against base+adapter on a sample,
   comparing parsed answers. Every later Week 4 number comes from these merged weights.
2. **4.3, endpoint, with its HTTP latency measured against `results/serve/`.** This is 20% of the grade (working deployed system).
   Protect it before 4.2.
3. **4.2, quantisation.** "Reject, serve bf16" is a legitimate, pre-named outcome. Fix the tolerance before
   measuring the delta.
4. **4.5 red-team, 4.6 model card, 4.7 report and clean-clone check, 4.8 demo.** Generate the card and the Week 4
   draft from committed `summary.json` files, the same way `scripts/report_week3.py` does. Anything not measured
   renders as `NOT MEASURED`.

**Useful references:**
- `docs/WEEK3_GPU_RUNBOOK.md` for the job-submission patterns.
- `docs/RUNS.md` for what each run folder guarantees.
- `scripts/rescore.py` to re-score committed results without a GPU.
