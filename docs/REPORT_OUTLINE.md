# 12-page report: outline and sources

The report is the team's writing; its numbers are not. Every figure comes from one of the generated files below,
copied as generated, so the report cannot drift from the runs. When a result changes, regenerate the file and copy
again; never edit a number in the report by hand.

| Generated file | Generator | Covers |
|---|---|---|
| `docs/report_draft_week3.md` | `scripts/report_week3.py --write` | retrieval, the three-way comparison, Eval-C, the judge, 3.14 latency |
| `docs/report_draft_week4.md` | `scripts/report_week4.py --write` | merge, quantisation, endpoint, container, red-team, clean clone, what was not established |
| `MODEL_CARD.md` | `scripts/serve/modelcard.py --write` | provenance, test results with paired tests, failure modes, refusals, deployment |
| `results/redteam/run_a_r16.md` | `scripts/serve/redteam.py --render` | every red-team attempt |
| `README.md` (Results, Week 2) | hand-written in Week 2 from `results/evals/` | the Week 2 comparison table |

**Rules for the whole report**
- Compare models with the paired exact McNemar test on identical items (discordant counts and p), never as two
  percentages. Put `n` beside any claim that rests on a difference of a few points (one score at n = 180 is ±7 points).
- State each deviation from the template in the section it affects, with its reason: `transformers.generate` instead
  of vLLM (quota, a second torch), no Eval-B, DPO cut, 4-bit rejected, the container never run on a GPU.
- A result that went against us is reported in the same tone as one that went for us: the vision arm's 4-point test
  drop beside its abstention gain, the ablation's val lead that did not hold on test, the 2.22-point quantisation drop.

## Sections (12 pages)

| # | Section | Pages | Content | Numbers from |
|---|---|---|---|---|
| 1 | Summary | 0.5 | Task, model, headline result (Run A vs few-shot), what was deployed, the two main findings (fine-tuning is the effect; the model detects missing information but answers anyway) | MODEL_CARD.md: Evaluation, Failure modes |
| 2 | Problem and setting | 0.75 | Chart QA for finance/economics; why grounded JSON answers (evidence, rationale, abstention); the 4-week plan and its decisions | Plan §1–2, §9 |
| 3 | Data: ChartAudit-Fin v2.0 | 1.5 | ChartQA lineage and revision, rule-based selection (exclusion counts), blind labelling and strict matching, acceptance rates, pilot and second-labeller checks, the human audit κ, Eval-C construction, split disjointness | MODEL_CARD.md: Training data provenance; `data/v2.0/README.md`; `data/eval_c/stats.json`; week 3 draft: Eval-C |
| 4 | Fine-tuning | 1.25 | QLoRA setup, Run A vs Run B (rank), answer-only ablation (showing working), vision-tower arm (3.0j); model selection on val before test | README Week 2 table; MODEL_CARD.md comparison table; `docs/RUNS.md`; W&B report |
| 5 | Retrieval (RAG) | 1 | BM25 over whole tables, query plans, the embedding and reranker comparisons and why none was adopted, the three-way comparison, retrieval failure | week 3 draft; `docs/RAG.md` |
| 6 | Evaluation method | 0.75 | Relaxed accuracy, schema validity, evidence grounding, the abstention pair, the error taxonomy, the LLM judge and its measured reliability | `scripts/scoring.py` docstring; week 3 draft: judge |
| 7 | Results | 2 | Test results with paired tests; per question type; abstention on test and Eval-C; error taxonomy; judge | MODEL_CARD.md: Evaluation, Refusal behaviour, Failure modes; week 3 draft |
| 8 | Serving | 1.5 | Verified merge (weights and answers); the endpoint (validation, retry, fallback, refusals); latency in-process vs HTTP; quantisation and its rejection; the container and its rules | week 4 draft: 4.1–4.4; `docs/SERVING.md` decision record |
| 9 | Responsible AI | 1.25 | Red-team design (7 families, answer vs abstain, misled vs wrong), results, the stated-then-answered pattern; intended and out-of-scope use; refusals | week 4 draft: 4.5; `results/redteam/run_a_r16.md`; MODEL_CARD.md: Intended use |
| 10 | Limitations and what was not established | 0.75 | Copy the closing lists of both drafts; nothing added from memory | week 4 draft: last section; week 3 draft: last section; MODEL_CARD.md: Not measured |
| 11 | Reproducibility | 0.5 | Pinned lock, manifests, provenance gates on every job, generated documents, the clean-clone result | README: Reproduce; week 4 draft: 4.7 |
| | Appendix (outside the 12 pages) | | Full red-team log, per-type tables, decision record | the generated files |

## Before submitting

- [ ] Your teammate's 3.0i and 3.7 results are committed (`results/audit/kappa.txt`, `results/judge/run_a_r16/agreement.json`),
      and `MODEL_CARD.md`, both drafts and the W&B report are regenerated after them.
- [x] Licences: recorded in `docs/licences.json` (2026-11-23) and rendered in `MODEL_CARD.md`: the base model is under the
      Qwen Research License (research and evaluation only; "Built with Qwen" shown), ChartQA under GPL-3.0. Put the
      non-commercial restriction in the report's intended-use paragraph. **Open for the team:** the repository has no
      LICENSE file of its own although it redistributes GPL-3.0-derived data; choose one deliberately.
- [ ] `python scripts/serve/release.py --tests` passes on the commit you submit, and its result is committed.
- [ ] Every number in the report appears in one of the generated files.
