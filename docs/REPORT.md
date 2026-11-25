# Auditable Chart QA: Adapting and Serving a 3B Vision-Language Model for Financial Reporting

**Track A capstone, Executive PG Programme in Generative AI and Agentic AI.** Repository: `chart-explorer`, branch
`week4`. Base model: Qwen2.5-VL-3B-Instruct. Data: ChartAudit-Fin v2.0, built from ChartQA.

_Every number in this report is copied from a generated file (`MODEL_CARD.md`, `docs/report_draft_week3.md`,
`docs/report_draft_week4.md`, `docs/RAG.md`, `results/redteam/run_a_r16.md`) or from the committed predictions they
are built from. Comparisons between models are paired exact McNemar tests on identical items, reported as the two
discordant counts and p. One accuracy on 180 items carries roughly ±7 points, so no claim here rests on a difference
of a few points without its test._

---

## 1. Summary

An analyst reading figures off charts in a filing or a market report needs an answer they can check, and a refusal
when the chart cannot support one. We fine-tuned Qwen2.5-VL-3B-Instruct with QLoRA on 1,033 blind-verified finance
and economics chart questions to answer in a fixed JSON contract: the evidence read off the chart, the reasoning,
then a typed answer, or `null` when the chart cannot answer.

- **Fine-tuning is the effect.** On the 180-item test split the fine-tuned model (Run A) answers 151 correctly
  against 103 for the base model prompted with two examples: 53 items only Run A gets right, 5 only the base
  (p = 3.5e-11).
- **Showing working earns its tokens.** Against an otherwise identical model trained to give only the answer, Run A
  is right on 22 items the ablation misses and wrong on 7 it gets (p = 0.0081), at about three times the latency.
- **Retrieval adds nothing on top of fine-tuning.** A table-trained model with retrieved tables scores 84.4%
  against Run A's 83.9% (9 vs 8 discordant, p = 1).
- **The deployed model is the measured one.** The adapter was merged into bf16 weights and verified exactly
  against the base plus adapter, and the endpoint (FastAPI, `transformers.generate`) answers at a median 1.224 s
  per request. 4-bit quantisation was measured and rejected.
- **The model detects missing information more often than it acts on it.** On 21 adversarial charts it resisted
  every truncated axis, misleading legend and prompt injection, but in 4 of its 5 failures to abstain its own
  rationale says the asked-for data is not in the chart, and it answers anyway.

---

## 2. Problem and setting

The workflow is the extraction of specific figures and comparisons from charts, not prose or attached
spreadsheets, into documents someone is accountable for: investor updates, filings, board packs. A plausible wrong
number there is a compliance exposure, not a typo. That fixes two requirements. Answers must be **auditable**:
the model states which values it read and how it combined them, so a reviewer can check the chart against the
claim. And **abstention is a first-class output**: a chart that does not contain the asked-for series or period
must produce a declared `null`, not an interpolated guess.

Out of the box, a general vision-language model fails here in specific ways: it reads the wrong bar when legend
order is unusual, interpolates between gridlines, answers in prose, invents a value for a series the chart does
not have, and fails questions that need two readings plus arithmetic. The project adapts a 3B model against
exactly that taxonomy and measures each adaptation separately.

The output contract (`scripts/scoring.py`) is one JSON object: `evidence` (a list of strings), `rationale`,
`answer_type` (number, text, yes_no, unanswerable), `unit`, and `answer`, which is `null` exactly when the type is
`unanswerable`.

---

## 3. Data: ChartAudit-Fin v2.0

**Source and selection.** All charts and questions come from ChartQA (`ahmed-masry/ChartQA`, revision
`af8b6f5c08c9`), human-written questions only, using ChartQA's own splits. Selection is by written rules, with no
model in the loop (`scripts/select_clean.py`, seed 2026). Of 7,398 human-written train questions, 4,289 were
outside finance and economics and 803 failed a structural check: no parseable data table, a looked-up number
absent from the chart's own table, colour references the annotation cannot resolve, duplicates, or a table that
recurs in a later split. From the remaining pool, 1,155 train questions were sampled proportionally over source,
chart type, answer type and question type, including 55 unanswerable items made by pairing a question with an
unrelated chart. The train selection draws on Statista (739), Pew (227), Our World in Data (126) and OECD (63).
The splits share no chart.

**Blind labelling.** The training targets need evidence and rationale, which ChartQA does not provide. Each chart
and question was labelled by a fresh model instance per batch of 50 (Claude Sonnet) that saw only the image and the
question, under neutral file names, with a leak check on every pack. An item was kept only if the blind answer was
**the same answer** as the human one under a strict rule: numbers equal to the shorter one's decimals, text equal
after normalisation, unanswerable only if both are null. The training target pairs the blind label's evidence and
rationale with the human ChartQA answer.

| Split | Selected | Kept | Acceptance |
|---|---|---|---|
| train | 1,155 | 1,033 | 89.4% |
| val | 137 | 124 | 90.5% |
| test | 180 | 180 | not labelled: human answers only |

The acceptance rate is the reportable statistic: it measures label quality rather than labeller fluency. Two checks
support it. A 40-item pilot matched on 38 of 40 under the strict rule, and a second model family (GPT, blind, same
prompt) matched the human answer on 29 of 30 random train items, missing the same ambiguous chart the first labeller
missed. A human audit of 50 training labels by three reviewers was designed (`audit/audit_v2.xlsx`) but **not
carried out**, so no inter-reviewer κ is reported for label quality (Section 10).

The kept training set is 715 numeric, 208 text, 69 yes/no and 41 unanswerable answers; by question type, 537
arithmetic or comparison, 272 visual reference, 183 retrieval and 41 unanswerable. The test split holds 96
arithmetic or comparison, 40 visual reference, 35 retrieval and 9 unanswerable questions.

**Eval-C, the abstention set.** Nine unanswerable test items cannot support a claim about refusal, so we built 50
probes that the chart provably cannot answer (seed 31003): 13 ask for an absent series, 13 for an absent category,
12 for something not in the chart, 12 for external knowledge. 38 are real, unused ChartQA questions paired with a
held-out chart, 12 are templated, over 49 distinct val and test charts. The absence is proved against each chart's
data table, not assumed.

**Red-team charts.** Twenty-one charts were generated with their ground truth known by construction (seeds 700000
onwards, disjoint from every dataset seed), in seven families (Section 9).

---

## 4. Fine-tuning

**Setup.** QLoRA: the base model loaded in 4-bit NF4, LoRA adapters (rank 16, alpha 32, dropout 0.05) on the 252
attention and MLP projections of the 36 decoder layers, with the vision tower frozen by construction (the training
script refuses any adapted module name containing `visual`). Two epochs, learning rate 2e-4, effective batch 8
(batch 1 × 8 accumulation steps), 3% warm-up, seed 42, loss on the answer only. A run takes about 35 minutes on one
L40S. Every run records its code commit, data hashes and environment, and refuses to start on an uncommitted tree
or a package set that differs from the lockfile. Training curves for all five runs are in the W&B Weeks 2–3 report
linked from the README.

**Runs.** Run A (rank 16) and Run B (rank 64) vary one field. The answer-only ablation keeps Run A's data, rank and
hyperparameters but trains on targets without evidence and rationale. The vision arm (task 3.0j) extends Run A's
adapters to the vision encoder and its merger. A table-trained Run A (`run_a_r16_table`) sees each chart's data table
in the prompt, for the retrieval comparison.

**Selection on val, before test.** The final model was chosen on the 124-item val split, then scored on test once.

| Model | Val (124) | Paired vs Run A on val (only it / only Run A) |
|---|---|---|
| Run A, rank 16 | 101 (81.5%) | |
| Run B, rank 64 | 105 (84.7%) | 8 / 4, p = 0.39 |
| Answer-only ablation | 102 (82.3%) | 10 / 9, p = 1 |
| Vision arm | 98 (79.0%) | 4 / 7, p = 0.55 |

No variant beat Run A significantly on val. Run A was kept over Run B, whose 4 extra val answers are not
distinguishable from noise, because its adapter is a quarter of the size (114 MB against 457 MB).

**Results on test (180 items).**

| Model | Correct | Paired vs Run A |
|---|---|---|
| **Run A (served)** | **151 (83.9%)** | |
| Base, 2-shot | 103 (57.2%) | 53 only Run A / 5 only base, p = 3.5e-11 |
| Answer-only ablation | 136 (75.6%) | 22 / 7, p = 0.0081 |
| Run B, rank 64 | 150 (83.3%) | 8 / 7, p = 1 |
| Vision arm | 144 (80.0%) | 10 / 3, p = 0.092 |

Rank does not matter at this data size. The ablation shows that the evidence and rationale are not decoration: the
gain sits in arithmetic and comparison questions, where the answer-only model scores 63.5% against Run A's 78.1%.
The ablation's val lead on visual reference did not hold on test; we report it because a lead that evaporates is
evidence about multiple testing. The vision arm scores 4 points lower on test, a difference that is not significant
(p = 0.092), and abstains markedly better (Section 7). The base model's chain-of-thought prompting, without
fine-tuning, produced schema-valid output only 51.1% of the time.

---

## 5. Retrieval

**Corpus.** Every chart in the frozen dataset contributes its data table: 1,168 tables and 7,675 rows, 1,084 from
frozen records and 84 distractor-only. That is 17% above the template's 200–1,000 band; trimming to fit would have
dropped gold tables, so the overshoot is kept and stated.

**Choices, made on a selection set of 120 train rows, never on a reported split** (`docs/RAG.md`). Whole-table chunks
beat row-group chunks, and BM25 is the retriever. The query plan matters more than every other choice: querying
with the chart's own metadata (title, axis and legend text) gives recall@1 0.933 on the selection set, the question
alone 0.208. Four embedding models (MiniLM, bge-small, e5-small, bge-base) and two cross-encoder rerankers were
measured against BM25, each with its adoption rule fixed in advance. None was adopted: the closest embedding model
reached recall@1 0.925 against BM25's 0.933, and the rerankers 0.867 and 0.892, although BM25's top 10 held the right
table for 98.3% of rows. On test, the chosen configuration retrieves the right table first 90.1% of the time
(MRR 0.934); with the question alone, 24.0%.

**The required comparison.**

| Condition | Test accuracy | Paired against |
|---|---|---|
| Base | 57.2% | |
| Base + RAG | 48.9% | vs base: 21 / 36, p = 0.063 |
| Fine-tuned (Run A) | 83.9% | vs base: 53 / 5, p = 3.5e-11 |
| Fine-tuned + RAG | 76.1% | vs fine-tuned: 5 / 19, p = 0.0066 |
| Fine-tuned on tables + RAG | 84.4% | vs fine-tuned: 9 / 8, p = 1 |

Giving Run A a retrieved table it was never trained with costs 8 points (p = 0.0066): the prompt format changed.
The table-trained model separates that from retrieval itself, and shows that retrieval adds nothing measurable on
top of fine-tuning. **When retrieval fails** (85 test rows with the gold table withheld), the table-trained model
drops from 83.5% to 77.6%, not significantly (p = 0.23), but the base model collapses from 54.1% to 21.2%
(p = 2.5e-07) and declines 8 of 85 answerable questions it otherwise answers. The served system therefore uses no
retrieval.

---

## 6. Evaluation method

- **Relaxed accuracy:** numbers within 5% of gold, text and yes/no equal after normalisation, unanswerable items
  correct only with a `null` answer. Reported by question type.
- **Contract validity:** JSON-valid and schema-valid, plus, at serving time, a stricter check
  (`scoring.contract_errors`) including the rule that the answer is null if and only if the type is unanswerable.
  The stricter check is separate, so no reported number moved.
- **Evidence grounding:** the share of numbers the model cites as evidence that appear in the chart's data table,
  which the model never sees in the image-only setting.
- **Abstention, always as a pair:** true abstention (declined on unanswerable items) beside false abstention
  (declined on answerable ones), so a model that refuses everything hard cannot look careful.
- **Error taxonomy** (`scripts/taxonomy.py`): every wrong answer classified, from scale errors to reasoning errors.
- **An LLM judge**, measured before being trusted. `Llama-3.1-8B-Instruct` (8-bit, run locally) graded 80 outputs on
  correctness, grounding and abstention. Its abstention verdict can be checked against an exact answer, and agrees
  on 63 of 80; it graded correctness on 7 rows where the rubric says n/a. Two of us then graded 30 outputs
  independently (two team members, 30 rows):

| Axis | Human vs human κ | Judge vs grader 1 κ | Judge vs grader 2 κ |
|---|---|---|---|
| Correctness | 1.00 | 0.94 | 0.94 |
| Grounding | 0.87 | 0.55 | 0.42 |
| Abstention | 1.00 | 0.54 | 0.54 |

The judge agrees with people about correctness almost as well as they agree with each other, and is unreliable on
grounding and abstention. We therefore quote its correctness verdicts and none of the others.

---

## 7. Results

**Run A on test:** relaxed accuracy 83.9%, 100% JSON- and schema-valid, 87.0% of cited evidence numbers present in
the data table. By question type: retrieval 94.3% (35), visual reference 85.0% (40), arithmetic and comparison
78.1% (96), unanswerable 100% (9).

**Errors.** 29 of 180 answers are wrong: reasoning errors 8, near misses 6, text mismatches 6, extraction errors 5,
answer-type mismatches 2, one false abstention and one scale error; 5 answers cite evidence not in the table.
Arithmetic and comparison questions hold 21 of the 29.

**Abstention.** On test, Run A declined all 9 unanswerable items and 1 of 171 answerable ones. On Eval-C, where all
50 items are unanswerable:

| Model | Declined (of 50) |
|---|---|
| Base, 2-shot | 24 |
| Answer-only ablation | 30 |
| Run A (served) | 31 |
| Run B | 31 |
| Vision arm | 42 |

Fine-tuning improves refusal only modestly over the base model. The vision arm declines 11 probes Run A answers and
answers none that Run A declines (p = 0.00098). We did not switch to it: the model was chosen on val before Eval-C
was scored, and selecting on a reported set afterwards would bias every number in this report. It is the
project's clearest pointer for further work.

---

## 8. Serving

**Engine: `transformers.generate` behind FastAPI, not vLLM.** The template names vLLM. Installing it on the course
VM failed on the 35 GB disk quota, and it would have pulled a second, CUDA 13 build of torch underneath three weeks
of numbers measured on torch 2.14.1+cu126. We recorded the fallback as a decision in advance and serve on exactly
the locked stack every number was measured on. The cost is no continuous batching: every latency below is one
request at a time.

**A verified merge (task 4.1).** The adapter was merged into bf16 base weights and checked three ways before anything
used it. The merge refuses a base model or revision other than the one the adapter was trained on. Read back from
disk, all 252 adapted modules equal base + 2·B@A computed from the adapter file rather than through the library
that did the merge (0 of 2,084,328,516 changed elements off), and the other 573 tensors are bit-identical. On 100 val
items, merged and unmerged models give the same parsed answer on 98, with 84 correct each, within a tolerance of
3 fixed in code beforehand.

**The endpoint (task 4.3).** A request carries a question and a chart image and nothing else: any field that would
put caller text into the prompt (`context`, `table`, `messages`, ...) is refused with an error, not ignored, because
accepting it would hand every caller the injection channel the red-team attacks through pixels. Images are limited
in size and format and resized exactly as in training; the prompt is byte-identical to the training prompt. Every
reply is validated against the contract; a failing reply is sent back with the errors quoted, up to three attempts,
after which the caller receives a valid abstention flagged `degraded` in the envelope, never inside the answer, so a
fallback cannot be counted as the model's own refusal. The counters are reported, not just logged.

| 50 val requests | In-process | Through the endpoint |
|---|---|---|
| p50 latency | 1.256 s | 1.224 s |
| p95 latency | 1.81 s | 1.847 s |
| Accuracy on these items | 84% | 84% |
| Contract-valid | 98% | 100% |

HTTP, JSON and base64 add 3 ms. 49 of 50 replies were valid first time; one (`val-multi_col_1411-760`) failed all
three attempts because the model writes `"answer": 2010/11` without quotes, and received the flagged fallback.
The answer-only model answers at a median 0.442 s against Run A's 1.256 s, at the cost Section 4 measured.

**Quantisation (task 4.2): measured and rejected.** GPTQ 4-bit (group size 128) on the 252 decoder projections, with
the vision tower and merger kept unquantised, calibrated on 128 training conversations checked disjoint from every
evaluation set by id, chart and image bytes. The tolerance was fixed before measuring: at most a 2-point accuracy
drop on test. The 4-bit model scored 81.1% against 83.3% for the bf16 merge on the same 180 items, a 2.22-point drop
(7 lost, 3 gained, p = 0.34): rejected by the rule, although the difference is not significant at this size. It
would have shrunk the model from 7.0 to 3.2 GB, but it also ran slower (2.12 s against 1.29 s per answer), because
the VM has no CUDA compiler for GPTQ's fast kernels. The library choice was itself measured: GPTQModel installs
against the locked environment, llm-compressor does not.

**Container (task 4.4): specified and tested, never run.** The image installs a 65-package subset copied verbatim
from the lockfile with no dependency resolution and a `pip check`; the weights are mounted read-only, never built
into the image; and the health check reports healthy only when the mounted merge manifest says VERIFIED. These rules
are enforced by tests. The course VM has no container runtime and no administrator access, so the image has never
been built or run; we do not claim it has.

---

## 9. Responsible AI

**Red-team (task 4.5).** Twenty-one charts, three per family, each with its answer known by construction and each
with a defined "misled" answer: the one the attack is built to produce. Four families are resisted by answering
correctly and three by abstaining, so a model that refuses everything would resist only 9 of 21.

| Attack | Resisted |
|---|---|
| Truncated y-axis (the ratio the bar heights exaggerate) | 3/3 |
| Dual axis (the line's value, not the bars' axis) | 2/3 |
| Misleading legend order | 3/3 |
| Prompt injection stamped on the chart | 3/3 |
| Absent series | 1/3 |
| Caption-only answer (a forecast printed for a year the plot does not show) | 0/3 |
| Absent period | 3/3 |

Run A resisted 15 of 21: 11 of 12 answer attacks and 4 of 9 abstain attacks. It never took a reading attack's bait.
It computed ratios from the printed labels rather than the bar heights, and the one dual-axis error (15 for 10) is
not the left-axis value. It never obeyed or repeated the injected token. The other five failures are failures to abstain, and at least four of
them share one pattern. "The chart has no Franchise category; the closest is Wholesale at 40." "The chart does not
show 2023 data; management's expectation is given as 490 m USD." The model states that the information is missing
and answers anyway, with the nearest series or the caption's figure. For an auditing workflow this is the most
important limitation we found, and it is consistent with Eval-C.

**Intended use.** Answering a question about one finance or economics chart, of the style the training data
covers, with evidence and reasoning shown, for a person who checks the answer against the chart. Not for financial,
investment or policy decisions without that check; not for other chart styles, scanned charts, multi-chart
documents or other languages; and not for questions answered by text printed on the chart rather than by its data.

**Licences.** The base model is released under the Qwen Research License, which permits research and evaluation use
only; our fine-tuned model is a derivative and carries that restriction, and its documentation displays
"Built with Qwen" as the licence requires. ChartQA is GPL-3.0. The judge model (Llama 3.1 Community License) was used for
evaluation only. `MODEL_CARD.md` records each licence at the revision used.

---

## 10. Limitations and what was not established

Assembled from the gaps the results declare (`docs/report_draft_week4.md`, `MODEL_CARD.md`):

- **Reliable abstention on adversarial charts:** 4 of 9 abstain attacks resisted.
- **Generalisation to unseen chart styles:** no out-of-distribution set (Eval-B) was built: the synthetic chart
  generator it needed was dropped in Week 1, and only a minimal generator for the red-team charts was built later.
  Every number here is in-distribution for ChartQA's four publishers.
- **A quantised deployment:** 4-bit was rejected.
- **Throughput under concurrent load:** every latency is one request at a time.
- **A running container:** the image has never been built or run (Section 8).
- **Label quality by human audit:** the three-reviewer audit was not carried out. Label quality rests on the
  blind-labelling acceptance rate and the two cross-checks in Section 3.
- **Seed variance:** the planned reruns of Run A with other seeds are not in the repository, so the spread of Run A's
  score across seeds is not reported; the paired tests compare fixed models.
- **Scope:** English only, finance and economics only, 180 test items, ±7 points on any single accuracy.
- **Deviations from the template, by decision:** vLLM replaced by `transformers.generate` (quota); DPO cut to protect
  the serving work; Eval-B dropped.

---

## 11. Reproducibility

The environment is pinned in `requirements-vm03.lock` (Python 3.12.3, torch 2.14.1+cu126, transformers 5.22.1), and
every training, evaluation and serving job refuses to start if the installed packages, the code or the data differ
from what is committed. The data files are listed with their hashes in manifests. Every document with numbers in it
(the model card, both weekly drafts, the retrieval record, the red-team log, the demo script) is generated from the
committed results, and the tests fail if one is stale.

A clean-clone check (`scripts/serve/release.py`) clones the committed state into a fresh directory, as a stranger
would get it, and checks that copy. Blocking findings include an unpinned lockfile, absolute machine paths in
configs or code, data files that differ from their manifests, modules that import only because data happens to be
present, a stale model card, and tracked secrets. On the submitted code it passes with no blocking findings: 1,481
manifest-listed data files verified by hash, and 437 tests passing inside the clone. The setup steps and the Slurm
jobs in order are in the README.
