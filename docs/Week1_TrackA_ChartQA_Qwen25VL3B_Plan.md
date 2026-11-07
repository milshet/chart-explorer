# Week One Deliverable: Track A, ChartQA with Qwen2.5-VL-3B

> **Programme:** IIITK Executive PG Programme in Generative AI and Agentic AI, Capstone
> **Track:** A, "Model-Side" (fine-tune and serve a small open model)
> **Working title:** **ChartAudit-VL**, a chart question-answering assistant that cites its evidence
> **Approved model:** **Qwen2.5-VL-3B-Instruct**
> **Dataset:** ChartAudit v1.0, 1,750 examples (1,403 train / 167 validation / 180 test)
> **Version:** 7 Nov 2026

**How to read this:** every dataset number comes from the actual files in `data/v1.0/` and the build logs, not from estimates. Where a number *is* an estimate (for example, a future single-domain version), it is marked **(estimate)**. Published model scores are the model authors' own figures.

---

## Summary in one table

| Item | Decision / status |
|---|---|
| Model | **Qwen2.5-VL-3B-Instruct** (approved), fine-tuned with QLoRA |
| Task | Look at a business chart image, answer a question, return JSON with the answer, unit, evidence and a short rationale; say "can't answer" when the chart doesn't contain the answer |
| Dataset | ChartAudit v1.0: 1,750 human-written ChartQA questions on 1,401 chart images |
| Splits | 1,403 train / 167 validation / 180 test, no chart shared between splits |
| Domain | Currently **multi-topic business reporting** (finance, economy, business, health). A finance-and-economy-only version is possible; numbers in Section 2 |
| Week 1 status | Dataset built and checked twice (against chart data and against the images). Baseline notebook ready. Still to do: baseline run, two-person audit, freeze |

---

## 1. The model: Qwen2.5-VL-3B-Instruct

### 1.1 What it is

Qwen2.5-VL-3B-Instruct is an open **vision-language model (VLM)** from Alibaba's Qwen team. "Vision-language" means it accepts a **picture and text together** and writes text back. You can show it a chart and ask "what is the average of the USA and France bars?".

| Fact | Value |
|---|---|
| Hugging Face ID | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Size | About 3B parameters in the language part, plus a vision encoder (about 3.8B in total) |
| Input | One or more images + text, in a chat format |
| Reported ChartQA score | About **84%** relaxed accuracy (Qwen2.5-VL technical report; full ChartQA test, human and machine questions) |
| Licence | Qwen Research Licence (fine for academic and capstone use; check before any commercial use) |

### 1.2 Why it suits this Track A project

| Track A need | How Qwen2.5-VL-3B meets it |
|---|---|
| **Model ≤ 8B** | About 3.8B in total, comfortably under the cap with no "does the vision encoder count?" question |
| **Fine-tune with QLoRA on one GPU** | In 4-bit it needs about 3 GB for the weights, so QLoRA training fits on a free/cheap Colab **T4 (16 GB)** and runs fast on an **L4 or A100**. Every team member can run their own training job, which the template asks for |
| **Real chart reading** | It reads the chart *image* directly, so colour, legend and position questions ("the dark blue bar") work without converting charts to tables first |
| **Already strong at charts** | Its reported ChartQA score (~84%) is only a few points behind the 7B version (~87%), at less than half the size |
| **Room to improve** | A small model gains most from fine-tuning on format, grounding, refusals and arithmetic, which is exactly what our dataset teaches |
| **Serving in Week 4** | Supported by vLLM and Hugging Face `transformers`; a 3B model is cheap to serve and quick to answer, which helps the latency targets |
| **Same family as the template** | Its language backbone belongs to the Qwen2.5 family that the template lists for text models |

**Trade-off to state openly:** a 3B model is a little weaker at multi-step arithmetic than a 7B one. We will measure that in the baseline and treat arithmetic as the main thing fine-tuning has to fix.

---

## 2. Dataset domain analysis: generic or domain-specific?

### 2.1 Verdict: **multi-topic within business reporting**, not a single domain

All 1,750 examples come from charts that matched a **business/economy/finance/health keyword filter** (words like *revenue, GDP, unemployment, price, hospital, mortality*). So the dataset isn't fully generic: there are no sports-only or entertainment-only charts. But it isn't a single narrow domain either, because it mixes four topics and the keyword filter is broad.

**Topic coverage of the answerable examples** (one example can have more than one topic):

| Topic | Train (of 1,340) | Val (of 160) | Test (of 171) | Typical charts |
|---|---:|---:|---:|---|
| Finance | 654 (49%) | 76 (48%) | 83 (49%) | Revenue, prices, market share, stock value, debt |
| Economy | 532 (40%) | 66 (41%) | 71 (42%) | GDP, unemployment, inflation, trade, wages |
| Business | 393 (29%) | 45 (28%) | 46 (27%) | Companies, brands, users, retail, shipments |
| Health | 256 (19%) | 37 (23%) | 31 (18%) | Mortality, disease, hospitals, life expectancy |
| **Finance *or* economy** | **1,038 (77%)** | **124 (78%)** | **136 (80%)** | |
| Health only (no other topic) | 186 (14%) | 27 (17%) | 23 (13%) | |
| More than one topic | 439 (33%) | 52 (33%) | 55 (32%) | |

So **about four in five examples are already finance or economy charts**. Health is the main "off-theme" part.

### 2.2 Limitations of the current mix

| Limitation | Why it matters |
|---|---|
| Keyword topics are approximate | A word like *market* or *cost* can tag a chart that isn't really financial; nobody reviewed topics by hand |
| Health charts (14% health-only) | They weaken the "financial reporting assistant" story if supervisors want one enterprise domain |
| Sources are mostly Statista (62%) | The style of charts is dominated by one publisher |
| No real company documents | Charts come from public statistics sites, not annual reports or filings |
| Not a regulated workflow by itself | We *frame* it as financial and economic reporting; supervisors should confirm this framing |

### 2.3 Option: make it specific to **financial and economic reporting**

If we narrow to one domain, **finance + economy** is the only realistic choice. Health alone is too small (see the table below), and finance + economy matches the reporting framing best.

**How much data is available** (ChartQA human-written questions that pass our cleaning rules, before teacher labelling):

| Domain | Train pool | Val pool | Test pool | Enough for the template (≥800 total, 150–200 test)? |
|---|---:|---:|---:|---|
| **Finance + economy** | **2,306** | **309** | **398** | **Yes, comfortably** |
| Finance only | 1,423 | 187 | 239 | Yes, but tighter after losses |
| Economy only | 1,213 | 170 | 213 | Yes, but tighter |
| Business only | 865 | 104 | 147 | Borderline; test below 150 |
| Health only | 553 | 103 | 112 | **No**: about 470 train after checks, test below 150 |

**Expected statistics for a finance + economy version, "ChartAudit-Fin v1.1" (estimate):**

| Split | Answerable | "Can't answer" (5%) | **Total** | Where the items come from |
|---|---:|---:|---:|---|
| Train | ~1,330 | ~70 | **~1,400** | Keep the 1,038 finance/economy items already checked; add ~300 new ones from the pool |
| Validation | ~160 | ~8 | **~168** | Keep 124; add ~36 |
| Test | ~171 | ~9 | **180** | Keep 136; add ~35 from the ChartQA test pool |

Expected mix (estimate, based on the finance + economy test pool): about 63% Statista, 20% OWID, 10% Pew, 7% OECD; about 37% vertical bar, 28% horizontal bar, 24% line, 11% pie; about 55% arithmetic, 24% visual reference, 21% lookup.

**Effort to switch:** about 370 new items are needed. Because 12–15% fail the checks, that means labelling about 430 candidates with the teacher model, checking them against the images, and then rebuilding the splits. About one day of work. The test set changes too, so the switch has to happen **before** the dataset is frozen and the baseline is run.

**Recommendation:** ask the supervisors which they prefer at the Week 1 review. If they want one clear enterprise domain, switch to finance + economy before freezing. Otherwise keep v1.0 and report the topic mix as above. Either way, scores can be broken down by topic, because every example carries `domain_tags`.

---

## 3. Dataset preparation: what has been done and why

### 3.1 The pipeline

```
 ChartQA (public dataset; 9,608 human-written questions on 4,804 charts)
   │
   ▼ 1. Download a fixed version, keep human-written questions only
   ▼ 2. Write each chart's data out as text (table, colours, title)
   ▼ 3. Clean: remove questions the chart can't support          ─ 8,615 pass
   ▼ 4. Topic filter: business / economy / finance / health      ─ 3,929 match
   ▼ 5. Stratified, chart-disjoint split into train / val / test
   ▼ 6. Add ~5% deliberate "can't answer" questions
   ▼ 7. Teacher model writes evidence + rationale
   ▼ 8. Automatic checker verifies every evidence item           ─ ~10% replaced
   ▼ 9. Attach the chart image to every example
   ▼ 10. Teacher checks every train/val image                     ─ 49 removed
   │
   ▼
 ChartAudit v1.0: 1,403 train / 167 val / 180 test
```

### 3.2 Each step, and why it was needed

| # | Step | What we did | Why |
|---|---|---|---|
| 1 | **Collect** | Downloaded `ChartQA Dataset.zip` from Hugging Face (`ahmed-masry/ChartQA`, fixed revision `af8b6f5`). Kept the 9,608 **human-written** questions and dropped the machine-generated ones | Human questions are harder and closer to what real users ask. A fixed version means anyone can rebuild the same data |
| 2 | **Describe each chart as text** | For every chart: source, chart type, title, which colour is which series, and the data table | Gives us an "answer key" to check evidence against, and powers the easier `image_table` variant |
| 3 | **Clean** | Removed questions whose answer isn't in the chart data, colour questions where the colour doesn't exist or is ambiguous, questions about fonts or gridlines, unreadable tables and duplicates (train: 803 of 7,398 removed) | The model can't learn from questions without a checkable answer |
| 4 | **Topic filter** | Kept charts whose title, labels or question match business, economy, finance or health keywords | Keeps the project on its financial/economic reporting theme |
| 5 | **Split** | Train and val from ChartQA's train and val; test from ChartQA's test. Each split has the same mix of source, chart type, answer type and question type | No chart appears twice, so the model can't memorise its way through the test |
| 6 | **"Can't answer" items** | ~5% of examples pair a question with the wrong chart; the right reply is `"answer": null` | Teaches the model to refuse instead of inventing a number |
| 7 | **Teacher labels** | A larger model (Claude Opus) was given the chart data, the question and the correct answer, and wrote the evidence (which cells, colours or title) and a rationale of at most 40 words. It could not change the answer | ChartQA gives only the final answer. We want the model to learn *how* to get there and to show it |
| 8 | **Automatic check** | `scripts/check_teacher.py` confirms every evidence item points at a real cell, colour or title, and that the answer matches. Items where the teacher disputed ChartQA's answer (143 of 1,378 train, 13 of 162 val) were replaced with similar items that passed | Catches mistakes in both the teacher's work and the original dataset |
| 9 | **Attach the image** | Each example points to its PNG; the model's input is *picture + question* | This is a vision dataset |
| 10 | **Image check** | The teacher looked at every train and val **picture**: can the question be answered from the picture, and does the stored answer match it? Wrong or ambiguous items were removed | The chart *data* and the *picture* can disagree. The model only sees the picture |

**What the image check found**

| | Checked | Wrong answer | Ambiguous | "Can't answer" was answerable | **Kept** |
|---|---:|---:|---:|---:|---:|
| Train | 1,449 | 20 | 17 | 9 | **1,403** |
| Validation | 170 | 0 | 2 | 1 | **167** |

16 of the 20 wrong answers had the same cause: the chart shows a percentage ("64%") and ChartQA stored a fraction (0.64). The **test set was not image-checked on purpose**, so it stays the untouched benchmark. Test errors on percentage questions will be reviewed by hand.

### 3.3 The numbers

**Size**

| | Train | Validation | Test | **Total** |
|---|---:|---:|---:|---:|
| Examples | 1,403 | 167 | 180 | **1,750** |
| Chart images | 1,104 | 140 | 157 | **1,401** |
| Share | 80% | 10% | 10% | |
| With evidence + rationale | 1,340 | 160 | answer only | |
| "Can't answer" | 63 | 7 | 9 | 79 (4.5%) |

**Chart source**

| Source | Train | Val | Test | Total | Share |
|---|---:|---:|---:|---:|---:|
| Statista | 888 | 90 | 110 | 1,088 | 62% |
| Pew Research | 257 | 17 | 19 | 293 | 17% |
| Our World in Data | 195 | 45 | 42 | 282 | 16% |
| OECD | 63 | 15 | 9 | 87 | 5% |

**Chart type**

| Type | Train | Val | Test | Total | Share |
|---|---:|---:|---:|---:|---:|
| Vertical bar | 508 | 50 | 60 | 618 | 35% |
| Horizontal bar | 428 | 62 | 58 | 548 | 31% |
| Line | 287 | 35 | 41 | 363 | 21% |
| Pie | 180 | 20 | 21 | 221 | 13% |

**Question type and answer type**

```
Question type (all 1,750)
arithmetic / comparison  ██████████████████████████  905  (52%)  "average of the USA and France bars"
visual reference         █████████████              452  (26%)  "value of the dark blue bar in 2019"
lookup (retrieval)       █████████                  314  (18%)  "what was the value in 2015?"
can't answer             ██                          79   (5%)  question paired with the wrong chart

Answer type (all 1,750)
number    █████████████████████████████████  1,165  (67%)
text      ███████████                          392  (22%)
yes / no  ███                                  114   (7%)
null      ██                                    79   (5%)
```

**Image readability (train and val, from the image check)**

| | Train | Val | Meaning |
|---|---:|---:|---|
| `yes` | 1,320 | 152 | Value printed on the chart or obvious |
| `approx` | 20 | 8 | Must be estimated from the axis |
| `no` | 63 | 7 | The "can't answer" items |

Image sizes: median 800 × 557 pixels, total about 70 MB.

---

## 4. Structure and documentation

### 4.1 Folder tree

```
data/v1.0/
├── image_only/                 ← MAIN: chart picture + question
│   ├── train.jsonl             ← 1,403 examples (evidence + rationale)
│   ├── val.jsonl               ←   167 examples (evidence + rationale)
│   ├── test.jsonl              ←   180 sealed test questions (answer only)
│   └── answer_only/            ← same examples, evidence [] and rationale null
├── image_table/                ← EASIER: picture + chart data as text + question
│   ├── train.jsonl  val.jsonl  test.jsonl
│   └── answer_only/
├── images/
│   ├── train/                  ← 1,104 PNGs  e.g. two_col_315.png
│   ├── val/                    ←   140 PNGs
│   └── test/                   ←   157 PNGs  e.g. 11832048004655.png
├── stats.json                  ← counts, image sizes, image-check results
├── teacher_stats.json          ← teacher accepted / rejected / replaced counts
└── MANIFEST.txt                ← SHA-256 fingerprint of every file
scripts/check_teacher.py        ← evidence checker (also used for scoring)
notebooks/01_vlm_baselines.ipynb← Colab notebook: baseline test of the model
docs/                           ← guides
```

### 4.2 What each file is for

| File | Contains | Used for |
|---|---|---|
| `image_only/train.jsonl` | One example per line: image path, question, full JSON answer | **Fine-tuning** |
| `image_only/val.jsonl` | Same format | Choosing settings, early stopping |
| `image_only/test.jsonl` | Same format, answer fields only | **Final score, once** |
| `*/answer_only/*.jsonl` | Same examples without evidence or rationale | Experiment: does "show your working" help? |
| `image_table/*.jsonl` | Same examples plus a `Chart data:` block | Diagnosis: is the error in *reading* the chart or in *reasoning*? |
| `images/train/two_col_315.png` | A Statista bar chart | What the model looks at |
| `stats.json` | Split sizes, answer types, image sizes, image-check counts | Numbers for reports |
| `teacher_stats.json` | Teacher results by source and reason | Explaining data quality |
| `MANIFEST.txt` | Checksums | Proving the data hasn't changed (`sha256sum -c MANIFEST.txt`) |

**Image file names** come from ChartQA: `two_col_*` and `multi_col_*` are Statista, short numbers such as `10002` are Pew, long numbers such as `001089240056` are Our World in Data, and `OECD_*` are OECD.

### 4.3 One example

**What the model sees and must answer**

```
┌─────────────────────────────┐   "In the 1990s, what is the average prize money
│  [bar chart PNG]            │ +  (in millions) in the USA and France?"
│  images/train/two_col_315   │
└─────────────────────────────┘
                 │
                 ▼  Qwen2.5-VL-3B
{"answer": 87,
 "answer_type": "number",
 "unit": "million U.S. dollars",
 "evidence": ["Row=1994 (USA), Col=Prize money in million U.S. dollars (71.0)",
              "Row=1998 (France), Col=Prize money in million U.S. dollars (103.0)"],
 "rationale": "USA 1994 was 71 and France 1998 was 103; (71 + 103) / 2 = 87."}
```

**How it is stored:** each line has a `messages` list in the standard Hugging Face chat format (system instruction, user turn with image + question, assistant turn with the JSON answer), plus labels for analysis:

| Field | Example | Meaning |
|---|---|---|
| `id` | `train-two_col_315-3611` | Unique name |
| `image` | `images/train/two_col_315.png` | Picture path |
| `question` | "In the 1990s, …" | The question |
| `gold_answer_raw` | `"87"` | ChartQA's original answer |
| `answer` | `87` | Cleaned answer |
| `source` / `chart_type` | `statista` / `v_bar` | Where the chart is from / what kind |
| `question_type` | `arithmetic_comparison` | lookup, arithmetic, visual reference, can't answer |
| `domain_tags` | `["finance"]` | Topic(s) |
| `flags.image_readable` | `yes` | Image-check result |
| `target_status` | `teacher_verified` | How the answer target was produced |

### 4.4 The answer format (JSON schema)

| Field | Values | Why it's there |
|---|---|---|
| `answer` | number, text or `null` | The result; `null` means "can't answer" |
| `answer_type` | `number`, `text`, `yes_no`, `unanswerable` | Lets us score each kind fairly |
| `unit` | e.g. `"%"`, `"million U.S. dollars"` or `null` | Avoids ambiguous numbers |
| `evidence` | list of chart elements | Audit trail: a person can check the answer in seconds |
| `rationale` | ≤ 40 words | Short reasoning, such as the arithmetic |

We use JSON because a program can score it automatically, check the evidence and count refusals reliably. Free text would make all three unreliable.

---

## 5. Next steps

### 5.1 Finish Week 1 (the gate)

| # | Task | Output |
|---|---|---|
| 1 | Supervisors decide: keep the multi-topic v1.0, or switch to finance + economy (Section 2.3) | Domain decision |
| 2 | Set `MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"` in `notebooks/01_vlm_baselines.ipynb` and run it on the 180 test items: zero-shot, then 2-shot, on `image_only`; also zero-shot on `image_table` | **Baseline scores** (the "before" numbers) |
| 3 | Two team members audit 150 random train examples (answer correct? evidence faithful? format right?) and compute their agreement (Cohen's kappa) | Quality evidence |
| 4 | Tag the repo `v1.0-frozen` | Locked data for all experiments |

### 5.2 How we will fine-tune Qwen2.5-VL-3B

```
 train.jsonl ──► load image + chat ──► Qwen2.5-VL-3B (4-bit, frozen)
                                         + small LoRA adapters (trained)
                                       │
                                       ▼  loss only on the assistant's JSON answer
                               checkpoint after each epoch
                                       │
                 val.jsonl ──► pick the best checkpoint
                                       │
                test.jsonl ──► score once ──► compare with the baseline
```

| Setting | Value (starting point) | Plain meaning |
|---|---|---|
| Method | **QLoRA**: base model loaded in 4-bit, small LoRA adapters trained | Cheap fine-tuning that fits on one GPU |
| What is trained | LoRA on the language model; the vision encoder stays frozen | We teach answering and format, not how to see |
| LoRA rank / alpha | 16 / 32 (light run), 64 / 128 (heavier run) | How big the adapters are |
| Learning rate | 2e-4 (light), 1e-4 (heavier) | How fast the model changes |
| Epochs | 2–3, with early stopping on validation | How many passes over the training data |
| Batch | 1–2 per GPU with gradient accumulation to an effective 8–16 | Fits in memory |
| Image size | Up to 800 × 800 pixels (`MAX_PIXELS`) | Labels stay readable; memory stays small |
| Loss | Only on the assistant's answer, not on the question | The model learns to *answer*, not to copy the prompt |
| Precision | fp16 on a T4, bf16 on an L4/A100 | T4 doesn't support bf16 |
| Tools | Hugging Face `transformers`, `peft`, `trl` (SFTTrainer), `bitsandbytes`; logging to W&B or MLflow | Standard, well-documented stack |

Because 3B is small, an **L4 or A100 can also train plain LoRA without 4-bit**, which is a little more accurate. That can be one of the comparison runs.

### 5.3 Week Two priorities

| Priority | Task | Why |
|---|---|---|
| 1 | **Training notebook** for Qwen2.5-VL-3B (QLoRA, settings above), tested on 20 examples first | Everything in Week 2 depends on it |
| 2 | **Run FT-1 (light) and FT-2 (heavier)**, logged to W&B/MLflow; each team member owns at least one run (different seed, rank, epochs or image size) | Template: ≥2 configurations, everyone runs a job |
| 3 | **Ablation**: train FT-1 on `answer_only/` | Shows whether evidence + rationale improve accuracy |
| 4 | **Evaluation** on the 180 test items with the same notebook metrics | A fair before/after comparison |
| 5 | **Domain adaptation** (if supervisors chose finance + economy): rebuild v1.1 *before* any training | The test set must not change after training starts |
| 6 | **Error review**: read 30–50 wrong answers, especially percentage and line-chart items | Guides Week 3 fixes |

**Evaluation metrics**

| Metric | What it measures | Target after fine-tuning |
|---|---|---|
| **Relaxed accuracy** (main) | Numbers within 5% of the correct answer, text by exact match. ChartQA's official metric | **+5 to +10 points** over the baseline |
| JSON / schema validity | The output is valid JSON with all fields | ≥ 98% |
| Evidence precision | Evidence points at a real chart element | ≥ 90% |
| Refusal F1 | Correctly says "can't answer" on the 9 can't-answer test items, without refusing answerable ones | ≥ 0.80 |
| Breakdowns | All of the above by question type, chart type, source and topic | Shows where gains come from |
| Latency / tokens per second | Speed on our serving setup | Reported in Week 3–4 |

**Where we expect gains:** the biggest improvements should be in JSON format, evidence and refusals, which the base model has never been trained to do our way. The next biggest should be in arithmetic questions. Simple lookups are already strong in the base model, so expect little change there.

### 5.4 Later weeks (for context)

| Week | Plan |
|---|---|
| 3 | Retrieval (RAG) over a 200–1,000 chart collection; compare base + RAG vs. fine-tuned vs. fine-tuned + RAG; LLM-as-judge scoring |
| 4 | Merge adapters, quantise, serve with vLLM + FastAPI in Docker; model card; 20-prompt red-team test |

---

## Glossary

| Term | Plain meaning |
|---|---|
| VLM | A model that understands pictures and text together |
| Fine-tuning | Teaching an existing model with your own examples |
| LoRA / QLoRA | Training only small add-on layers (LoRA); QLoRA also compresses the main model to 4-bit so it fits on one GPU |
| Baseline | The model's score before fine-tuning |
| Train / validation / test | Learn from / tune on / final sealed exam |
| Teacher model | A bigger AI (Claude Opus) that wrote the evidence and rationales |
| Evidence | The exact bars, cells, colours or title the answer is based on |
| Relaxed accuracy | A number counts as right if it's within 5% of the correct answer |
| Ablation | An experiment that removes one ingredient to see how much it matters |
| Freeze | Lock the dataset so every experiment uses the same data |
