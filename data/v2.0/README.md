# ChartAudit-Fin v2.0

Finance and economics chart questions from ChartQA for fine-tuning a vision-language model. All labels were written **blind**: the labeller never saw the correct answers.

| Split | Selected | Kept | Dropped (blind answer disagreed) |
|---|---|---|---|
| train | 1,155 | **1,070** | 85 |
| val | 137 | **126** | 11 |
| test | 180 | **180** | not labelled (human answers only) |

The splits share 0 charts.

## How it was built

1. **Raw data.** `ChartQA Dataset.zip` from `huggingface.co/datasets/ahmed-masry/ChartQA`, revision `af8b6f5c08c95085271561c2a3f9d15f2b5a9031` (zip SHA-256 `1bf310e5a51101681495c4a24f4f29d22c4f70b52df24e2e4feb0d79cae3c160`). Only the human-written questions (`*_human.json`) are used, with ChartQA's official splits.
2. **Selection by rules, no model.** `scripts/select_clean.py` (rules in `scripts/chartqa_rules.py`, seed 2026):
    - drops structural problems (no data table, duplicate or malformed question, empty answer, a train/val table identical to one in a later split, a looked-up number not present in the chart's own table, colour questions the annotation can't resolve);
    - keeps questions whose keyword domain tags include `finance` or `economy`;
    - samples proportionally over source × chart type × answer type × question type;
    - adds about 5% "unanswerable" items, where a question is paired with an unrelated chart.

   Counts for every exclusion reason are in `selection/stats.json`.
3. **Question packs.** `blind_labelling/scripts/make_packs.py` copies each chart PNG under a neutral name (`q0001.png`) next to its question. The answers go to a separate `key/` folder. `leak_check.py` fails if a pack holds anything other than `pack_id`, `image` and `question`.
4. **Blind labelling.** A fresh model instance per batch of 50 (Claude Sonnet) read only the pack and `blind_labelling/LABELLER_PROMPT.md`. It returned, in this key order: `evidence`, `rationale`, `answer_type`, `unit`, `answer`, `confidence`.
5. **Compare.** `blind_labelling/scripts/compare.py` opens the key only after labelling. The final confidence comes from agreement with the human answer:
    - numbers within 5%, or the same text: **high**;
    - numbers within 10%, or a match the labeller itself marked low: **medium**;
    - anything else: **low**, and the item is dropped.
6. **Final files.** `scripts/build_final.py` keeps high and medium items. The training target uses the blind label's evidence and rationale with the **human ChartQA answer**. Test is never labelled.

All packs, keys, labels and comparisons are in `blind_labelling/runs/`. The 40-item pilot (`pilot40`, from an earlier selection) matched on 39 of 40.

## All domains (counts only, not labelled)

`selection/all_domains_catalogue.jsonl` lists every human-written question that passes the structural checks, **in all domains**, with its `domain_tags`, a single `domain` (`finance_economy`, `health`, `business`, `other`) and `in_finance_economy`. It has no labels and is not used for training or testing. It exists so anyone can see how ChartQA splits by topic and filter on `domain` later.

| Domain (one per question) | train | val | test |
|---|---|---|---|
| finance_economy | 2,306 | 309 | 398 |
| health | 433 | 78 | 88 |
| business | 248 | 23 | 46 |
| other (no keyword match) | 3,608 | 458 | 620 |
| **total** | **6,595** | **868** | **1,152** |

Tags overlap (a chart can be both finance and economy), so the per-tag counts are larger. In train: finance 1,423, economy 1,213, business 865, health 553. Precedence for the single `domain` is finance/economy, then health, then business.

## Files

| Path | What |
|---|---|
| `image_only/{train,val,test}.jsonl` | Chart image + question → JSON with evidence and rationale first (main setting, chain-of-thought style) |
| `image_only/answer_only/` | Same items, target is only `answer_type`, `unit`, `answer` (zero-/few-shot baselines, Week 2 ablation) |
| `image_table/` | Same, plus the chart's data table as text in the prompt |
| `images/{split}/` | Original ChartQA PNGs |
| `selection/` | Output of the rule-based selection (human answers, no labels), with its own stats and manifest |
| `dropped_{train,val}.jsonl` | Items dropped because the blind answer disagreed |
| `stats.json`, `MANIFEST.txt` | Counts; SHA-256 of every output file |

Each record has:
- `domain` (`finance_economy`) and `domain_tags` (all keyword topics, e.g. `["business", "finance"]`);
- `confidence` (`high` or `medium`; `null` on test);
- `evidence_in_table_share`, the share of numbers in the evidence found in the chart's data table. The labeller never saw that table. Computed values such as sums lower the share, so it is a flag, not a filter.

## Results of the blind check

| | train | val |
|---|---|---|
| Blind answer matches human answer (high) | 1,042 (90.2%) | 123 (89.8%) |
| Close (medium) | 28 | 3 |
| Disagrees (dropped) | 85 (7.4%) | 11 (8.0%) |
| Kept items whose evidence cites numbers | 1,001 | 114 |
| …all of those numbers in the data table | 820 | 89 |
| …at least half in the data table | 901 | 99 |

Disagreements are mostly arithmetic questions (53 of 85 in train). Common causes:
- ChartQA answers that store a percentage as a fraction (0.46 for 46%);
- ambiguous questions;
- colour names (navy versus dark blue);
- mistakes by either side.

### Second model (GPT) cross-check

On a random sample of 30 train items, GPT (free ChatGPT and Microsoft Copilot web apps, same prompt, answers never shown) matched the human answer on 29 of 30. Claude matched on the same 29 and both missed the same ambiguous item, which was already dropped. Details: `blind_labelling/runs/gpt_check/RESULTS.md`.

## Limits

- The labeller ran in a shared workspace, so isolation rests on the packs, the leak check and the saved prompt, not a sandbox. The packs and outputs are kept so anyone can rerun the labelling with another model.
- ChartQA is public, so a large model may have seen it in training. Agreement checks and number grounding reduce, but don't remove, that risk.
- Dropping disagreements removes some hard questions from training. The test set is not filtered.

## Rebuild

```bash
python scripts/select_clean.py --chartqa "/path/to/ChartQA Dataset" --out data/v2.0/selection --images data/v2.0/images
# blind labelling: make_packs.py -> leak_check.py -> labeller -> compare.py (see blind_labelling/)
python scripts/build_final.py --selection data/v2.0/selection --runs blind_labelling/runs --out data/v2.0
```

Licence: ChartQA is GPL-3.0.
