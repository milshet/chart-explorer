# ChartQA VLM Capstone (Track A)

IIITK Exec PG GenAI capstone: QLoRA fine-tuning of a ≤8B **vision-language** model (Qwen2.5-VL-7B / Qwen2-VL-7B) on ChartQA human-written questions, with grounded JSON answers.

Dataset **ChartAudit v1.0**: ChartQA human-written questions selected and stratified for the project. Each item has the chart image, the question and an answer target with evidence and a short rationale, written by Claude Opus as teacher and checked by `scripts/check_teacher.py`.

## Layout

| Path | What it is |
|---|---|
| `data/v1.0/image_only/{train,val,test}.jsonl` | Chart image + question. The model has to read the chart. **Main setting.** |
| `data/v1.0/image_table/{train,val,test}.jsonl` | Chart image + the chart data as text (table and colours) + question (easier) |
| `data/v1.0/*/answer_only/` | Same items with `evidence: []`, `rationale: null` (Week 2 no-rationale ablation) |
| `data/v1.0/images/{split}/{chart_id}.png` | 1,436 ChartQA chart PNGs (72 MB), one per chart used |
| `data/v1.0/stats.json`, `teacher_stats.json`, `MANIFEST.txt` | Counts and image sizes; teacher acceptance and rejections; SHA-256 of every file |
| `scripts/check_teacher.py` | Checks that an evidence string points at a real cell, colour or title of the chart; used to verify the teacher targets and to score grounding in the notebook |
| `notebooks/01_vlm_baselines.ipynb` | Colab: zero/few-shot baseline of a 4-bit VLM on the test split |

Splits: **1,403 train / 167 val / 180 test** (test held out; never train or tune on it). Chart overlap between splits is 0. Check the files with `cd data/v1.0 && sha256sum -c MANIFEST.txt`.

## One record (`image_only`)

```json
{"id": "test-11832048004655-229", "chart_id": "11832048004655", "image": "images/test/11832048004655.png",
 "variant": "image_only", "answer": 28.3, "answer_type": "number", "input_text": "...",
 "messages": [
  {"role": "system", "content": "You are a chart analyst. Answer questions about the chart image provided.\nReturn ONLY valid JSON matching the schema. If the chart cannot answer the question,\nset \"answer\" to null and explain in \"rationale\"."},
  {"role": "user", "content": [
     {"type": "image", "image": "images/test/11832048004655.png"},
     {"type": "text", "text": "Question: What's the add up value of Lithuania and Bolivia?\n\nRespond with JSON: {...schema...}"}]},
  {"role": "assistant", "content": "{\"answer\": 28.3, \"answer_type\": \"number\", \"unit\": null, \"evidence\": [...], \"rationale\": \"...\"}"}]}
```

- `messages` follows the Hugging Face / Qwen2-VL / TRL multimodal chat format. Image paths are relative to `data/v1.0/`; load them as PIL images and replace the entry with `{"type": "image"}` for `processor.apply_chat_template` (the notebook shows how).
- `image_table` is identical except the system prompt says "the chart image and its data provided" and the text part starts with a `Chart data:` block (the table and colours as text).
- Metadata: `source`, `chart_type`, `answer_type`, `question_type`, `domain_tags`, `flags`, `target_status`, ... `input_text` is the chart's data as text; in `image_only` the model never sees it, it is only used to check the evidence.
- Evidence strings name table cells (`Row=Cocaine, Col=Maximum price (133)`), colours (`Colour: Maximum price = dark blue`) or the title, which for a chart image means category, series, colour and title.
- Unanswerable items (5%) show the chart `chart_id` with a question written for another chart (`question_source_chart`).

## Image check (train and val)

Claude Opus also looked at every train and val chart image and answered two questions per item: can it be answered from the image alone, and does the gold match what the image shows? Items where the gold was wrong or ambiguous, and "unanswerable" items the image can actually answer, were removed.

| | Checked | Gold wrong | Gold ambiguous | Refusal not valid | Kept |
|---|---:|---:|---:|---:|---:|
| Train | 1,449 | 20 | 17 | 9 | 1,403 |
| Val | 170 | 0 | 2 | 1 | 167 |

Most wrong golds (16 of 20) are a scale mistake: the chart prints a percentage such as 64% or "11 points", and the gold is the fraction 0.64 or 0.11. The test split was not checked and probably has the same pattern, so look at percentage items when you read test errors.

The result is stored per item as `flags.image_readable`: `yes` (values printed or unambiguous), `approx` (needs an estimate from the axis; 20 train, 8 val) or `no` (the unanswerable items).

## Caveats for the image-only setting

- Items were selected as answerable **from the chart's data table**. Most ChartQA charts print their values, but some line and Pew charts don't label every point, so a few `image_only` items need a value the model must estimate from the axis. The 5% relaxed-accuracy rule absorbs part of this.
- Colour names in evidence and in "what colour" answers come from a rule-based mapping of the annotation hex codes ("dark blue"), which may not match what a model or person calls the colour.
- Pew tables were noisy (garbled or missing cells); the image is often cleaner than the table, so a correct image reading can occasionally disagree with a table-derived gold. Test golds are not teacher-checked; the teacher disputed about 10% of train/val golds, so expect a similar share of wrong test golds.

## Source and licence

Charts and questions: ChartQA (Masry et al., Findings of ACL 2022), `ChartQA Dataset.zip` from `huggingface.co/datasets/ahmed-masry/ChartQA` revision `af8b6f5`. ChartQA is GPL-3.0, including its chart images; this derived dataset carries the same licence. Evidence and rationales were written by Claude Opus as the teacher model.
