# chart-explorer
Ask a question about any chart image, get a grounded answer.

## Dataset

**ChartAudit-Fin v2.0** (`data/v2.0/`): 1,033 train / 124 val / 180 test finance and economics chart questions from ChartQA, selected by written rules and labelled blind (the labeller never saw the correct answers). Method, results and rebuild steps: [`data/v2.0/README.md`](data/v2.0/README.md).

| Folder | What |
|---|---|
| `data/v2.0/` | Current dataset (image_only, image_table, answer_only variants, images, selection) |
| `scripts/select_clean.py`, `scripts/chartqa_rules.py` | Rule-based selection from raw ChartQA |
| `blind_labelling/` | Labeller prompt, packs, hidden answer keys, labels and comparisons (incl. a GPT cross-check) |
| `scripts/build_final.py` | Builds the final files from the selection and the blind labels |
| `audit/` | 50-item three-reviewer human audit (review page `audit/review.html`, Fleiss' kappa) |
| `notebooks/01_vlm_baselines.ipynb` | Zero-shot, few-shot and chain-of-thought baselines (Qwen2.5-VL-3B) |
| `data/v1.0/` | Earlier version, superseded by v2.0 |
