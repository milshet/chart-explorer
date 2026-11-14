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

## Results (Week 2, test set)

Qwen2.5-VL-3B on the 180 held-out test items, image only, greedy decoding, the Week 1 scorer
(`scripts/scoring.py`). Every model was configured and chosen on val before test was scored, and each was
scored on test once. Paired test: exact McNemar against few-shot on the same items.

| Model | Relaxed acc. | Schema valid | Refusals | Arith./comp. (96) | Retrieval (35) | Visual ref. (40) | Tokens out | s / answer | vs few-shot |
|---|---|---|---|---|---|---|---|---|---|
| Base, zero-shot | 52.8% | 98.9% | 4/9 | 43.8% | 74.3% | 57.5% | 22 | 0.30 | p = 0.23 |
| Base, few-shot (2) | 57.2% | 98.9% | 7/9 | 42.7% | 82.9% | 65.0% | 21 | 0.52 | reference |
| Base, chain-of-thought | 52.2% | 51.1% | 6/9 | 49.0% | 68.6% | 42.5% | 145 | 1.66 | p = 0.26 |
| **QLoRA Run A, r16 (final model)** | **83.9%** | 100% | 9/9 | 78.1% | 94.3% | 85.0% | 73 | 1.18 | **+26.7 pts, p < 0.001** |
| QLoRA Run B, r64 | 83.3% | 99.4% | 9/9 | 79.2% | 91.4% | 82.5% | 74 | 1.19 | p < 0.001 |
| Ablation: r16, answer only | 75.6% | 98.3% | 8/9 | 63.5% | 100% | 80.0% | 21 | 0.39 | p < 0.001 |

- **Fine-tuning works:** Run A beats the few-shot bar by 26.7 points (53 items only Run A gets right, 5 only few-shot).
- **Rank does not matter here:** Run A vs Run B, 8 vs 7 items (p = 1.0). r16 is kept: same accuracy, adapter 114 MB vs 457 MB.
- **Showing working helps:** Run A vs the answer-only ablation, 22 vs 7 items (p = 0.008), mostly on arithmetic and comparison questions (19 vs 5). The answer-only model is 3x faster per answer.
- Base models were evaluated in 4-bit (as in Week 1), fine-tuned models in bf16 with the adapter. Margin of error for one score at n = 180: about +/-7 points.

Details, val results and raw predictions: [`results/evals/`](results/evals/README.md). Training runs: [`docs/RUNS.md`](docs/RUNS.md).
