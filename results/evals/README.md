# Evaluation results

Each folder is one `scripts/evaluate.py` run, copied from VM03 `~/evals/`: `summary.json` (metrics, settings,
provenance) and `predictions.jsonl` (one line per item). Folder name:
`<model>_<split>_<variant>_<style>_<shots>shot_<precision>`. All runs here are `"reportable": true` (clean code,
locked environment, data matching `data/v2.0/MANIFEST.txt`).

## Week 1 baselines reproduced on test (#19)

Base Qwen2.5-VL-3B, the notebook's settings (4-bit NF4, batch 4, greedy, `max_new_tokens` 384), 180 items.

| Baseline | Week 1 notebook | `evaluate.py` | Schema valid |
|---|---|---|---|
| Zero-shot | 52.8% | **52.8%** | 98.9% |
| Few-shot (2 pinned) | 57.2% | **57.2%** | 98.9% |
| Chain-of-thought | 52.2% | **52.2%** | 51.1% |

Identical, so the script and `scripts/scoring.py` reproduce the Week 1 numbers. The escaped-quote fix in
`scoring.extract_json` changes the parse of none of these outputs.

## Runs A and B on val (#7)

bf16 base + adapter, greedy, `max_new_tokens` 512, 124 items (7 unanswerable). No output hit the limit.

| Adapter | Relaxed acc. | Schema valid | Refusals | Arith./comp. (63) | Retrieval (20) | Visual ref. (34) | Tokens out |
|---|---|---|---|---|---|---|---|
| Run A r16, epoch 1 | 79.8% | 99.2% | 7/7 | 79.4% | 95.0% | 67.6% | 70.6 |
| Run A r16, final | 81.5% | 99.2% | 7/7 | 81.0% | 95.0% | 70.6% | 71.0 |
| Run B r64, epoch 1 | 82.3% | 100% | 7/7 | 77.8% | 95.0% | 79.4% | 70.6 |
| Run B r64, final | 84.7% | 100% | 6/7 | 82.5% | 90.0% | 85.3% | 70.1 |

Paired on the same items (exact McNemar): A final vs B final, 4 items only A gets right and 8 only B,
p = 0.39. Epoch 1 vs final: A p = 0.63, B p = 0.55. None of the differences is significant at n = 124
(about +/-7 to 9 points for one score). Base-model reference on test: 52.2% (CoT) to 57.2% (few-shot).
