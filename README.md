# chart-explorer
Ask a question about any chart image, get a grounded answer.

## Reproduce

Everything runs from a clone. The one thing outside it is the base model, `Qwen/Qwen2.5-VL-3B-Instruct`: download it
once into the Hugging Face cache (`huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct`), because the jobs run
offline. The environment is pinned: `requirements-vm03.lock` (Python 3.12.3, torch 2.14.1+cu126, transformers 5.22.1)
is the only supported one, and every GPU job refuses to start if the installed packages differ from it.

```bash
git clone https://github.com/milshet/chart-explorer.git && cd chart-explorer
python3.12 -m venv ~/venv && source ~/venv/bin/activate
pip install -r requirements-vm03.lock --extra-index-url https://download.pytorch.org/whl/cu126
python -m pytest -q tests                      # laptop-safe: no GPU, no model download
python scripts/serve/release.py --tests        # the clean-clone check: clones HEAD and checks it as a stranger would
```

GPU work runs as Slurm jobs driven by committed scripts (`slurm/*.sbatch`), in this order:

| Step | Job | Writes | Procedure |
|---|---|---|---|
| Train | `slurm/train.sbatch configs/run_a.yaml` | `~/runs/run_a_r16/` | [`docs/RUNS.md`](docs/RUNS.md) |
| Evaluate | `slurm/eval.sbatch --adapter ... --split val` | `results/evals/` | [`docs/RUNS.md`](docs/RUNS.md) |
| Merge and verify (4.1) | `slurm/merge.sbatch --adapter ~/runs/run_a_r16/adapter` | `~/merged/run_a_r16/`, `results/serve/merge_run_a_r16.json` | [`docs/SERVING.md`](docs/SERVING.md) |
| Serve (4.3) | `slurm/serve.sbatch` | an endpoint on `127.0.0.1:8000` | [`docs/SERVING.md`](docs/SERVING.md) |
| Latency over HTTP (4.3) | `slurm/serve_http_bench.sbatch` | `results/serve/run_a_r16_http.json` | [`docs/SERVING.md`](docs/SERVING.md) |
| Quantise (4.2) | `slurm/quantise.sbatch` (in `~/venv-quant`) | `results/serve/quantise_run_a_r16.json` | [`docs/SERVING.md`](docs/SERVING.md) |
| Red-team (4.5) | `slurm/redteam.sbatch` | `results/redteam/run_a_r16.{json,md}` | `scripts/serve/redteam.py` |

Documents with numbers in them are generated from the committed results; regenerate them, never edit them:
`MODEL_CARD.md` (`scripts/serve/modelcard.py --write`), [`docs/report_draft_week3.md`](docs/report_draft_week3.md)
(`scripts/report_week3.py --write`) and [`docs/report_draft_week4.md`](docs/report_draft_week4.md)
(`scripts/report_week4.py --write`). The tests fail when one of them is stale.

## Week 4: serving and the model card

The served model is Run A merged into bf16 weights and verified against the unmerged adapter, behind a FastAPI
endpoint (`transformers.generate`, not vLLM: [`docs/SERVING.md`](docs/SERVING.md) has the reason). What it is for,
how it was built, how it fails and when it refuses: [`MODEL_CARD.md`](MODEL_CARD.md). Week 4's numbers, and what the
week did not establish: [`docs/report_draft_week4.md`](docs/report_draft_week4.md).

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

## Experiment tracking

All training runs log to the Weights & Biases project `chartqa-week2`. The **[Weeks 2–3 report](https://forge.coreweave.com/wandb/learner01-iit-iit-kharagpur/chartqa-week2/reports/Weeks-2-3-fine-tuning-retrieval-and-evaluation-of-Qwen2-5-VL-3B-on-ChartAudit-Fin-v2-0--VmlldzoxODM0NTM5OA?accessToken=x6s5vpnsngjjlsxj8222ooqb1v1hv9hd0ckdg4anuh4sqzqqjhhyq5aq7y5d1123)** shows the
training curves of all five runs (Run A, Run B, the answer-only ablation, the table-trained Run A and the vision LoRA),
the val, test and Eval-C tables, and the Week 3 results: retrieval, the base / +RAG / fine-tuned / +RAG comparison, the
LLM judge and served latency. It is generated from the committed results by `scripts/wandb_report_week3.py`, so it
matches [`docs/report_draft_week3.md`](docs/report_draft_week3.md); rerun the script after new results are committed.
