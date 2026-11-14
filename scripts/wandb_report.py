#!/usr/bin/env python3
"""Build the Week 2 Weights & Biases report (#8): training curves of the runs plus the val and test tables.

The tables are generated from results/evals/ (the committed evaluations), so the report and the README
show the same numbers. Needs `wandb-workspaces`, which is not part of the locked training environment;
on VM03 it lives in its own venv so the training packages stay untouched:

    python3 -m venv ~/venv-report && ~/venv-report/bin/pip install wandb-workspaces
    ~/venv-report/bin/python scripts/wandb_report.py            # prints the report URL
    ~/venv-report/bin/python scripts/wandb_report.py --dry-run  # prints the Markdown only

Uses the W&B login already on the machine (~/.netrc); it never reads or prints an API key.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare  # noqa: E402  load() and mcnemar() over results/evals

ENTITY, PROJECT = "learner01-iit-iit-kharagpur", "chartqa-week2"
REPO_URL = "https://github.com/milshet/chart-explorer"
# W&B run name -> (label, colour); the runs compared in Week 2.
RUNS = {"run_a_r16": ("Run A, r16 (final)", "#D9731A"), "run_b_r64": ("Run B, r64", "#1F5BC4"),
        "ablation_answer_only": ("Ablation, answer only", "#5B8C5A")}
LABELS = {  # evaluation folder prefix -> table label
    "base_test_image_only_answer_only_0shot": "Base, zero-shot", "base_test_image_only_answer_only_2shot": "Base, few-shot (2)",
    "base_test_image_only_cot_0shot": "Base, chain-of-thought",
    "run_a_r16-adapter": "QLoRA Run A, r16 (final)", "run_a_r16-epoch-1": "Run A, r16, epoch 1",
    "run_b_r64-adapter": "QLoRA Run B, r64", "run_b_r64-epoch-1": "Run B, r64, epoch 1",
    "ablation_answer_only-adapter": "Ablation, answer only",
}


def label(folder):
    return next((v for k, v in LABELS.items() if folder.startswith(k)), folder)


def table(split, reference):
    evals = compare.load(compare.REPO / "results" / "evals", split)
    ref = evals[reference][1]
    rows = ["| Model | Relaxed acc. | Schema valid | Refusals | Arith./comp. | Retrieval | Visual ref. | s / answer | Paired vs " +
            label(reference) + " |", "|---|---|---|---|---|---|---|---|---|"]
    order = list(LABELS)
    rank = lambda name: next((i for i, k in enumerate(order) if name.startswith(k)), len(order))
    for name, (s, correct) in sorted(evals.items(), key=lambda kv: rank(kv[0])):
        o, by = s["overall"], s["by"]["question_type"]
        u = by.get("unanswerable", {"mean": 0, "count": 0})
        if name == reference:
            paired = "reference"
        else:
            x, y, p = compare.mcnemar(correct, ref)
            paired = f"{x} vs {y} items, p = {p:.2g}"
        bold = "**" if "final" in label(name) else ""
        rows.append(f"| {bold}{label(name)}{bold} | {bold}{compare.pct(o['relaxed_accuracy'])}{bold} | "
                    f"{compare.pct(o['schema_valid'])} | {round(u['mean'] * u['count'])}/{u['count']} | "
                    + " | ".join(compare.pct(by[q]["mean"]) if q in by else "" for q in compare.QTYPES)
                    + f" | {o['seconds_per_item']} | {paired} |")
    return "\n".join(rows)


def markdown():
    test = table("test", "base_test_image_only_answer_only_2shot_4bit")
    val = table("val", "run_a_r16-adapter_val_image_only_cot_0shot_bf16")
    return {
        "summary": (
            "Qwen2.5-VL-3B fine-tuned with QLoRA on ChartAudit-Fin v2.0 (1,033 train / 124 val / 180 test finance and "
            "economics chart questions). **On the 180 held-out test items, the final model (Run A, rank 16) scores 83.9% "
            "relaxed accuracy against 57.2% for the best Week 1 baseline (few-shot)**: 53 items only the fine-tuned model "
            "gets right, 5 only the baseline (exact McNemar p < 0.001).\n\n"
            "- **Rank does not matter here:** Run A (r16) and Run B (r64) are a tie on test (8 vs 7 items, p = 1.0); r16 is kept, "
            "with a 4x smaller adapter.\n"
            "- **Showing working helps:** Run A beats the answer-only ablation, 22 vs 7 items (p = 0.008), mostly on "
            "arithmetic and comparison questions. Answer-only is about 3x faster per answer.\n"
            "- Every model was configured and chosen on val before test was scored, and each was scored on test once.\n\n"
            f"Code, raw predictions and provenance: [{REPO_URL}]({REPO_URL}) (`README.md`, `results/evals/`, `docs/RUNS.md`)."),
        "setup": (
            "- **Model:** Qwen2.5-VL-3B-Instruct in 4-bit NF4; LoRA on the language model's attention and MLP layers; vision "
            "encoder frozen; loss on the answer tokens only.\n"
            "- **Shared settings:** lr 2e-4, cosine with 3% warm-up, batch 1 x 8 gradient accumulation, 2 epochs = 260 updates, "
            "validation on the 124 val items every 50 updates and at each epoch end.\n"
            "- **Runs:** Run A rank 16 / alpha 32; Run B rank 64 / alpha 128 (rank is the only change); ablation = Run A with "
            "answer-only targets (no evidence or rationale, about 23 instead of 81 answer tokens).\n"
            "- **Cost on one L40S:** about 31 min per run, 6.5 s per update, peak GPU memory 5.9-7.7 GB.\n"
            "- **Reproducible:** each run refused to start unless the data matched `data/v2.0/MANIFEST.txt`, the code was committed "
            "and the packages matched `requirements-vm03.lock`; every run records this in `provenance.json`."),
        "curves": (
            "Validation loss drops fast in epoch 1 and barely moves in epoch 2 (Run A 0.3725 -> 0.3654, Run B 0.3613 -> 0.3614), so "
            "2 epochs were enough. Rank 64 trained stably at lr 2e-4, so the planned rerun at 1e-4 was not needed. The ablation's "
            "loss is over much shorter targets, so its scale is not comparable with Runs A and B."),
        "val": ("Adapters in bf16, greedy decoding, up to 512 new tokens (no output was cut off). Model selection used val "
                "only. Run B leads Run A by 3.2 points, but item by item it is 8 vs 4 (p = 0.39): a tie at n = 124, so the "
                "cheaper run (Run A) was chosen, as planned in issue #7.\n\n" + val),
        "test": ("Base models in 4-bit (as in Week 1, so their numbers reproduce exactly), fine-tuned models in bf16 with the "
                 "adapter. One score is about +/-7 points at n = 180; the paired column compares the same items.\n\n" + test),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print the Markdown, do not create the report")
    args = p.parse_args()
    md = markdown()
    if args.dry_run:
        for k, v in md.items():
            print(f"## {k}\n{v}\n")
        return

    import wandb
    import wandb_workspaces.reports.v2 as wr

    ids = {r.name: r.id for r in wandb.Api().runs(f"{ENTITY}/{PROJECT}") if r.name in RUNS}
    missing = set(RUNS) - set(ids)
    if missing:
        sys.exit(f"runs not found in {ENTITY}/{PROJECT}: {sorted(missing)}")
    names = ", ".join(f"'{n}'" for n in RUNS)
    runset = wr.Runset(entity=ENTITY, project=PROJECT, name="Week 2 runs", filters=f"Metric('Name') in [{names}]",
                       run_settings={ids[n]: wr.RunSettings(color=c) for n, (_, c) in RUNS.items()})

    def plot(key, title, y_range=(None, None)):
        return wr.LinePlot(title=title, x="Step", y=[key], range_y=y_range, title_x="Optimizer update")

    report = wr.Report(
        entity=ENTITY, project=PROJECT, width="fluid",
        title="Week 2: QLoRA fine-tuning of Qwen2.5-VL-3B on ChartAudit-Fin v2.0",
        description="Runs A (r16), B (r64) and the answer-only ablation: training curves, model choice on val, final test results.",
        blocks=[
            wr.TableOfContents(),
            wr.H1("Summary"), wr.MarkdownBlock(text=md["summary"]),
            wr.H1("Setup"), wr.MarkdownBlock(text=md["setup"]),
            wr.H1("Training curves"), wr.MarkdownBlock(text=md["curves"]),
            wr.PanelGrid(runsets=[runset], panels=[
                plot("val/loss", "Validation loss (124 val items)"), plot("train/loss", "Training loss"),
                plot("train/lr", "Learning rate"), plot("train/grad_norm", "Gradient norm (clipped at 1.0)"),
                plot("train/seconds_per_step", "Seconds per update"), plot("train/answer_tokens", "Answer tokens per update"),
            ]),
            wr.H1("Choosing the model on val"), wr.MarkdownBlock(text=md["val"]),
            wr.H1("Results on the held-out test set"), wr.MarkdownBlock(text=md["test"]),
        ],
    )
    report.save()
    print(report.url)


if __name__ == "__main__":
    main()
