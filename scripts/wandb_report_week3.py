#!/usr/bin/env python3
"""Build the Weeks 2-3 Weights & Biases report (task 3.16): every training run's curves plus the Week 3 results.

Nothing in it is typed: the val and test tables come from results/evals/ through wandb_report.table (the Week 2
report's own code), the Eval-C table and the summary from the committed summary.json files, and every Week 3
section (corpus, Eval-C, abstention, taxonomy, retrieval, comparison, judge, latency, open gaps) is
scripts/report_week3.py's generated draft, section for section. So the dashboard, the README and
docs/report_draft_week3.md cannot disagree, and a number not yet measured reads NOT MEASURED here too. The human
judge kappa appears as soon as results/judge/<run>/agreement.json is committed; rerun this script then.

It lives in the same W&B project as the Week 2 report (moving runs would orphan their history) and leaves that
report as it was. Same environment as wandb_report.py:

    ~/venv-report/bin/python scripts/wandb_report_week3.py            # prints the report URL
    ~/venv-report/bin/python scripts/wandb_report_week3.py --dry-run  # prints the Markdown only, no W&B access

Uses the W&B login already on the machine (~/.netrc); it never reads or prints an API key.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare  # noqa: E402
import report_week3  # noqa: E402
import wandb_report as w2  # noqa: E402

NM = report_week3.NM
ENTITY, PROJECT, REPO_URL = w2.ENTITY, w2.PROJECT, w2.REPO_URL
# W&B run name -> (label, colour): Week 2's three runs keep their colours; Week 3 adds two
RUNS = {**w2.RUNS, "run_a_r16_table": ("Run A, table-trained", "#8E44AD"),
        "run_a_r16_vision": ("Run A + vision LoRA", "#C0392B")}
RUN_A = {"test": "run_a_r16-adapter_test_image_only_cot_0shot_bf16",
         "val": "run_a_r16-adapter_val_image_only_cot_0shot_bf16",
         "eval_c": "run_a_r16-adapter_eval_c_image_only_cot_0shot_bf16"}
BASE_TEST = "base_test_image_only_answer_only_2shot_4bit"
VISION = {"test": "run_a_r16_vision-adapter_test_image_only_cot_0shot_bf16",
          "eval_c": "run_a_r16_vision-adapter_eval_c_image_only_cot_0shot_bf16"}
TABLE_RAG = "run_a_r16_table-adapter_rag_test_image_table_cot_0shot_bf16"


def draft_sections():
    """report_week3's generated draft as [(heading, body)], verbatim."""
    text = report_week3.build()
    return [(c.split("\n", 1)[0].strip(), c.split("\n", 1)[1].strip()) for c in text.split("\n## ")[1:]]


def paired(evals, a, b):
    """'x% vs y%: n vs m items, p = ...' for two evaluations on the same items, or NOT MEASURED."""
    if a not in evals or b not in evals:
        return NM
    x, y, p = compare.mcnemar(evals[a][1], evals[b][1])
    acc = lambda k: compare.pct(evals[k][0]["overall"]["relaxed_accuracy"])
    return f"{acc(a)} vs {acc(b)} ({x} vs {y} items, exact McNemar p = {p:.2g})"


def eval_c_table(evals):
    rows = ["| Model | True abstention | Paired vs Run A (only this / only Run A) |", "|---|---|---|"]
    ref = RUN_A["eval_c"]
    for name, (s, correct) in sorted(evals.items(), key=lambda kv: -kv[0].startswith("run_a_r16-")):
        a = s["overall"]["abstention"]
        k, n = round(a["true_abstention"] * a["n_unanswerable"]), a["n_unanswerable"]
        if name == ref:
            versus = "reference"
        elif ref in evals:
            x, y, p = compare.mcnemar(correct, evals[ref][1])
            versus = f"{x} / {y}, p = {p:.2g}"
        else:
            versus = NM
        rows.append(f"| {w2.label(name)} | {k}/{n} | {versus} |")
    return "\n".join(rows)


def judge_line():
    out = []
    for p in sorted((report_week3.REPO / "results/judge").glob("*/summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        x = s.get("abstention_vs_exact")
        agree = p.parent / "agreement.json"
        kappa = NM
        if agree.is_file():
            a = json.loads(agree.read_text(encoding="utf-8"))
            kap = lambda t: "undefined" if t[2] is None else t[2]
            kappa = "; ".join(f"{k} {kap(v['human_vs_human'])} between the two graders, "
                              f"{kap(v['judge_vs_grader_1'])} / {kap(v['judge_vs_grader_2'])} judge vs each"
                              for k, v in a.items())
        out.append(f"- **LLM judge on {p.parent.name}** (`{s.get('judge_model')}`): abstention verdict agrees with the "
                   f"exact answer on {x['agree']}/{x['n']} rows. Human spot-check, Cohen's κ: {kappa}." if x else
                   f"- **LLM judge on {p.parent.name}**: exact-abstention check {NM}.")
    return out or [f"- **LLM judge:** {NM}."]


def markdown():
    test, val = compare.load(compare.REPO / "results/evals", "test"), compare.load(compare.REPO / "results/evals", "val")
    rag = compare.load(compare.REPO / "results/evals", "rag_test")
    evc = compare.load(compare.REPO / "results/evals", "eval_c")
    both = {**test, **rag}
    def ec(k):
        if k not in evc:
            return NM
        a = evc[k][0]["overall"]["abstention"]
        return f"{round(a['true_abstention'] * a['n_unanswerable'])}/{a['n_unanswerable']}"
    summary = [
        "Qwen2.5-VL-3B fine-tuned with QLoRA on ChartAudit-Fin v2.0. Week 2 trained and chose the model; Week 3 added "
        "two training runs (a table-trained Run A for retrieval, and Run A with LoRA on the vision tower), retrieval "
        "over 1,168 chart tables, a 50-probe abstention set (Eval-C), an LLM judge and served latency. Every figure "
        "below is generated from the committed results; comparisons are paired on identical items.",
        "",
        f"- **Fine-tuning is the effect:** Run A vs few-shot base on test: {paired(test, RUN_A['test'], BASE_TEST)}.",
        f"- **Retrieval adds nothing on top:** table-trained Run A + RAG vs Run A: {paired(both, TABLE_RAG, RUN_A['test'])}.",
        f"- **The vision LoRA abstains better; its accuracy change is not significant:** test {paired(test, VISION['test'], RUN_A['test'])}; "
        f"Eval-C true abstention {ec(VISION['eval_c'])} vs {ec(RUN_A['eval_c'])} for Run A.",
        *judge_line(),
        "",
        f"Code, raw predictions and provenance: [{REPO_URL}]({REPO_URL}) (`docs/report_draft_week3.md`, "
        "`docs/RAG.md`, `results/`).",
    ]
    return {
        "summary": "\n".join(summary),
        "runs": (
            "Same data fingerprint, lock and recipe as Week 2 (lr 2e-4, cosine, 2 epochs, rank 16), one change each:\n\n"
            "- **Run A, table-trained** (`run_a_r16_table`, task 3.0h): trained on `image_table` prompts (chart plus its "
            "data table), so the retrieval conditions score a model on the prompt format it was trained on.\n"
            "- **Run A + vision LoRA** (`run_a_r16_vision`, task 3.0j): LoRA extended to the vision encoder blocks and the "
            "vision-language merger.\n\n"
            "Week 2's runs are shown alongside for reference. The table-trained run's loss is over longer prompts with "
            "the same targets, so compare its shape, not its level."),
        "val": "Model selection used val only; the served model (Run A) was chosen here before test or Eval-C were "
               "scored.\n\n" + w2.table("val", RUN_A["val"]),
        "test": "Image-only test set, 180 items, scored once per model. Paired column vs Run A.\n\n"
                + w2.table("test", RUN_A["test"]),
        "eval_c": "Eval-C: 50 questions the paired chart cannot answer (task 3.0c); the right behaviour is to decline. "
                  "Paired column: per-probe McNemar against Run A.\n\n" + eval_c_table(evc),
        "draft": draft_sections(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print the Markdown, do not create the report")
    args = p.parse_args()
    md = markdown()
    if args.dry_run:
        for k, v in md.items():
            if k == "draft":
                for head, body in v:
                    print(f"## {head}\n{body}\n")
            else:
                print(f"## {k}\n{v}\n")
        return

    import wandb
    import wandb_workspaces.reports.v2 as wr

    ids = {r.name: r.id for r in wandb.Api().runs(f"{ENTITY}/{PROJECT}") if r.name in RUNS}
    missing = set(RUNS) - set(ids)
    if missing:
        sys.exit(f"runs not found in {ENTITY}/{PROJECT}: {sorted(missing)}")
    names = ", ".join(f"'{n}'" for n in RUNS)
    runset = wr.Runset(entity=ENTITY, project=PROJECT, name="Weeks 2-3 runs", filters=f"Metric('Name') in [{names}]",
                       run_settings={ids[n]: wr.RunSettings(color=c) for n, (_, c) in RUNS.items()})

    def plot(key, title):
        return wr.LinePlot(title=title, x="Step", y=[key], title_x="Optimizer update")

    blocks = [
        wr.TableOfContents(),
        wr.H1("Summary"), wr.MarkdownBlock(text=md["summary"]),
        wr.H1("Training runs"), wr.MarkdownBlock(text=md["runs"]),
        wr.PanelGrid(runsets=[runset], panels=[
            plot("val/loss", "Validation loss (124 val items)"), plot("train/loss", "Training loss"),
            plot("train/lr", "Learning rate"), plot("train/grad_norm", "Gradient norm (clipped at 1.0)"),
            plot("train/seconds_per_step", "Seconds per update"), plot("train/answer_tokens", "Answer tokens per update"),
        ]),
        wr.H1("Validation (model selection)"), wr.MarkdownBlock(text=md["val"]),
        wr.H1("Held-out test set"), wr.MarkdownBlock(text=md["test"]),
        wr.H1("Abstention: Eval-C"), wr.MarkdownBlock(text=md["eval_c"]),
        wr.H1("Week 3 results (generated draft)"),
    ]
    for head, body in md["draft"]:
        blocks += [wr.H2(head), wr.MarkdownBlock(text=body)]
    report = wr.Report(
        entity=ENTITY, project=PROJECT, width="fluid",
        title="Weeks 2-3: fine-tuning, retrieval and evaluation of Qwen2.5-VL-3B on ChartAudit-Fin v2.0",
        description="All five training runs' curves; val, test and Eval-C tables; retrieval, the required comparison, "
                    "the LLM judge and served latency, generated from the committed results.",
        blocks=blocks,
    )
    report.save()
    print(report.url)


if __name__ == "__main__":
    main()
