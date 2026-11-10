#!/usr/bin/env python3
"""Make a three-reviewer audit workbook for a random sample of v2.0 training items.

Each reviewer gets their own sheet with the chart image, the question, the label
(evidence, rationale, answer, unit) and three Yes/No/Unsure columns to fill in.
The Summary sheet computes agreement and Fleiss' kappa with Excel formulas, and
scripts/audit_kappa.py computes the same from the filled-in file.

Usage:
  python scripts/make_audit.py --data data/v2.0 --n 50 --seed 7 --out audit/audit_v2.xlsx
"""
import argparse, io, json, os, random
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from PIL import Image

CHECKS = [
    ("answer_correct", "Is the ANSWER correct when you read the chart yourself?"),
    ("evidence_matches", "Do the EVIDENCE values really appear in the chart?"),
    ("rationale_ok", "Does the RATIONALE correctly lead to the answer?"),
]
OPTIONS = ["Yes", "No", "Unsure"]


def thumb(path, max_w=420, max_h=300):
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_w, max_h))
    buf = io.BytesIO(); im.save(buf, format="PNG"); buf.seek(0)
    return XLImage(buf), im.size


def reviewer_sheet(wb, title, items, root):
    ws = wb.create_sheet(title)
    head = ["#", "Chart", "Question", "Answer (unit)", "Evidence", "Rationale"] + [c[0] for c in CHECKS] + ["Notes", "id"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="305496")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    widths = [5, 60, 40, 18, 40, 45, 16, 16, 16, 30, 28]
    for i, w in enumerate(widths):
        ws.column_dimensions[chr(65 + i)].width = w
    dv = DataValidation(type="list", formula1='"' + ",".join(OPTIONS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    for n, r in enumerate(items, 1):
        t = json.loads(r["messages"][2]["content"])
        ans = f'{t["answer"]}' + (f' ({t["unit"]})' if t.get("unit") else "")
        row = n + 1
        ws.append([n, "", r["question"], ans, "\n".join(t.get("evidence") or []), t.get("rationale") or "",
                   "", "", "", "", r["id"]])
        img, (w, h) = thumb(os.path.join(root, r["image"]))
        ws.add_image(img, f"B{row}")
        ws.row_dimensions[row].height = max(60, h * 0.78)
        for col in "ACDEFGHIJ":
            ws[f"{col}{row}"].alignment = Alignment(wrap_text=True, vertical="top")
        for col in "GHI":
            dv.add(f"{col}{row}")
            ws[f"{col}{row}"].fill = PatternFill("solid", fgColor="FFF2CC")
    ws.freeze_panes = "C2"
    return ws


REVIEWERS = ["A", "B", "C"]


def calc_sheet(wb, n):
    """Hidden helper sheet: per item and check, how many reviewers said Yes / No / Unsure."""
    ws = wb.create_sheet("Calc")
    last = n + 1
    for i, (key, _) in enumerate(CHECKS):
        col = chr(ord("G") + i)
        base = 1 + i * 5
        for j, h in enumerate([f"{key} Yes", "No", "Unsure", "rated by", "P_i"]):
            ws.cell(row=1, column=base + j, value=h)
        for r in range(2, last + 1):
            cells = [f"'Reviewer {x}'!{col}{r}" for x in REVIEWERS]
            for j, o in enumerate(OPTIONS):
                ws.cell(row=r, column=base + j, value="=" + "+".join(f'({c}="{o}")*1' for c in cells))
            y, nn, u, k, pi = (ws.cell(row=r, column=base + j).coordinate for j in range(5))
            ws.cell(row=r, column=base + 3, value=f"={y}+{nn}+{u}")
            m = len(REVIEWERS)
            ws.cell(row=r, column=base + 4, value=f'=IF({k}={m},({y}^2+{nn}^2+{u}^2-{m})/({m}*({m}-1)),"")')
    ws.sheet_state = "hidden"
    return ws


def summary_sheet(wb, n):
    ws = wb.create_sheet("Summary", 0)
    ws["A1"] = "Audit of ChartAudit-Fin v2.0 training labels (three reviewers)"; ws["A1"].font = Font(bold=True, size=14)
    ws["A3"] = "How to fill in"; ws["A3"].font = Font(bold=True)
    steps = ["Reviewers A, B and C each fill their own sheet (or use audit/review.html). Work separately; don't look at each other's answers.",
             "For each row look at the chart image, then pick Yes / No / Unsure in the three yellow columns.",
             "answer_correct: read the chart yourself. Is the answer right? (numbers within ~5% count as right)",
             "evidence_matches: are the values quoted in Evidence really on the chart?",
             "rationale_ok: does the reasoning correctly lead to the answer?",
             "Write a short note for any No or Unsure. The numbers below fill in automatically."]
    for i, s in enumerate(steps):
        ws.cell(row=4 + i, column=1, value=f"{i + 1}. {s}")
    r0 = 11
    hdr = ["Check", "A: % Yes", "B: % Yes", "C: % Yes", "All 3 agree %", "Fleiss' kappa"]
    for j, h in enumerate(hdr):
        c = ws.cell(row=r0, column=1 + j, value=h); c.font = Font(bold=True)
    last = n + 1
    m = len(REVIEWERS)
    for i, (key, desc) in enumerate(CHECKS):
        col = chr(ord("G") + i)
        row = r0 + 1 + i
        ws.cell(row=row, column=1, value=key)
        for j, x in enumerate(REVIEWERS):
            R = f"'Reviewer {x}'!{col}2:{col}{last}"
            ws.cell(row=row, column=2 + j, value=f'=IFERROR(COUNTIF({R},"Yes")/COUNTA({R}),"")')
        base = 1 + i * 5
        L = lambda j: chr(64 + base + j)
        Y, N_, U, K, P = (f"Calc!{L(j)}2:{L(j)}{last}" for j in range(5))
        full = f"COUNTIF({K},{m})"
        allagree = f'SUMPRODUCT(({K}={m})*(({Y}={m})+({N_}={m})+({U}={m})))'
        ws.cell(row=row, column=5, value=f'=IFERROR({allagree}/{full},"")')
        pj = lambda R: f"(SUMPRODUCT(({K}={m})*{R})/({m}*{full}))"
        pe = "+".join(f"{pj(R)}^2" for R in (Y, N_, U))
        ws.cell(row=row, column=6, value=f'=IFERROR((AVERAGE({P})-({pe}))/(1-({pe})),"")')
        for c in "BCDE":
            ws[f"{c}{row}"].number_format = "0.0%"
        ws[f"F{row}"].number_format = "0.00"
    ws.cell(row=r0 + 5, column=1, value="Fleiss' kappa = agreement among all three reviewers beyond chance. Above 0.8 almost perfect, "
            "0.6-0.8 substantial, 0.4-0.6 moderate. If almost everything is 'Yes', kappa can look low even when agreement "
            "is high; report agreement % too. scripts/audit_kappa.py also prints Cohen's kappa for each pair.")
    ws.column_dimensions["A"].width = 22
    for c in "BCDEF":
        ws.column_dimensions[c].width = 18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v2.0")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="audit/audit_v2.xlsx")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.data, "image_only", "train.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["answer_type"] != "unanswerable"]
    items = random.Random(a.seed).sample(rows, a.n)
    wb = Workbook(); wb.remove(wb.active)
    for x in REVIEWERS:
        reviewer_sheet(wb, f"Reviewer {x}", items, a.data)
    summary_sheet(wb, a.n)
    calc_sheet(wb, a.n)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    wb.save(a.out)
    with open(os.path.splitext(a.out)[0] + "_ids.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(r["id"] for r in items) + "\n")
    print(f"wrote {a.out} with {a.n} items")


if __name__ == "__main__":
    main()
