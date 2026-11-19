"""Query construction from the chart (task 3.2), with a leak gate.

Two plans, measured as a bracket rather than a winner:

  question_only    the user's question: the floor, what serving has with no reading of the chart at all
  chart_metadata   the question plus the chart's printed labels: title, category axis, series (legend) names,
                   column headers and row labels. This is what perfect OCR of the chart's text would give; a
                   deployed pipeline lands between the two.

The gate: a query the team builds must not contain a value only the gold table knows. If it does, recall@k goes
to 1.0, RAG looks transformative in task 3.5, and none of it survives serving, where there is no table to build
a query from. Row and column labels are exempt (they are the axis ticks and the legend, printed on the chart);
the data cells are not. The gate raises only on the chart-side text the team writes. A ChartQA *question* that
quotes one of its own table's values is counted and reported beside recall (question_value_leaks), never raised
on: nobody on the team wrote it, and users at serving time ask exactly such questions.
"""
import re

PLANS = ("question_only", "chart_metadata")


class LeakError(ValueError):
    pass


def numbers_in(text):
    return {float(x.replace(",", "")) for x in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", text)}


def label_and_value_numbers(table):
    """(numbers printed as labels, numbers that are data cells only)."""
    labels = set()
    for s in [table["columns"], [r[0] for r in table["rows"] if r[0]], [s["name"] for s in table["series"]],
              [table.get("title") or "", table.get("category_axis") or ""]]:
        for x in s:
            labels |= numbers_in(str(x))
    values = set()
    for r in table["rows"]:
        for c in r[1:]:
            if c:
                values |= numbers_in(str(c).rstrip("%"))
    return labels, values - labels


def chart_side(table):
    """The chart-side text a chart_metadata query adds: only what is printed as a label on the chart."""
    parts = [table.get("title") or "", table.get("category_axis") or "",
             " ".join(s["name"] for s in table["series"]), " ".join(table["columns"]),
             " ".join(r[0] for r in table["rows"] if r[0])]
    return " ".join(p for p in parts if p)


def gate(team_text, table):
    """Raise LeakError if text the team wrote carries a data value of the gold table."""
    _, values = label_and_value_numbers(table)
    leaked = numbers_in(team_text) & values
    if leaked:
        raise LeakError(f"query carries data values of the gold table: {sorted(leaked)[:5]}")


def build(question, plan, table=None):
    """(query text, question_value_leak) for one row. table is the chart the user is looking at."""
    if plan not in PLANS:
        raise ValueError(f"plan must be one of {PLANS}")
    leak = False
    if table is not None:
        _, values = label_and_value_numbers(table)
        leak = bool(numbers_in(question) & values)
    if plan == "question_only":
        return question, leak
    side = chart_side(table)
    gate(side, table)
    return f"{question}\n{side}", leak
