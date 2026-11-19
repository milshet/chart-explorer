"""Retrieval corpus over the table corpus (task 3.1): every chart in the freeze, two chunking strategies.

Every one of the 1,168 tables is indexed, not only the ones an eval row points at: distractors are what make
retrieval a task. Two strategies:

  whole_table   one chunk per table: header fields, column names and every row
  row_group     groups of ROWS_PER_GROUP rows; EVERY chunk repeats the title, category axis, series names and
                column headers. A chunk of bare numbers shares no vocabulary with a question, because the words
                that connect them live in the header; without the repeat, recall goes flat in a way that looks
                like a bad retriever rather than bad chunking.

Each chunk carries its parent chart_id. Retrieval collapses chunks to tables before k is applied (retrieve.py).
"""
ROWS_PER_GROUP = 5
STRATEGIES = ("whole_table", "row_group")


def header(t):
    parts = [t.get("title") or "", t.get("category_axis") or "", " ".join(s["name"] for s in t["series"]),
             " ".join(t["columns"])]
    return "\n".join(p for p in parts if p)


def row_text(row):
    return " ".join(c for c in row if c)


def chunks(table, strategy):
    """[{"chart_id", "chunk_id", "text"}] for one corpus table."""
    head = header(table)
    if strategy == "whole_table":
        body = "\n".join(row_text(r) for r in table["rows"])
        return [{"chart_id": table["chart_id"], "chunk_id": f"{table['chart_id']}#0", "text": f"{head}\n{body}"}]
    if strategy == "row_group":
        rows = table["rows"] or [[]]
        out = []
        for i in range(0, len(rows), ROWS_PER_GROUP):
            body = "\n".join(row_text(r) for r in rows[i:i + ROWS_PER_GROUP])
            out.append({"chart_id": table["chart_id"], "chunk_id": f"{table['chart_id']}#{i // ROWS_PER_GROUP}",
                        "text": f"{head}\n{body}"})
        return out
    raise ValueError(f"strategy must be one of {STRATEGIES}")


def build(corpus, strategy):
    return [c for t in corpus for c in chunks(t, strategy)]
