"""Rule library for selecting ChartQA items (no model involved).

Deterministic, standard-library helpers used by select_clean.py:
  * question typing (retrieval / visual_reference / arithmetic_comparison) by regex,
  * domain tagging by keyword lexicon over the chart's table header, row labels, title,
    y-axis label and the question,
  * structural sanity filters that check each human answer against the chart's own
    data table and annotation (e.g. a looked-up number must exist in the table),
  * stratified sampling and construction of "unanswerable" items (a question paired
    with an unrelated chart).

Source files: "ChartQA Dataset.zip" from huggingface.co/datasets/ahmed-masry/ChartQA.
"""
import argparse
import collections
import colorsys
import csv
import difflib
import hashlib
import json
import os
import random
import re

UNANSWERABLE_RATE = 0.05
STRATA = ("source", "chart_type", "answer_type", "question_type")

CHART_TYPE_TEXT = {"v_bar": "vertical bar", "h_bar": "horizontal bar", "line": "line", "pie": "pie"}

# --------------------------------------------------------------------------- lexicons
COLOUR_WORDS = (r"red|blue|green|yellow|orange|purple|pink|grey|gray|black|brown|violet|navy|maroon|"
                r"cyan|teal|magenta|turquoise|olive|beige|golden|gold|silver|white|lilac|peach|lime|"
                r"tan|khaki|crimson|scarlet|mustard")
COLOUR = re.compile(r"\b(" + COLOUR_WORDS + r"|colou?rs?|colou?red)\b", re.I)
STRUCTURE = re.compile(
    r"\b(bars?|lines?|slices?|segments?|sections?|pies?|graphs?|plots?|legends?|leftmost|"
    r"rightmost|topmost|bottom|left|right|top|first|last|tallest|shortest|longest|"
    r"smallest|largest|biggest|highest|lowest|peak|x-axis|y-axis|axis|dotted|dashed)\b", re.I)
ARITH = re.compile(
    r"\b(difference|differ|sum|summation|addition|total|add|added|average|mean|median|ratio|times|divide|divided|"
    r"multipl\w*|product|subtract\w*|minus|plus|more than|less than|greater|fewer|higher than|"
    r"lower than|compare|percent(age)? change|increase|decrease|how many|count|gap|twice|half|"
    r"exceed\w*|combined|altogether|together)\b", re.I)

# A colour phrase: optional modifier + colour word ("light blue", "navy blue", "navy", "maroon").
COLOUR_PHRASE = re.compile(
    r"\b(?:(light|dark|navy|sky|pale|bright|deep|lighter|darker)[\s-]+)?(" + COLOUR_WORDS + r")\b", re.I)
# Questions whose answer is a colour, or that ask how many colours there are.
COLOUR_ASK = re.compile(
    r"(what|which)\s+(is\s+the\s+)?colou?r|colou?r\s+(is|of|does)|how many colou?rs", re.I)
TITLE_ASK = re.compile(r"\btitle\b", re.I)
LEGEND = re.compile(r"\blegend\b", re.I)
# Still unanswerable from the serialisation even with annotations.
APPEARANCE_OTHER = re.compile(r"\bgridlines?\b|\bdotted\b|\bdashed\b|\bfont\b|\bshaded?\b|\bshades?\b", re.I)

DOMAIN = {
    "finance": r"revenue|sales|profit|market share|market|price|stock|investment|invest|bank|debt|budget|"
               r"dollar|euro|usd|\$|€|£|financ|earning|asset|capital|interest rate|loan|credit",
    "economy": r"gdp|inflation|unemploy|trade|export|import|income|spending|spend|tax|wage|salary|"
               r"econom|poverty|employ|labou?r|cost|value added|consumer price|per capita",
    "health":  r"hospital|health|mortality|death|vaccin|insurance|disease|covid|life expectancy|patient",
    "business": r"compan|brand|retail|consumer|advertis|employee|industry|revenue|shipment|"
                r"subscriber|user|store|firm|enterprise|startup|manufactur",
}
DOMAIN_RE = {k: re.compile(v, re.I) for k, v in DOMAIN.items()}

UNIT_PATTERNS = [
    (re.compile(r"\bin (billion|million|thousand|trillion)s? ([\w.\s$€£]+?)(\)|$|,)", re.I), None),
    (re.compile(r"\((%|percent|percentage)\)", re.I), "%"),
    (re.compile(r"\b(share|percentage|percent|rate)\b|%", re.I), "%"),
    (re.compile(r"\bin (u\.s\. dollars|us dollars|dollars|euros|pounds|yen|people|tons|"
                r"metric tons|hours|years|days|minutes|kilograms|kg|km|kilometers|degrees)\b", re.I), None),
]

# --------------------------------------------------------------------------- colours
# Colour words people use in questions -> (canonical family, implied modifier).
WORD_FAMILY = {
    "red": ("red", None), "crimson": ("red", None), "scarlet": ("red", None), "maroon": ("red", "dark"),
    "orange": ("orange", None), "peach": ("orange", "light"),
    "yellow": ("yellow", None), "gold": ("yellow", None), "golden": ("yellow", None), "mustard": ("yellow", None),
    "green": ("green", None), "lime": ("green", "light"), "olive": ("green", "dark"),
    "teal": ("teal", None), "turquoise": ("teal", None), "cyan": ("teal", None),
    "blue": ("blue", None), "navy": ("blue", "dark"),
    "purple": ("purple", None), "violet": ("purple", None), "lilac": ("purple", "light"),
    "pink": ("pink", None), "magenta": ("pink", None),
    "brown": ("brown", None), "beige": ("brown", "light"), "tan": ("brown", "light"), "khaki": ("brown", "light"),
    "grey": ("grey", None), "gray": ("grey", None), "silver": ("grey", "light"),
    "black": ("black", None), "white": ("white", None),
}
MODIFIER = {"light": "light", "lighter": "light", "sky": "light", "pale": "light", "bright": "light",
            "dark": "dark", "darker": "dark", "navy": "dark", "deep": "dark"}
# Families a reader might plausibly use for a colour of the key family (used only when no
# colour in the chart is a direct match).
NEIGHBOURS = {
    "red": {"pink", "brown", "orange"}, "orange": {"red", "brown", "yellow"}, "yellow": {"orange"},
    "green": {"teal"}, "teal": {"green", "blue"}, "blue": {"teal", "purple"}, "purple": {"pink", "blue"},
    "pink": {"purple", "red"}, "brown": {"red", "orange"}, "grey": {"black", "white"},
    "black": {"grey"}, "white": {"grey"},
}
PEW_TYPOS = {"dark drown": "dark brown", "light brownn": "light brown", "dark brownn": "dark brown",
             "light brow": "light brown", "brow": "brown", "dark greenn": "dark green",
             "light brown.": "light brown", "hrey": "grey", "orannnge": "orange", "orage": "orange",
             "lemon": "yellow"}


def parse_hex(c):
    c = str(c).strip().lower()
    m = re.fullmatch(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", c)
    if m:
        return tuple(int(x) for x in m.groups())
    m = re.fullmatch(r"#([0-9a-f]{6})", c)
    if m:
        return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    return None


def describe_colour(raw):
    """Map an annotation colour (hex, rgb() or a Pew colour name) to
    {name, family, lightness, extra}. Returns None for unknown colours (Pew "UNK")."""
    s = str(raw).strip().lower()
    s = PEW_TYPOS.get(s, s)
    rgb = parse_hex(s)
    if rgb is None:
        m = re.fullmatch(r"(?:(light|dark)\s+)?([a-z]+)", s)
        if not m or m.group(2) not in WORD_FAMILY:
            return None
        fam, implied = WORD_FAMILY[m.group(2)]
        mod = m.group(1) or implied
        light = {"light": 0.72, None: 0.5, "dark": 0.28}[mod]
        return {"name": (mod + " " if mod else "") + fam, "family": fam, "lightness": light, "extra": set()}
    r, g, b = (x / 255 for x in rgb)
    h, l, sat = colorsys.rgb_to_hls(r, g, b)
    h *= 360
    extra = set()
    if l >= 0.93:
        fam, mod = "white", None
    elif l <= 0.12 or (sat < 0.15 and l < 0.2):
        fam, mod = "black", None
    elif sat < 0.15:
        fam = "grey"
        mod = "light" if l >= 0.68 else ("dark" if l < 0.38 else None)
    else:
        if h < 18 or h >= 345:
            fam = "red"
        elif h < 40:
            fam = "brown" if l < 0.3 else "orange"
        elif h < 70:
            fam = "yellow"
        elif h < 165:
            fam = "green"
        elif h < 192:
            fam = "teal"
        elif h < 258:
            fam = "blue"
        elif h < 300:
            fam = "purple"
        else:
            fam = "pink"
        mod = "light" if l >= 0.65 else ("dark" if l < 0.3 or (sat < 0.35 and l < 0.4) else None)
        if sat < 0.35:
            extra.add("grey")
        if fam == "blue" and l < 0.22:
            extra.add("black")
        if fam == "red" and l < 0.4:
            extra.add("brown")
        if fam == "orange" and l < 0.42:
            extra |= {"red", "brown"}
    return {"name": (mod + " " if mod else "") + fam, "family": fam, "lightness": l, "extra": extra}


def colour_phrases(text):
    """Colour phrases in a question as (modifier, family) pairs, in order."""
    out = []
    for mod, word in COLOUR_PHRASE.findall(text):
        fam, implied = WORD_FAMILY[word.lower()]
        out.append((MODIFIER.get(mod.lower()) if mod else implied, fam))
    return out


def resolve_phrase(phrase, colours):
    """Return the indices of distinct chart colours a phrase can refer to (empty = no match,
    more than one = ambiguous). `colours` is a list of described colours, all distinct by name."""
    mod, fam = phrase
    cand = [i for i, c in enumerate(colours) if c["family"] == fam]
    if not cand:
        cand = [i for i, c in enumerate(colours) if fam in c["extra"]]
    if not cand:
        cand = [i for i, c in enumerate(colours) if fam in NEIGHBOURS.get(c["family"], ())]
    if len(cand) <= 1:
        return cand
    names = [colours[i]["name"] for i in cand]
    want = (mod + " " if mod else "") + fam
    exact = [i for i in cand if colours[i]["name"] == want]
    if len(exact) == 1:
        return exact
    if mod in ("light", "dark"):
        ls = sorted(cand, key=lambda i: colours[i]["lightness"])
        lo, hi = colours[ls[0]]["lightness"], colours[ls[-1]]["lightness"]
        if hi - lo >= 0.08:
            pick = ls[-1] if mod == "light" else ls[0]
            # The extreme must be clearly apart from the next one.
            nxt = ls[-2] if mod == "light" else ls[1]
            if abs(colours[pick]["lightness"] - colours[nxt]["lightness"]) >= 0.06:
                return [pick]
    if mod is None:
        plain = [i for i in cand if colours[i]["name"] == fam]
        if len(plain) == 1:
            return plain
    del names
    return cand


# --------------------------------------------------------------------------- helpers
def source_of(name):
    if name.startswith(("two_col", "multi_col")):
        return "statista"
    if name.startswith("OECD"):
        return "oecd"
    if re.fullmatch(r"\d{14}", name):
        return "owid"
    return "pew"


def read_table(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if len(rows) < 2:
        return None
    header, body = rows[0], rows[1:]
    width = len(header)
    body = [(r + [""] * width)[:width] for r in body]
    return header, body


def to_number(s):
    s = str(s).strip().replace(",", "").replace("%", "").replace("$", "").replace("€", "").replace("£", "")
    s = s.rstrip(".").strip()
    if s.lower().startswith("*"):
        s = s.lstrip("*")
    try:
        v = float(s)
    except ValueError:
        return None
    return v


def numeric_cells(body):
    return [to_number(c) for r in body for c in r[1:] if to_number(c) is not None]


def norm_text(s):
    s = str(s).lower().strip().rstrip(".")
    s = re.sub(r"[\"'`’*]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def normalise_answer(label):
    raw = str(label).strip()
    low = raw.lower().rstrip(".")
    if low in ("yes", "no"):
        return low, "yes_no"
    v = to_number(raw)
    if v is not None and re.fullmatch(r"[\s\-+\d.,%$€£*]+", raw):
        return (int(v) if v == int(v) and "." not in raw.replace(",", "") else v), "number"
    return norm_text(raw), "text"


def md_cell(c):
    return str(c).replace("|", "/").replace("\n", " ").strip()


def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def best_label(name, labels):
    """Map an annotation series/slice name to the table's spelling when they clearly match."""
    name = clean(name)
    if not labels:
        return name
    low = {clean(l).lower(): clean(l) for l in labels if clean(l)}
    if name.lower() in low:
        return low[name.lower()]
    m = difflib.get_close_matches(name.lower(), list(low), n=1, cutoff=0.8)
    return low[m[0]] if m else name


# --------------------------------------------------------------------------- annotations
def load_annotation(path, header, body):
    """Extract what the serialisation needs from one annotation JSON:
    chart_type, title, y-axis label, per-series colours, per-mark colours."""
    d = json.load(open(path, encoding="utf-8"))
    info = d.get("general_figure_info") or {}
    title = clean((info.get("title") or {}).get("text") or "") or None
    yl = (info.get("y_axis") or {}).get("label") or {}
    y_label = clean(yl.get("text") or "") if isinstance(yl, dict) else ""
    ctype = d.get("type")
    models = d.get("models") or []
    cols = [clean(h) for h in header[1:]]
    rows = [clean(r[0]) for r in body]
    series, marks = [], []   # series: (label, colour_raw); marks: (series_label|None, mark_label, colour_raw)
    if ctype == "pie":
        labels = [m.get("text_label", m.get("name")) for m in models]
        if any(l is None for l in labels):
            labels = rows if len(rows) == len(models) else [None] * len(models)
        for m, lab in zip(models, labels):
            if lab is not None:
                marks.append((None, best_label(lab, rows), m.get("color")))
    else:
        for m in models:
            nm = m.get("name")
            if not isinstance(nm, str) or nm.strip() in ("bars", "line", "", "{}"):
                nm = cols[0] if len(cols) == 1 and len(models) == 1 else None
            label = best_label(nm, cols + rows) if nm else None
            c = m.get("colors", m.get("color"))
            if isinstance(c, list):
                xs = m.get("x") or m.get("labels") or []
                uniq = {str(v).lower() for v in c}
                if len(uniq) == 1:
                    series.append((label, c[0]))
                else:
                    for x, v in zip(xs, c):
                        marks.append((label, best_label(str(x), rows + cols), v))
            else:
                series.append((label, c))
    return {"chart_type": ctype, "title": title, "y_label": y_label or None,
            "series": series, "marks": marks, "n_models": len(models),
            "mark_colours_unreliable": mark_colours_unreliable(ctype, models, marks, header, body)}


def mark_colours_unreliable(ctype, models, marks, header, body):
    """True when the per-bar / per-slice colour list can't be trusted. Found after v1.0 was sampled
    (so it only feeds the --pool-out dump and add_teacher_labels.py, leaving the sample unchanged):
      * pies whose annotation repeats a slice label or has a different number of slices than the table;
      * bar charts whose annotation x labels are not in the same order as the y values (about half of
        the OECD bar charts): the colours follow the y values, so pairing them with x mislabels bars."""
    if not marks:
        return False
    rows = [clean(r[0]) for r in body]
    if ctype == "pie":
        labels = [m for _, m, _ in marks]
        return len(labels) != len(set(labels)) or len(labels) != len(rows)
    lookup = {clean(r[0]).lower(): r for r in body}
    for m in models:
        cs = m.get("colors")
        if not isinstance(cs, list) or len({str(x).lower() for x in cs}) <= 1:
            continue
        for x, y in zip(m.get("x") or [], m.get("y") or []):
            r = lookup.get(best_label(str(x), rows).lower())
            yv = to_number(y)
            if r is None or yv is None:
                continue
            if not any(to_number(v) is not None and abs(to_number(v) - yv) < 1e-6 for v in r[1:]):
                return True
    return False


def chart_colours(ann):
    """Distinct described colours in a chart, plus whether any colour is unknown."""
    seen, out, unknown = set(), [], False
    for raw in [c for _, c in ann["series"]] + [c for _, _, c in ann["marks"]]:
        d = describe_colour(raw)
        if d is None:
            unknown = True
            continue
        key = str(raw).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        # Two raw colours with the same name are indistinguishable to a reader: merge them.
        same = [o for o in out if o["name"] == d["name"]]
        if same:
            same[0]["raws"].add(key)
            continue
        d["raws"] = {key}
        out.append(d)
    return out, unknown


def label_colours(ann):
    """(label, raw colour) for every named series and mark."""
    return [(l, c) for l, c in ann["series"] if l] + [(m, c) for _, m, c in ann["marks"]]


def colour_name(raw):
    d = describe_colour(raw)
    return d["name"] if d else "unknown"


def linearise(header, body, source, ann):
    lines = [f"Source: {source.upper() if source in ('owid', 'oecd') else source.capitalize()}"]
    series = header[1:]
    if ann:
        ct = CHART_TYPE_TEXT.get(ann["chart_type"], ann["chart_type"])
        n = len(ann["series"]) or ann["n_models"]
        lines.append(f"Chart type: {ct}" + (f", {n} series" if ct != "pie" and n > 1 else ""))
        if ann["title"]:
            lines.append(f"Title: {ann['title']}")
    lines.append(f"Category axis: {header[0] or 'Category'}")
    if ann and ann["series"]:
        parts = [f"{lab or '(unnamed)'}" + (f" (colour: {colour_name(c)})" if describe_colour(c) else "")
                 for lab, c in ann["series"]]
        if len(parts) == 1:
            mark = "bars" if "bar" in (ann["chart_type"] or "") else "points"
            parts[0] = (parts[0][:-1] + f"; single series, all {mark} share this colour)"
                        if parts[0].endswith(")") and "(colour:" in parts[0] else parts[0] + " (single series)")
        lines.append("Series: " + " | ".join(parts))
        listed = {clean(l).lower() for l, _ in ann["series"] if l}
        missing = [s for s in series if clean(s).lower() not in listed]
        if missing and len(series) > 1 and len(ann["series"]) != len(series):
            lines.append("Table columns: " + " | ".join(series))
    elif len(series) == 1:
        lines.append(f"Series: {series[0]}")
    else:
        lines.append("Series: " + " | ".join(series))
    if ann and ann["marks"]:
        word = "Slice" if ann["chart_type"] == "pie" else ("Bar" if "bar" in (ann["chart_type"] or "") else "Point")
        by_series = collections.OrderedDict()
        for s, m, c in ann["marks"]:
            if describe_colour(c) is None:
                continue
            by_series.setdefault(s, []).append(f"{m} = {colour_name(c)}")
        for s, items in by_series.items():
            prefix = f"{word} colours" + (f" ({s})" if s and len(by_series) > 1 else "")
            lines.append(f"{prefix}: " + "; ".join(items))
    lines.append("Table:")
    lines.append("| " + " | ".join(md_cell(h) for h in header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in body:
        lines.append("| " + " | ".join(md_cell(c) for c in r) + " |")
    return "\n".join(lines)


def infer_unit(header):
    text = " ".join(header[1:])
    for pat, fixed in UNIT_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if fixed:
            return fixed
        return re.sub(r"\s+", " ", m.group(0)).strip(" ,)").removeprefix("in ").strip()
    return None


def domain_tags(header, body, question, ann):
    extra = " ".join(x for x in ((ann or {}).get("title"), (ann or {}).get("y_label")) if x)
    blob = " ".join(header) + " " + " ".join(r[0] for r in body) + " " + question + " " + extra
    return sorted(k for k, rx in DOMAIN_RE.items() if rx.search(blob))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path):
    h = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        h.update(name.encode())
        h.update(bytes.fromhex(sha256(os.path.join(path, name))))
    return h.hexdigest()


# --------------------------------------------------------------------------- filtering
def classify(item, header, body, ann, text):
    """Return (exclusion_reason | None, flags dict). `text` is the full serialisation."""
    q = item["query"]
    flags = {
        "mentions_colour": bool(COLOUR.search(q)),
        "mentions_structure": bool(STRUCTURE.search(q)),
        "arithmetic_cue": bool(ARITH.search(q)),
    }
    if flags["arithmetic_cue"]:
        qtype = "arithmetic_comparison"
    elif flags["mentions_colour"] or flags["mentions_structure"]:
        qtype = "visual_reference"
    else:
        qtype = "retrieval"
    flags["question_type"] = qtype
    flags["n_series"] = len(header) - 1
    ans, atype = item["answer"], item["answer_type"]
    colour_answer = False

    if APPEARANCE_OTHER.search(q):
        return "asks_about_chart_appearance", flags
    if TITLE_ASK.search(q) and not (ann and ann["title"]):
        return "asks_about_title_but_chart_has_none", flags
    if flags["mentions_colour"] or LEGEND.search(q):
        if ann is None:
            return "colour_reference_no_annotation", flags
        colours, unknown = chart_colours(ann)
        if not colours or unknown:
            # Mostly Pew charts whose annotator wrote "UNK" for some or all colours.
            return "colour_reference_annotation_lacks_colour", flags
        phrases = colour_phrases(q)
        resolved = []
        for p in phrases:
            hit = resolve_phrase(p, colours)
            if not hit:
                return "colour_reference_not_in_chart", flags
            if len(hit) > 1:
                return "colour_reference_ambiguous", flags
            resolved.append(colours[hit[0]]["name"])
        flags["colour_resolved_to"] = resolved
        if COLOUR_ASK.search(q) and atype == "text":
            # The answer itself is a colour: it must resolve, by its own colour family, to
            # exactly one colour in this chart.
            ap = colour_phrases(ans)
            hit = resolve_phrase(ap[0], colours) if ap else []
            if len(hit) != 1 or ap[0][1] not in {colours[hit[0]]["family"]} | colours[hit[0]]["extra"]:
                return "colour_answer_not_in_annotation", flags
            # If the question names exactly one series/mark, the gold colour must be that one's.
            named = {colour_name(c) for l, c in label_colours(ann) if len(norm_text(l)) >= 3
                     and re.search(r"(?<!\w)" + re.escape(norm_text(l)) + r"(?!\w)", norm_text(q))}
            if len(named) == 1 and colours[hit[0]]["name"] not in named:
                return "colour_answer_conflicts_with_annotation", flags
            colour_answer = True
            flags["colour_answer_input_name"] = colours[hit[0]]["name"]
        elif resolved and atype == "text":
            # "Which country does the red line represent?": when the gold answer is a series or
            # mark label, that label must carry the colour the question's phrase resolved to.
            label_colour = {norm_text(l): colour_name(c) for l, c in label_colours(ann)}
            if ans in label_colour and label_colour[ans] not in resolved:
                return "colour_reference_conflicts_with_gold", flags
        elif LEGEND.search(q) and not phrases:
            return "asks_about_chart_appearance", flags

    t_norm = norm_text(text)
    if atype == "text" and not colour_answer:
        # List answers such as "[United States, Spain]" must have every element in the serialisation.
        parts = [p for p in re.split(r"\s*,\s*", ans.strip("[]")) if p] if ans.startswith("[") else [ans]
        flat = re.sub(r"[^\w ]", "", t_norm)
        for a in parts:
            a = norm_text(a)
            if a not in t_norm and re.sub(r"[^\w ]", "", a) not in flat:
                return "text_answer_not_in_input", flags
    if atype == "number" and qtype != "arithmetic_comparison" and not COLOUR_ASK.search(q):
        # Plain look-up of a number: the gold value must be in the table (within rounding),
        # be a category label (e.g. a year) or appear in the title.
        vals = numeric_cells(body) + [v for v in (to_number(r[0]) for r in body) if v is not None] \
            + [v for v in (to_number(h) for h in header) if v is not None]
        if ann and ann["title"]:
            vals += [float(x) for x in re.findall(r"\d+(?:\.\d+)?", ann["title"].replace(",", ""))]
        g = float(ans)
        if not any(abs(v - g) <= max(0.011, abs(g) * 0.005) for v in vals):
            return "numeric_lookup_not_in_table", flags
    return None, flags


# --------------------------------------------------------------------------- sampling
def stratified_sample(items, n, rng, keys=STRATA):
    """Proportional allocation over strata (largest remainder), random within stratum."""
    strata = collections.defaultdict(list)
    for it in items:
        strata[tuple(it[k] for k in keys)].append(it)
    total = len(items)
    quotas = {k: n * len(v) / total for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    rest = n - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: (quotas[k] - alloc[k], rng.random()), reverse=True)[:rest]:
        alloc[k] += 1
    out = []
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda x: x["id"])
        rng.shuffle(pool)
        out += pool[: alloc[k]]
    return sorted(out, key=lambda x: x["id"])


def make_unanswerable(pool_items, texts, colours, n, rng, used_ids):
    """Pair a question with a different chart whose serialisation (title, series, colours,
    table) shares no content words or numbers with it."""
    stop = set("the a an of in on to for and or is are was were what which how many much value "
               "between by from with does do did this that it its at as be".split())
    charts = sorted(texts)
    candidates = sorted((x for x in pool_items if x["id"] not in used_ids), key=lambda x: x["id"])
    rng.shuffle(candidates)
    out = []
    for it in candidates:
        if len(out) >= n:
            break
        qwords = {w for w in re.findall(r"[a-z]{3,}", it["query"].lower()) if w not in stop}
        qnums = set(re.findall(r"\d+(?:\.\d+)?", it["query"]))
        for _ in range(50):
            other = rng.choice(charts)
            if other == it["chart"]:
                continue
            ttext = texts[other].lower()
            twords = set(re.findall(r"[a-z]{3,}", ttext)) - {"source", "chart", "type", "category", "axis",
                                                            "series", "table", "colour", "colours", "single"}
            tnums = set(re.findall(r"\d+(?:\.\d+)?", ttext))
            if qwords & twords or qnums & tnums:
                continue
            if any(resolve_phrase(p, colours[other]) for p in colour_phrases(it["query"])):
                continue
            out.append((it, other))
            break
    return out
