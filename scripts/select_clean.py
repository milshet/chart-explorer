#!/usr/bin/env python3
"""Select the ChartAudit-Fin items straight from the raw ChartQA files, by written rules only.

No model is involved anywhere in this script. Steps, per official split (train/val/test,
chart-disjoint):
  1. Read the human-written questions (<split>_human.json).
  2. Drop structural problems: missing data table, duplicate or malformed question, empty
     answer, unparseable table, a train/val table identical to one in a later split, and the
     rule-based checks in chartqa_rules.classify (e.g. a looked-up number must exist in the
     chart's own data table).
  3. Keep questions whose keyword domain tags include finance or economy.
  4. Stratified random sample (seed fixed) over source x chart type x answer type x
     question type, plus ~5% "unanswerable" items (a question paired with an unrelated chart).
  5. Copy the chart PNGs and write <out>/<split>.jsonl with the human answer as gold.
  Also writes all_domains_catalogue.jsonl: every question that passes step 2, in ALL domains,
  with its domain tags and no labels, and per-domain counts in stats.json ("all_domains").
     No evidence or rationale is written here; those come from blind labelling.

Rules at a glance (each rule is marked "RULE n" in the code below):
  RULE 1  Basic validity       no data table / question under 3 words / empty answer /
                               table with fewer than 2 numbers.
                               Issue: items we cannot check or train on.
  RULE 2  De-duplication       same chart + same question text (case-insensitive) kept once.
                               Issue: repeated items over-weight one question.
  RULE 3  Leakage control      a train/val chart whose data table is byte-identical to a table
                               in a later split (val/test) is dropped.
                               Issue: the model could "remember" a test chart from training.
  RULE 4  Consistency checks   chartqa_rules.classify(): a looked-up number must be in the
                               chart's own table, a text answer must appear in the chart,
                               colour questions must resolve to exactly one colour, appearance
                               and missing-title questions are dropped.
                               Issue: wrong or unverifiable gold answers (e.g. 0.119 stored for
                               11.9%) and questions the image cannot answer reliably.
  RULE 5  Domain filter        keyword tags; keep finance or economy (all domains are still
                               counted in the catalogue).
                               Issue: project scope is finance/economics charts.
  RULE 6  Stratified sampling  proportional over source x chart type x answer type x question
                               type, fixed seed.
                               Issue: an unbalanced mix, and results nobody can reproduce.
  RULE 7  Unanswerable items   ~5% questions paired with an unrelated chart.
                               Issue: teach the model to say "cannot be answered" instead of
                               guessing.
  CHECK   Chart overlap        after writing, count charts shared between splits (must be 0).

Usage:
  python scripts/select_clean.py --chartqa "/path/to/ChartQA Dataset" --out data/v2.0/selection --images data/v2.0/images
"""
import argparse, collections, hashlib, json, os, random, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chartqa_rules as R

# How many items to pick per split: (answerable, unanswerable). Unanswerable is ~5% (RULE 7).
TARGET = {"train": (1100, 55), "val": (130, 7), "test": (171, 9)}
# RULE 5: the domain tags a question must have (at least one) to enter the selection.
KEEP_DOMAINS = {"finance", "economy"}


def primary_domain(tags):
    """One label per question for counting: finance/economy first, then health, business, other."""
    t = set(tags)
    if t & KEEP_DOMAINS:
        return "finance_economy"
    if "health" in t:
        return "health"
    if "business" in t:
        return "business"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chartqa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", required=True, help="folder for chart PNGs; record paths are images/<split>/<chart>.png relative to its parent")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    # RULE 6: one fixed seed drives every random choice, so a rerun gives identical files.
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    # Load every split first. Two things are recorded on the way:
    #  - the SHA-256 of each raw question file (MANIFEST.txt), so anyone can prove which raw
    #    ChartQA files were used;
    #  - an MD5 fingerprint of every chart's data table, used by RULE 3 (leakage control).
    raw, table_hash, manifest = {}, {}, {}
    for split in TARGET:
        qa = os.path.join(a.chartqa, split, f"{split}_human.json")
        manifest[f"raw {split}/{split}_human.json"] = R.sha256(qa)
        raw[split] = json.load(open(qa, encoding="utf-8"))
        for x in raw[split]:
            p = os.path.join(a.chartqa, split, "tables", x["imgname"][:-4] + ".csv")
            if os.path.exists(p):
                table_hash[(split, x["imgname"][:-4])] = hashlib.md5(open(p, "rb").read()).hexdigest()
    # RULE 3: which splits count as "later" for each split. Test is never changed; if a train
    # or val table is identical to one in val/test, the train/val item is the one dropped.
    later = {"train": {"val", "test"}, "val": {"test"}, "test": set()}

    stats = {"seed": a.seed, "keep_domains": sorted(KEEP_DOMAINS), "splits": {}}
    catalogue = []   # every question that passes the structural checks, all domains, no labels
    for split, (n_ans, n_unans) in TARGET.items():
        reasons, seen, pool, charts = collections.Counter(), set(), [], {}
        # RULE 3: fingerprints of all tables in the later splits.
        block = {h for (s, _), h in table_hash.items() if s in later[split]}
        # Every question goes through the rules below in order. The first rule it fails is
        # counted in `reasons` (saved as "excluded" in stats.json) and the question is skipped.
        for qidx, x in enumerate(raw[split]):
            name = x["imgname"][:-4]
            p = os.path.join(a.chartqa, split, "tables", name + ".csv")
            q = x["query"].strip()
            # RULE 1 (basic validity): no data table means nothing to check the answer against.
            if not os.path.exists(p):
                reasons["no_data_table"] += 1; continue
            # RULE 2 (de-duplication): same chart and same question text, case-insensitive.
            if (name, q.lower()) in seen:
                reasons["duplicate"] += 1; continue
            seen.add((name, q.lower()))
            # RULE 1 (basic validity): questions under 3 words are fragments, not real questions.
            if len(q.split()) < 3:
                reasons["malformed_question"] += 1; continue
            # RULE 1 (basic validity): an empty gold answer cannot be trained on or scored.
            if str(x["label"]).strip() == "":
                reasons["empty_answer"] += 1; continue
            # Read each chart once and cache it: table header + rows, the annotation file
            # (title, colours, chart type) and the chart as text for later checks.
            if name not in charts:
                t = R.read_table(p)
                if t is None or len(R.numeric_cells(t[1])) < 2:
                    charts[name] = None
                else:
                    anp = os.path.join(a.chartqa, split, "annotations", name + ".json")
                    ann = R.load_annotation(anp, *t) if os.path.exists(anp) else None
                    charts[name] = (t[0], t[1], ann, R.linearise(t[0], t[1], R.source_of(name), ann))
            c = charts[name]
            # RULE 1 (basic validity): the table could not be read, or has under 2 numbers.
            if c is None:
                reasons["table_unparseable"] += 1; continue
            # RULE 3 (leakage control): same table already exists in a later split.
            if table_hash[(split, name)] in block:
                reasons["same_table_in_later_split"] += 1; continue
            header, body, ann, text = c
            ans, atype = R.normalise_answer(x["label"])
            it = {"id": f"{split}-{name}-{qidx}", "chart": name, "qidx": qidx, "query": q, "label": x["label"],
                  "answer": ans, "answer_type": atype, "source": R.source_of(name),
                  "chart_type": (ann or {}).get("chart_type") or "unknown"}
            # RULE 4 (consistency checks), all in chartqa_rules.classify(). Possible reasons:
            #   numeric_lookup_not_in_table      plain look-up whose gold number is not in the table
            #                                    (catches 0.119 vs 11.9 style errors)
            #   text_answer_not_in_input         text answer that appears nowhere in the chart
            #   colour_reference_*               the colour in the question is missing, ambiguous
            #                                    or contradicts the chart's annotation
            #   colour_answer_*                  a colour answer the annotation cannot confirm
            #   asks_about_chart_appearance      fonts, styling etc., not chart data
            #   asks_about_title_but_chart_has_none
            # It also sets flags such as question_type (retrieval / arithmetic_comparison /
            # visual_reference), which RULE 6 uses for balancing.
            why, flags = R.classify(it, header, body, ann, text)
            if why:
                reasons[why] += 1; continue
            it["flags"], it["question_type"] = flags, flags["question_type"]
            # RULE 5 (domain): keyword tags from the title, table and question (finance, economy,
            # health, business). Done by fixed word lists, not a model.
            it["domain_tags"] = R.domain_tags(header, body, q, ann)
            # Catalogue: record EVERY question that passed rules 1-4, in all domains, before the
            # domain filter. Counts only, no labels; lets us report the full ChartQA picture.
            catalogue.append({"id": it["id"], "split": split, "chart_id": name, "source": it["source"],
                              "chart_type": it["chart_type"], "question": q, "gold_answer_raw": x["label"],
                              "answer_type": atype, "question_type": flags["question_type"],
                              "domain_tags": it["domain_tags"],
                              "domain": primary_domain(it["domain_tags"]),
                              "in_finance_economy": bool(KEEP_DOMAINS & set(it["domain_tags"]))})
            # RULE 5 (domain filter): keep only finance or economy.
            if not KEEP_DOMAINS & set(it["domain_tags"]):
                reasons["not_finance_or_economy"] += 1; continue
            pool.append(it)
        charts = {k: v for k, v in charts.items() if v is not None}

        # RULE 6 (stratified sampling): pick n_ans items so every group of
        # source x chart type x answer type x question type keeps its share of the pool.
        chosen = R.stratified_sample(pool, n_ans, rng)
        used = {x["id"] for x in chosen}
        colours = {k: (R.chart_colours(v[2])[0] if v[2] else []) for k, v in charts.items()}
        # RULE 7 (unanswerable items): take unused questions and pair each with a different
        # chart that shares no content words, numbers or colours with it. Gold answer is None.
        unans = R.make_unanswerable(pool, {k: v[3] for k, v in charts.items()}, colours, n_unans, rng, used)

        # Write the selected items. Gold = the human ChartQA answer. No evidence or rationale here.
        img_dir = os.path.join(a.images, split)
        os.makedirs(img_dir, exist_ok=True)
        recs = []
        for it, other in [(i, None) for i in chosen] + list(unans):
            chart = other or it["chart"]
            shutil.copy(os.path.join(a.chartqa, split, "png", chart + ".png"), os.path.join(img_dir, chart + ".png"))
            ann = charts[chart][2]
            un = other is not None
            recs.append({
                "id": it["id"] + ("-unans" if un else ""), "split": split, "chart_id": chart,
                "question_source_chart": it["chart"], "source": R.source_of(chart),
                "chart_type": (ann or {}).get("chart_type") or "unknown",
                "question": it["query"],
                "gold_answer_raw": None if un else it["label"],
                "answer": None if un else it["answer"],
                "answer_type": "unanswerable" if un else it["answer_type"],
                "question_type": "unanswerable" if un else it["question_type"],
                "domain": "finance_economy", "domain_tags": it["domain_tags"],
                "flags": {} if un else it["flags"],
                "image": f"images/{split}/{chart}.png",
                "chart_data_table": charts[chart][3],
            })
        rng.shuffle(recs)
        with open(os.path.join(a.out, f"{split}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["splits"][split] = {
            "raw_human_questions": len(raw[split]), "excluded": dict(reasons.most_common()),
            "finance_economy_pool": len(pool), "selected_answerable": len(chosen),
            "selected_unanswerable": len(unans), "charts": len({r["chart_id"] for r in recs}),
            "by_answer_type": dict(collections.Counter(r["answer_type"] for r in recs)),
            "by_question_type": dict(collections.Counter(r["question_type"] for r in recs)),
            "by_source": dict(collections.Counter(r["source"] for r in recs)),
        }
    with open(os.path.join(a.out, "all_domains_catalogue.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for c in catalogue:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    stats["all_domains"] = {
        s: {"questions_after_structural_checks": sum(c["split"] == s for c in catalogue),
            "by_domain": dict(collections.Counter(c["domain"] for c in catalogue if c["split"] == s).most_common()),
            "by_tag_any": dict(collections.Counter(t for c in catalogue if c["split"] == s
                                                   for t in (c["domain_tags"] or ["none"])).most_common())}
        for s in TARGET}
    # CHECK (chart overlap): count charts shared between splits in the written files. Must be 0.
    ids = {s: {json.loads(l)["chart_id"] for l in open(os.path.join(a.out, f"{s}.jsonl"), encoding="utf-8")} for s in TARGET}
    stats["chart_overlap"] = {"train_val": len(ids["train"] & ids["val"]), "train_test": len(ids["train"] & ids["test"]),
                              "val_test": len(ids["val"] & ids["test"])}
    # MANIFEST.txt: SHA-256 of raw inputs and outputs, so a rerun can be proven identical.
    for s in TARGET:
        manifest[f"out {s}.jsonl"] = R.sha256(os.path.join(a.out, f"{s}.jsonl"))
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8", newline="\n"), indent=2)
    with open(os.path.join(a.out, "MANIFEST.txt"), "w", encoding="utf-8", newline="\n") as f:
        for k, v in manifest.items():
            f.write(f"{v}  {k}\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
