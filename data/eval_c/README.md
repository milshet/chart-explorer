# Eval-C — abstention probes (task 3.0c)

50 questions the paired chart cannot answer, each for a reason proved from the data tables when the set was
built (`scripts/build_eval_c.py`, seed 31003). Held out: never trained on, never used for selection. Each
compared model is scored on it once: `evaluate.py --split eval_c` (adapters also need `--final-test`).

| `abstain_reason` | n | How it is proved |
|---|---:|---|
| `series_absent` | 13 | the question names a series and a category of its own chart; the paired chart has the category, not the series |
| `category_absent` | 13 | the paired chart has the series, not the category |
| `not_in_chart` | 12 | no table entity the question names occurs in the paired chart's table |
| `requires_external_knowledge` | 12 | the row and series are in the chart; the quantity asked for (inflation adjustment, world share, cause, currency conversion) is not. **Templated** (`origin: "template"`) |

The other 38 questions are real, unused human ChartQA finance/economy questions (`origin: "catalogue"`,
`source_question_id`), excluded by id and by text from every question the v2.0 selection used. Each record
lists `absent_entities` and `present_entities`; `tests/test_evaluate.py` re-checks every proof against the
committed tables.

**Limitations.** The paired charts are val/test charts (§9 decision 10), so Eval-C shares images with Eval-A.
That is deliberate here: the same chart is answerable in Eval-A and unanswerable in Eval-C, so a model cannot
pass by refusing on how the chart looks. Absence is proved against the serialised table, not against every
pixel; a chart's image may carry footnotes the table does not. Eval-C holds no answerable items, so its
false-abstention half is always empty: it is read beside Eval-A's false-abstention rate, never alone.

Layout matches `data/v2.0` (`image_only/`, `image_table/`, each with `answer_only/`); images resolve to
`../v2.0/images/`. `MANIFEST.txt` hashes all four files; `stats.json` records the balance.
