# Second-teacher (GPT) blind check: result

- Sample: 30 random train items (seed 11) from the v2.0 selection, before any filtering by the first labeller. 16 arithmetic, 9 visual-reference, 4 retrieval, 1 unanswerable.
- Labeller: OpenAI GPT through the free web apps, 2026-11-09. g001-g003 in ChatGPT, g004-g030 in Microsoft Copilot. Same prompt and rules as the main labelling; the answers (`key/`) were never shown.
- Compared with `scripts/compare.py` against the human ChartQA answers.

| | items |
|---|---|
| GPT matches human answer | **29 / 30** (all high) |
| GPT disagrees | 1 (g006) |
| Key order correct | 30 / 30 |

The first labeller (Claude Sonnet) also matched on 29 of these 30 and missed the same item, g006 ("How many years have the same value for both investment and development?", human answer 2, both models said 0). That item was already dropped from the training set as low confidence.

So two different model families, labelling blind and independently, agree with the human answers on the same 29 items, and the only disagreement is an ambiguous chart: in 2007 and 2008 the two lines almost touch and the Development values are not printed, so "the same value" cannot be confirmed from the image.
