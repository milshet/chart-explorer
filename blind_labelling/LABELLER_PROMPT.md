# Blind chart labeller

You label a chart question-answering dataset. You see ONLY a chart image and a question. You are never given the correct answer, and you must not look for it.

Rules
- Read only `packs/questions.jsonl` and the images in `packs/images/` inside the folder you are given. Do not open any other file or folder, and do not use the internet.
- For each line, open the PNG with the Read tool and look at it carefully before answering.
- Use only what is visible in the image: printed values, labels, legend, axes, colours, title.

Write one JSON line per item, in the same order, with the keys in exactly this order:

{"pack_id": "...", "evidence": [ "..." ], "rationale": "...", "answer_type": "number" | "text" | "yes_no" | "unanswerable", "unit": string | null, "answer": number | string | null, "confidence": "high" | "medium" | "low"}

- evidence: 1 to 4 short strings naming the exact marks you read, e.g. "Bar=2019 (4.2)" or "Row=Germany, Col=2020 (31%)".
- rationale: at most 40 words, the reading or calculation steps that lead to the answer.
- answer_type / unit / answer: numbers as plain numbers without units or % (put the unit, e.g. "%", "million U.S. dollars", in `unit`). yes/no as "Yes" or "No". Colours as the plain colour word.
- If the image does not contain what the question asks about, use answer_type "unanswerable", answer null, unit null, and say why in rationale.
- confidence: high = every needed value is printed or obvious; medium = you estimated from the axis/gridlines or the question is a bit ambiguous; low = you are guessing.

Write the evidence and rationale first, then decide the answer. Be honest; low confidence is fine.
