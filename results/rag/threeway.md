| Condition | Relaxed acc. | True abst. | False abst. | vs base (only this / only base, p) | vs fine-tuned |
|---|---|---|---|---|---|
| base | 57.2% | 7/9 | 0/171 |  | 5 / 53 (p = 3.5e-11) |
| base+RAG | 48.9% | 8/9 | 5/171 | 21 / 36 (p = 0.0627) | 7 / 70 (p = 3.52e-14) |
| fine-tuned | 83.9% | 9/9 | 1/171 | 53 / 5 (p = 3.5e-11) |  |
| fine-tuned+RAG | 76.1% | 9/9 | 2/171 | 49 / 15 (p = 2.44e-05) | 5 / 19 (p = 0.00661) |
| fine-tuned(table)+RAG | 84.4% | 9/9 | 1/171 | 55 / 6 (p = 5.38e-11) | 9 / 8 (p = 1) |

Retrieval failure (task 3.6), rows whose gold table was withheld:

- fine-tuned(table): **lower, not significant** — 77.6% with RAG vs 83.5% without RAG (Run A, image only) (n = 85, p = 0.227); false abstention 1/85 vs 1/85
- base: **harmful** + over-abstaining — 21.2% with RAG vs 54.1% without RAG (base, image only) (n = 85, p = 2.46e-07); false abstention 8/85 vs 0/85
