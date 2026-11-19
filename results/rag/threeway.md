| Condition | Relaxed acc. | True abst. | False abst. | vs base (only this / only base, p) | vs fine-tuned |
|---|---|---|---|---|---|
| base | 57.2% | 7/9 | 0/171 |  | 5 / 53 (p = 3.5e-11) |
| base+RAG | NOT MEASURED | | | | |
| fine-tuned | 83.9% | 9/9 | 1/171 | 53 / 5 (p = 3.5e-11) |  |
| fine-tuned+RAG | NOT MEASURED | | | | |
| fine-tuned(table)+RAG | NOT MEASURED | | | | |

Retrieval failure (task 3.6), rows whose gold table was withheld:

- fine-tuned(table): NOT MEASURED
- base: NOT MEASURED
