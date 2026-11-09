# Second-teacher check with free ChatGPT (30 items)

A blind cross-check of 30 random train items with a different model (ChatGPT), using exactly the same rules as the main labelling. ChatGPT never sees the answers.

Folders:
- `packs/` : what you give ChatGPT (images g001-g030 and questions). Safe to share.
- `key/answers.jsonl` : the human answers. **Do not upload or paste this to ChatGPT.**
- `outputs/` : put ChatGPT's replies here.

Steps (10 rounds of 3 images; free ChatGPT limits uploads, so if it says you hit the limit, continue later or the next day):
1. Open a **new chat** at chatgpt.com (a fresh chat per round, so earlier rounds can't influence it).
2. Upload the 3 images for the round from `packs/images/` (e.g. g001.png to g003.png).
3. Paste the whole of `PROMPT_CHATGPT.txt`, then the 3 question lines for that round from `packs/questions_to_paste.txt`. Send.
4. Copy the JSON lines it returns into `outputs/chatgpt.jsonl` (one line per item, append each round).
5. When all 30 are done, from the repo folder run:
   `python blind_labelling/scripts/compare.py blind_labelling/runs/gpt_check`
   It prints the agreement and writes `compare.jsonl` and `compare_report.txt`.

Write down which ChatGPT model the chat used (shown at the top of the chat) and the date, for the report.
