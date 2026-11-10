#!/usr/bin/env python3
"""Make audit/review.html: a one-item-per-screen page for the human audit.

The page shows the same 50 items as audit/audit_v2.xlsx (ids in audit/audit_v2_ids.txt),
the chart large, and three Yes / No / Unsure questions per item. Progress is kept in the
browser; at the end the reviewer downloads audit_reviewer_<A|B|C>.json, and
scripts/audit_import.py writes it into the workbook.

Open audit/review.html from inside the repo (it loads the charts from ../data/v2.0/images).

Usage:
  python scripts/make_review_page.py --data data/v2.0 --ids audit/audit_v2_ids.txt --out audit/review.html
"""
import argparse, json, os

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label audit</title>
<style>
:root{--bg:#f7f6f1;--card:#fff;--ink:#14213d;--soft:#4a5568;--line:#ddd8cc;--acc:#b4531a;--yes:#2b5d9b;--no:#b4531a;--uns:#6b7a90}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;gap:16px;align-items:center;justify-content:space-between;padding:12px 20px;background:var(--ink);color:#f7f6f1;flex-wrap:wrap}
header b{font-size:18px}.bar{flex:1;min-width:160px;height:8px;background:#3a5584;border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:#f2a65a}
main{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(320px,1fr);gap:20px;padding:20px;max-width:1500px;margin:auto}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.chart img{width:100%;height:auto;display:block;border-radius:6px}
h2{font-size:20px;margin:0 0 8px}.lbl{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--acc);font-weight:700;margin-top:12px}
.ans{font-size:22px;font-weight:700}ul{margin:4px 0;padding-left:20px}
.q{margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}.q p{margin:0 0 6px;font-weight:600}
.opts{display:flex;gap:8px}.opts button{flex:1;padding:10px;border:2px solid var(--line);background:#fff;border-radius:8px;font-size:15px;cursor:pointer}
.opts button.on[data-v=Yes]{background:var(--yes);border-color:var(--yes);color:#fff}.opts button.on[data-v=No]{background:var(--no);border-color:var(--no);color:#fff}.opts button.on[data-v=Unsure]{background:var(--uns);border-color:var(--uns);color:#fff}
textarea{width:100%;min-height:60px;border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit}
nav{display:flex;gap:8px;margin-top:14px}nav button,.big{padding:10px 16px;border-radius:8px;border:0;background:var(--ink);color:#fff;font-size:15px;cursor:pointer}nav button.sec{background:#c9d3e3;color:var(--ink)}
.start{max-width:640px;margin:60px auto}.start label{display:block;margin:10px 0}.hint{color:var(--soft);font-size:14px}
.warn{color:var(--no);font-weight:600}
</style></head><body>
<div id="app"></div>
<script>
const ITEMS = __ITEMS__;
const CHECKS = [["answer_correct","Is the ANSWER correct when you read the chart yourself? (numbers within about 5% count)"],
 ["evidence_matches","Do the EVIDENCE values really appear in the chart?"],
 ["rationale_ok","Does the RATIONALE correctly lead to the answer?"]];
let who = null, i = 0, data = {};
const store = {get(k){try{return JSON.parse(localStorage.getItem(k))}catch(e){return null}}, set(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}}};
const key = () => "audit_v2_reviewer_" + who;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const done = it => CHECKS.every(([k]) => (data[it.id]||{})[k]);
function save(){ store.set(key(), {i, data}); }
function start(){
  document.getElementById("app").innerHTML = `<div class="start card"><h2>Human audit of ChartAudit-Fin v2.0 labels</h2>
  <p>You will see ${ITEMS.length} training items, one at a time. For each, look at the chart and answer three questions with Yes, No or Unsure. Add a short note for any No or Unsure.</p>
  <p>Work on your own. Do not look at the other reviewers' answers.</p>
  <p class="lbl">I am</p>
  <label><input type="radio" name="r" value="A"> Reviewer A</label><label><input type="radio" name="r" value="B"> Reviewer B</label><label><input type="radio" name="r" value="C"> Reviewer C</label>
  <p class="hint">Your progress is saved in this browser, so you can close the page and come back. At the end, click "Download my results".</p>
  <button class="big" id="go">Start</button></div>`;
  document.getElementById("go").onclick = () => {
    const r = document.querySelector("input[name=r]:checked"); if(!r) return alert("Choose Reviewer A, B or C");
    who = r.value; const s = store.get(key()); if(s){ i = s.i||0; data = s.data||{}; } render(); };
}
function render(){
  const it = ITEMS[i], d = data[it.id] || (data[it.id] = {}), n = ITEMS.filter(done).length;
  document.getElementById("app").innerHTML = `<header><b>Reviewer ${who} · item ${i+1} of ${ITEMS.length}</b>
   <div class="bar"><i style="width:${100*n/ITEMS.length}%"></i></div><span>${n} done</span>
   <button class="big" id="dl" style="background:#f2a65a;color:#14213d">Download my results</button></header>
  <main><div class="card chart"><img src="${esc(it.image)}" alt="Chart for item ${i+1}" onerror="this.replaceWith(Object.assign(document.createElement('p'),{className:'warn',textContent:'Chart not found. Open this page from the audit folder inside the repo.'}))"></div>
  <div class="card"><h2>${esc(it.question)}</h2>
   <div class="lbl">Answer</div><div class="ans">${esc(it.answer)}</div>
   <div class="lbl">Evidence</div><ul>${(it.evidence||[]).map(e=>`<li>${esc(e)}</li>`).join("") || "<li>(none)</li>"}</ul>
   <div class="lbl">Rationale</div><div>${esc(it.rationale)}</div>
   ${CHECKS.map(([k,t]) => `<div class="q"><p>${t}</p><div class="opts">${["Yes","No","Unsure"].map(v=>`<button data-k="${k}" data-v="${v}" class="${d[k]===v?"on":""}">${v}</button>`).join("")}</div></div>`).join("")}
   <div class="q"><p>Notes (needed for any No or Unsure)</p><textarea id="note">${esc(d.notes||"")}</textarea></div>
   <nav><button class="sec" id="prev">◀ Back</button><button id="next">${i===ITEMS.length-1?"Finish":"Next ▶"}</button><button class="sec" id="skip">Next unfinished</button></nav>
  </div></main>`;
  document.querySelectorAll(".opts button").forEach(b => b.onclick = () => { d[b.dataset.k] = b.dataset.v; save(); render(); });
  document.getElementById("note").oninput = e => { d.notes = e.target.value; save(); };
  document.getElementById("prev").onclick = () => { if(i>0){ i--; save(); render(); } };
  document.getElementById("next").onclick = () => { if(i<ITEMS.length-1){ i++; save(); render(); } else finish(); };
  document.getElementById("skip").onclick = () => { const j = ITEMS.findIndex(x => !done(x)); if(j>=0){ i=j; save(); render(); } else finish(); };
  document.getElementById("dl").onclick = download;
}
function finish(){
  const left = ITEMS.filter(x => !done(x)).length;
  if(left) { alert(left + " item(s) still have unanswered questions. 'Next unfinished' takes you to them."); return; }
  download();
}
function download(){
  const out = {reviewer: who, made: new Date().toISOString(), items: ITEMS.map((it,n) => ({n: n+1, id: it.id, ...(data[it.id]||{})}))};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "audit_reviewer_" + who + ".json"; document.body.appendChild(a); a.click(); a.remove();
}
start();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v2.0")
    ap.add_argument("--ids", default="audit/audit_v2_ids.txt")
    ap.add_argument("--out", default="audit/review.html")
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids, encoding="utf-8") if l.strip()]
    rows = {}
    for l in open(os.path.join(a.data, "image_only", "train.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        if r["id"] in ids:
            rows[r["id"]] = r
    rel = os.path.relpath(a.data, os.path.dirname(os.path.abspath(a.out)) or ".").replace(os.sep, "/")
    items = []
    for i in ids:
        r = rows[i]
        t = json.loads(r["messages"][2]["content"])
        ans = f'{t["answer"]}' + (f' ({t["unit"]})' if t.get("unit") else "")
        items.append({"id": i, "image": f"{rel}/{r['image']}", "question": r["question"], "answer": ans,
                      "evidence": t.get("evidence") or [], "rationale": t.get("rationale") or ""})
    html = PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"wrote {a.out} with {len(items)} items")


if __name__ == "__main__":
    main()
