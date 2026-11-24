#!/usr/bin/env python3
"""Served latency through HTTP (task 4.3): the same 50 requests task 3.14 timed in-process, sent to the endpoint.

  sbatch slurm/serve_http_bench.sbatch          # starts the endpoint in the job, runs this against it, stops it
  python scripts/serve/http_bench.py --dry-run  # login shell: items, provenance, the 3.14 file it compares with

results/serve/run_a_r16.json (3.14) timed preprocess + generate inside one process: p50 1.256 s. Here each request
is a real POST with the chart as base64 PNG, so the difference is what HTTP, JSON, base64 and validation cost:

  e2e_s        client send -> response parsed: what a caller waits for
  server_s     the service's own total (decode the image, build the prompt, generate, validate, retry)
  overhead     e2e_s - server_s per request: the transport alone

The bench also reports the retry and fallback counters the endpoint accumulated over these requests (/metrics
before and after), and scores every returned contract object with scripts/scoring.py, beside the 3.14 accuracy on
the same items. It refuses to run unless its items are exactly the ones in the 3.14 file.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import evaluate as ev  # noqa: E402
import scoring  # noqa: E402
import serve_bench as sb  # noqa: E402  bench_items(), pct(): the 3.14 sample and percentiles
from serve import client  # noqa: E402

OUT = REPO / "results" / "serve"


def dist(values):
    v = [x for x in values if x is not None]
    return {"p50": round(sb.pct(v, 50), 3), "p95": round(sb.pct(v, 95), 3), "mean": round(sum(v) / len(v), 3)}


def score_row(item, status, payload, e2e, style):
    row = {"id": item["id"], "status": status, "e2e_s": round(e2e, 4)}
    if status != 200:
        return {**row, "error": payload.get("error"), "correct": False, "schema_valid": False}
    res = scoring.score(item, json.dumps(payload["result"]), style)
    return {**row, "server_s": payload["timing"]["total_s"], "generate_s": payload["timing"]["generate_s"],
            "overhead_s": round(e2e - payload["timing"]["total_s"], 4), "attempts": payload["attempts"],
            "degraded": payload["degraded"], "new_tokens": payload["usage"]["new_tokens"],
            "correct": bool(res["correct"]), "schema_valid": bool(res["schema_valid"])}


def summarise(rows, direct):
    ok = [r for r in rows if r["status"] == 200]
    d = direct["summary"]
    s = {"n": len(rows), "http_errors": len(rows) - len(ok),
         "e2e_s": dist(r["e2e_s"] for r in ok), "server_s": dist(r["server_s"] for r in ok),
         "overhead_s": dist(r["overhead_s"] for r in ok),
         "retried": sum(r["attempts"] > 1 for r in ok), "degraded": sum(r["degraded"] for r in ok),
         "accuracy_on_these_items": round(sum(r["correct"] for r in rows) / len(rows), 4),
         "schema_valid": round(sum(r["schema_valid"] for r in rows) / len(rows), 4)}
    s["vs_in_process_3_14"] = {
        "in_process_e2e_s": d["e2e_s"], "in_process_accuracy": d["accuracy_on_these_items"],
        "in_process_schema_valid": d["schema_valid"],
        "http_minus_in_process_p50_s": round(s["e2e_s"]["p50"] - d["e2e_s"]["p50"], 3),
        "http_minus_in_process_p95_s": round(s["e2e_s"]["p95"] - d["e2e_s"]["p95"], 3)}
    return s


def wait_for(url, seconds):
    deadline = time.monotonic() + seconds
    while True:
        try:
            return client.get(url, "/health")
        except OSError:
            if time.monotonic() > deadline:
                sys.exit(f"the endpoint at {url} did not come up within {seconds} s")
            time.sleep(5)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=Path.home() / "runs/run_a_r16/adapter",
                    help="the served model's adapter: selects the same val items and prompt style as 3.14")
    ap.add_argument("--url", default=client.URL)
    ap.add_argument("--direct", type=Path, help="the 3.14 in-process result (default results/serve/<run>.json)")
    ap.add_argument("--out", type=Path, help="default results/serve/<run>_http.json")
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for /health (the model loads first)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-unclean", action="store_true")
    args = ap.parse_args(argv)

    ns = ev.parse_args(["--adapter", str(args.adapter), "--split", "val"])
    s = ev.resolve(ns)
    items, _, fingerprint = ev.load_rows(s, "val", ns)
    prov, problems = ev.provenance(fingerprint, s["adapter"], s["data_dir"])
    warm, timed = sb.bench_items(items)
    direct_file = args.direct or OUT / f"{s['run_name']}.json"
    direct = json.loads(direct_file.read_text(encoding="utf-8"))
    if direct["settings"]["items"] != [i["id"] for i in timed]:
        sys.exit(f"{direct_file} timed different items: the HTTP numbers would not be comparable")
    out = args.out or OUT / f"{s['run_name']}_http.json"
    if args.dry_run:
        print(json.dumps({"run": s["run_name"], "url": args.url, "items": len(timed), "warmup": len(warm),
                          "compares_with": str(direct_file), "out": str(out), "provenance": prov}, indent=1))
        for p in problems:
            print(f"WARNING: the GPU job would refuse to start: {p}")
        print("HTTP_BENCH_DRY_RUN_OK")
        return 0
    if problems and not args.allow_unclean:
        sys.exit("refusing to start:\n  " + "\n  ".join(problems))

    health = wait_for(args.url, args.wait)
    if health["model"]["status"] != "VERIFIED":
        sys.exit(f"the endpoint serves a checkpoint whose merge status is {health['model']['status']!r}")
    image = lambda it: REPO / s["items_dir"] / it["image"]  # noqa: E731
    for it in warm:
        client.post(args.url, client.request_body(image(it), it["question"]))
    before = client.get(args.url, "/metrics")
    rows = []
    for i, it in enumerate(timed, 1):
        body = client.request_body(image(it), it["question"])
        t0 = time.perf_counter()
        status, payload = client.post(args.url, body)
        rows.append(score_row(it, status, payload, time.perf_counter() - t0, s["style"]))
        print(f"  {i}/{len(timed)} {status} {rows[-1]['e2e_s']:.2f}s", flush=True)
    after = client.get(args.url, "/metrics")
    delta = {k: after["counters"][k] - before["counters"].get(k, 0) for k in after["counters"]}
    report = {"task": "4.3", "run": s["run_name"], "url": args.url, "health": health,
              "settings": {**direct["settings"], "transport": "HTTP POST, JSON, base64 PNG", "client": "urllib"},
              "summary": summarise(rows, direct), "endpoint_counters_over_timed_requests": delta,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "provenance": prov, "reportable": prov["clean"],
              "requests": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"], indent=1))
    print("HTTP_BENCH_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
