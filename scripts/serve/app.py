#!/usr/bin/env python3
"""The endpoint (task 4.3): FastAPI in front of serve/service.py, serving the verified merge with transformers.generate.

Served with transformers.generate, not vLLM (plan §9 decision 12, docs/SERVING.md): vLLM's dependencies did not fit
the VM's 35 GB quota and pinned a second torch. The cost is no continuous batching; requests are answered one at a
time, and the latency quoted is the measured one (results/serve/), never a vLLM figure.

  sbatch slurm/serve.sbatch                                  # GPU job: serves ~/merged/run_a_r16 on 127.0.0.1:8000
  python scripts/serve/app.py --model ~/merged/run_a_r16     # the same, inside any GPU job

  curl -s localhost:8000/health
  python scripts/serve/client.py data/v2.0/images/val/two_col_24100.png "What is the sum all the blue bar?"

This file holds no logic: the request path, its limits, refusals and counters are in service.py, tested on the
laptop against a scripted engine. The app refuses to start on a checkpoint whose merge manifest is not VERIFIED.
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from serve import service as svc  # noqa: E402

MANIFEST = "merge_manifest.json"


def create_app(service):
    from fastapi import FastAPI, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse

    app = FastAPI(title="ChartAudit-Fin chart QA", version="4.3",
                  description="Qwen2.5-VL-3B + Run A LoRA, merged (task 4.1); transformers.generate, batch 1.")

    @app.post("/v1/answer")
    async def answer(request: Request):
        body = await request.body()
        status, payload = await run_in_threadpool(service.handle, body)  # generation never blocks /health
        return JSONResponse(payload, status_code=status)

    @app.get("/health")
    def health():
        return service.health()

    @app.get("/metrics")
    def metrics():
        return service.metrics()

    return app


def load_manifest(model_dir):
    path = Path(model_dir) / MANIFEST
    if not path.is_file():
        raise SystemExit(f"{path} is missing: serve only a checkpoint written by scripts/serve/merge.py (task 4.1)")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path.home() / "merged" / "run_a_r16")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--targets", choices=tuple(svc.STYLE), help="default: the adapter's train_config.json")
    ap.add_argument("--allow-unverified", action="store_true", help="serve a checkpoint whose manifest is not VERIFIED")
    args = ap.parse_args(argv)
    model_dir = args.model.expanduser().resolve()
    manifest = load_manifest(model_dir)
    style = svc.served_style(manifest, args.targets)
    if manifest.get("status") != "VERIFIED" and not args.allow_unverified:
        raise SystemExit(f"{model_dir}: merge manifest status {manifest.get('status')!r}, not VERIFIED")
    import train  # git state for /health
    import uvicorn
    from serve import generation
    engine = generation.HFEngine(model_dir=model_dir)
    commit, dirty = train.git_state()
    service = svc.Service(engine, manifest, style, allow_unverified=args.allow_unverified,
                          serving_commit=commit + ("+dirty" if dirty else "") if commit else None)
    print(json.dumps(service.health(), indent=1), flush=True)
    uvicorn.run(create_app(service), host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
