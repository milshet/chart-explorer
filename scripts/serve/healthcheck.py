#!/usr/bin/env python3
"""Container health check (task 4.4): healthy only if the endpoint is up AND serves a VERIFIED merge.

"Is the process up?" is not the question during a demo; "is the thing that is up the thing that was measured?"
is. So this reads /health, which reports the merge manifest (task 4.1), and exits 0 only when the service says
"ok" and the manifest status is VERIFIED. Standard library only: it runs inside the image every 30 s.

  python scripts/serve/healthcheck.py [--url http://127.0.0.1:8000]
"""
import argparse
import json
import sys
import urllib.request


def healthy(payload):
    """(ok, reason) for a /health payload."""
    if payload.get("status") != "ok":
        return False, f"service status {payload.get('status')!r}"
    model = payload.get("model") or {}
    if model.get("status") != "VERIFIED":
        return False, f"merge manifest status {model.get('status')!r}"
    if not model.get("merged_files_sha256"):
        return False, "the manifest lists no weight hashes"
    return True, f"serving {model.get('run')} ({model.get('status')}, adapter {str(model.get('adapter_sha256'))[:12]})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args(argv)
    try:
        with urllib.request.urlopen(f"{args.url.rstrip('/')}/health", timeout=5) as r:
            ok, reason = healthy(json.loads(r.read()))
    except (OSError, ValueError) as e:
        ok, reason = False, f"no /health: {e}"
    print(("healthy: " if ok else "unhealthy: ") + reason)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
