#!/usr/bin/env python3
"""Ask the endpoint one question about one chart image (standard library only; also used by the bench and the demo).

  python scripts/serve/client.py data/v2.0/images/val/two_col_24100.png "What is the sum all the blue bar?"
  python scripts/serve/client.py --url http://127.0.0.1:8000 chart.png "Which year had the highest revenue?"
"""
import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8000"


def request_body(image_path, question):
    return json.dumps({"question": question,
                       "image_base64": base64.b64encode(Path(image_path).read_bytes()).decode()}).encode()


def post(url, body, timeout=120):
    """(HTTP status, parsed JSON) for a POST to {url}/v1/answer; 4xx/5xx come back as values, not exceptions."""
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/answer", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get(url, path, timeout=10):
    with urllib.request.urlopen(f"{url.rstrip('/')}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("question")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args(argv)
    status, payload = post(args.url, request_body(args.image, args.question))
    print(json.dumps(payload, indent=1))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
