#!/usr/bin/env python3
"""Task 4.7: the clean-clone check. Clones the current commit into a temporary directory, as a stranger gets the
repository, and checks that copy. Never the working tree: uncommitted files would hide what a clone lacks.

  python scripts/serve/release.py            # check, print, write results/release/clean_clone.json; exit 1 if blocking
  python scripts/serve/release.py --tests    # also run the test suite inside the clone

Findings are split, because a gate that treats everything as fatal gets switched off in the week it matters:

  BLOCKING  the clone does not reproduce what the report says it does
            - the lockfile is missing or has an unpinned line; the image's requirements stray from it
            - any requirements file names a VCS or URL dependency
            - a config, compose file, Dockerfile or code literal carries an absolute machine path
            - a data manifest lists a file that is missing or whose bytes differ
            - a module fails to import once data/ is moved away (it worked only because the data happened to exist)
            - the committed model card is stale against the committed results
            - a tracked file looks like a secret (.env, a key, a token file)
            - with --tests: the test suite fails in the clone
  WARNING   a gap in the report, not a broken repository
            - an unpinned requirements file other than the lock; README without setup instructions
            - an absolute path in a comment, a job script's usage notes or a document
            - a generated draft that is stale; a large tracked file
            - every NOT MEASURED item in the model card
"""
import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parent
OUT = REPO / "results" / "release" / "clean_clone.json"
LOCK = "requirements-vm03.lock"
ABS_PATH = re.compile(r"(?<![\w.])(/storage/|/Users/|/home/|/mnt/|[A-Za-z]:\\\\)")
VCS = re.compile(r"(git\+|hg\+|svn\+|bzr\+|https?://|\s@\s|file:)", re.I)
SECRET_NAMES = re.compile(r"(^|/)(\.env(\..+)?|id_(rsa|ed25519)|.*\.pem|.*token.*\.txt|credentials\.json)$", re.I)
LARGE = 50 * 2**20


class Findings:
    def __init__(self):
        self.blocking, self.warnings, self.checks = [], [], {}

    def block(self, check, detail):
        self.blocking.append({"check": check, "detail": detail})

    def warn(self, check, detail):
        self.warnings.append({"check": check, "detail": detail})

    def done(self, check):
        b = sum(f["check"] == check for f in self.blocking)
        w = sum(f["check"] == check for f in self.warnings)
        self.checks[check] = "blocking" if b else ("warning" if w else "ok")


def requirement_lines(path):
    """(line number, text) of every requirement line, comments and blanks removed."""
    out = []
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split(" #")[0].strip()
        if line and not line.startswith("#"):
            out.append((n, line))
    return out


def check_requirements(root, f):
    lock = root / LOCK
    if not lock.is_file():
        f.block("requirements", f"{LOCK} is missing")
    else:
        for n, line in requirement_lines(lock):
            if line.startswith("-"):
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.\-\[\],]+==[^\s=<>~!]+", line):
                f.block("requirements", f"{LOCK}:{n} is not an exact pin: {line}")
    lock_pins = {l for _, l in requirement_lines(lock)} if lock.is_file() else set()
    for req in sorted(root.glob("requirements*.txt")):
        lines = requirement_lines(req)
        for n, line in lines:
            if line.startswith("--extra-index-url") or line.startswith("--index-url"):
                continue
            if VCS.search(line):
                f.block("requirements", f"{req.name}:{n} is a VCS or URL dependency: {line}")
        if req.name == "requirements-serve.txt":
            stray = [l for _, l in lines if not l.startswith("-") and l not in lock_pins]
            if stray:
                f.block("requirements", f"requirements-serve.txt has {len(stray)} line(s) not in {LOCK}, e.g. {stray[:2]}")
        else:
            loose = [l for _, l in lines if not l.startswith("-") and "==" not in l]
            if loose:
                f.warn("requirements", f"{req.name} has {len(loose)} unpinned line(s), e.g. {loose[:3]}; "
                                       f"the reproducible environment is {LOCK}")
    f.done("requirements")


def code_literals(path):
    """String constants in a Python file, docstrings excluded (they are documentation, not behaviour).

    Raises SyntaxError for a file that does not parse; the caller reports it as its own finding.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
                docstrings.add(id(first.value))
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


def check_paths(root, f):
    configs = [*root.glob("configs/**/*.y*ml"), *root.glob("compose*.y*ml"), *root.glob("docker/**/Dockerfile*")]
    for p in configs:
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if ABS_PATH.search(line.split("#")[0]):
                f.block("absolute_paths", f"{p.relative_to(root)}:{n}: {line.strip()[:100]}")
    for p in sorted([*root.glob("scripts/**/*.py"), *root.glob("tests/**/*.py")]):
        try:
            literals = code_literals(p)
        except SyntaxError as e:
            f.block("absolute_paths", f"{p.relative_to(root)}:{e.lineno}: does not parse ({e.msg})")
            continue
        for n, value in literals:
            if ABS_PATH.match(value):
                f.block("absolute_paths", f"{p.relative_to(root)}:{n}: string literal {value[:80]!r}")
    for p in sorted(root.glob("slurm/*.sbatch")):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if ABS_PATH.search(line):
                (f.warn if line.lstrip().startswith("#") else f.block)(
                    "absolute_paths", f"{p.relative_to(root)}:{n}: {line.strip()[:100]}")
    f.done("absolute_paths")


def check_manifests(root, f):
    manifests = sorted(root.glob("data/*/MANIFEST.txt"))
    if not manifests:
        f.block("manifests", "no data/*/MANIFEST.txt in the clone")
    files = 0
    for m in manifests:
        for line in m.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            digest, rel = line.split(maxsplit=1)
            p = m.parent / rel
            files += 1
            if not p.is_file():
                f.block("manifests", f"{m.relative_to(root)} lists {rel}, which the clone does not have")
            elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                f.block("manifests", f"{m.relative_to(root)}: {rel} differs from its recorded sha256")
    f.checks["manifest_files"] = files
    f.done("manifests")


def repo_modules(root):
    mods = []
    for p in sorted((root / "scripts").rglob("*.py")):
        rel = p.relative_to(root / "scripts").with_suffix("")
        if rel.name == "__init__" or rel.name.startswith("test_"):
            continue
        mods.append(".".join(rel.parts))
    return mods


def check_imports_without_data(root, f, python=sys.executable):
    """Import every module under scripts/ with data/ moved away. A third-party package missing from the interpreter
    running this check is reported as skipped, not as a failure of the repository."""
    data, hidden = root / "data", root / ".data_hidden"
    if data.exists():
        data.rename(hidden)
    skipped = []
    try:
        for mod in repo_modules(root):
            code = f"import sys; sys.path.insert(0, 'scripts'); import {mod}"
            r = subprocess.run([python, "-c", code], cwd=root, capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                continue
            err = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "no output"
            missing = re.search(r"No module named '([\w.]+)'", err)
            if missing and not (root / "scripts" / missing.group(1).split(".")[0]).exists() \
                    and not (root / "scripts" / f"{missing.group(1).split('.')[0]}.py").exists():
                skipped.append(f"{mod} ({missing.group(1)})")
            else:
                f.block("imports_without_data", f"{mod}: {err[:160]}")
    finally:
        if hidden.exists():
            hidden.rename(data)
    f.checks["imports_skipped_missing_packages"] = skipped
    f.done("imports_without_data")


def check_generated(root, f, python=sys.executable):
    r = subprocess.run([python, "scripts/serve/modelcard.py", "--check"], cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        f.block("generated_documents", "MODEL_CARD.md is stale or does not build: " + (r.stdout + r.stderr).strip()[-200:])
    for script, doc in (("scripts/report_week3.py", "docs/report_draft_week3.md"),
                        ("scripts/report_week4.py", "docs/report_draft_week4.md")):
        if not (root / script).is_file():
            continue
        r = subprocess.run([python, script], cwd=root, capture_output=True, text=True)
        target = root / doc
        if r.returncode != 0:
            f.warn("generated_documents", f"{script} fails: {r.stderr.strip()[-160:]}")
        elif not target.is_file() or target.read_text(encoding="utf-8").strip() != r.stdout.strip():
            f.warn("generated_documents", f"{doc} is stale: regenerate with {script} --write")
    f.done("generated_documents")


def check_hygiene(root, f, tracked):
    for rel in tracked:
        if SECRET_NAMES.search(rel):
            f.block("hygiene", f"{rel} looks like a secret and is tracked")
        p = root / rel
        if p.is_file() and p.stat().st_size > LARGE:
            f.warn("hygiene", f"{rel} is {p.stat().st_size / 2**20:.0f} MB")
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").is_file() else ""
    if LOCK not in readme:
        f.warn("hygiene", f"README.md does not tell a reader how to set up the environment (no mention of {LOCK})")
    f.done("hygiene")


def check_card_gaps(root, f):
    sys.path.insert(0, str(root / "scripts"))
    try:
        from serve import modelcard
        _, gaps = modelcard.build(root)
    finally:
        sys.path.pop(0)
    for g in gaps:
        f.warn("not_measured", g)
    f.done("not_measured")


def check_tests(root, f, python=sys.executable):
    r = subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=root,
                       capture_output=True, text=True, timeout=1800)
    tail = (r.stdout.strip().splitlines() or ["no output"])[-1]
    f.checks["tests_summary"] = tail
    if r.returncode != 0:
        f.block("tests", f"the test suite fails in the clone: {tail}")
    f.done("tests")


def clone(source, dest):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    subprocess.run(["git", "clone", "--quiet", "--no-local", "--no-checkout", str(source), str(dest)], check=True)
    subprocess.run(["git", "checkout", "--quiet", commit], cwd=dest, check=True)
    tracked = subprocess.check_output(["git", "ls-files"], cwd=dest, text=True).splitlines()
    return commit, tracked


def run(source=REPO, tests=False, python=sys.executable):
    f = Findings()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True)
    with tempfile.TemporaryDirectory(prefix="clean_clone_") as tmp:
        root = Path(tmp) / "clone"
        commit, tracked = clone(source, root)
        check_requirements(root, f)
        check_paths(root, f)
        check_manifests(root, f)
        check_imports_without_data(root, f, python)
        check_generated(root, f, python)
        check_hygiene(root, f, tracked)
        check_card_gaps(root, f)
        if tests:
            check_tests(root, f, python)
    return {"task": "4.7", "commit": commit, "tracked_files": len(tracked),
            "uncommitted_changes_not_checked": [l[3:] for l in dirty.splitlines()],
            "verdict": "BLOCKED" if f.blocking else "PASS", "checks": f.checks,
            "blocking": f.blocking, "warnings": f.warnings}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", action="store_true", help="also run the test suite in the clone")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    report = run(tests=args.tests)
    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"clean clone of {report['commit'][:7]} ({report['tracked_files']} files): {report['verdict']}")
    for name, status in report["checks"].items():
        if isinstance(status, str):
            print(f"  {name:22} {status}")
    for b in report["blocking"]:
        print(f"BLOCKING  [{b['check']}] {b['detail']}")
    for w in report["warnings"]:
        print(f"warning   [{w['check']}] {w['detail']}")
    return 1 if report["blocking"] else 0


if __name__ == "__main__":
    sys.exit(main())
