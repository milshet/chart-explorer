"""Task 4.7: the clean-clone check finds each kind of defect, classifies it, and the Week 4 draft is current."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import report_week4  # noqa: E402
from serve import release as R  # noqa: E402


def write(root, files):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text) if isinstance(text, bytes) else p.write_text(text, encoding="utf-8")
    return root


def findings(check, root, *args):
    f = R.Findings()
    check(root, f, *args)
    return f


def details(items):
    return " | ".join(x["detail"] for x in items)


# --- requirements ---------------------------------------------------------------------------------------------------

def test_a_fully_pinned_lock_passes(tmp_path):
    write(tmp_path, {R.LOCK: "# header\ntorch==2.14.1+cu126\nnumpy==2.5.4\n"})
    f = findings(R.check_requirements, tmp_path)
    assert not f.blocking and f.checks["requirements"] == "ok"


@pytest.mark.parametrize("line", ["numpy>=2.0", "numpy", "numpy~=2.5", "numpy==2.*,<3"])
def test_an_unpinned_lock_line_blocks(tmp_path, line):
    write(tmp_path, {R.LOCK: f"torch==2.14.1\n{line}\n"})
    assert "not an exact pin" in details(findings(R.check_requirements, tmp_path).blocking)


def test_a_missing_lock_blocks(tmp_path):
    assert "is missing" in details(findings(R.check_requirements, tmp_path).blocking)


@pytest.mark.parametrize("line", ["mylib @ git+https://github.com/x/y.git", "git+https://github.com/x/y.git",
                                  "https://example.com/pkg-1.0.tar.gz", "file:./wheels/pkg.whl"])
def test_a_vcs_or_url_dependency_blocks_in_any_requirements_file(tmp_path, line):
    write(tmp_path, {R.LOCK: "a==1\n", "requirements.txt": f"{line}\n"})
    assert "VCS or URL" in details(findings(R.check_requirements, tmp_path).blocking)


def test_an_unpinned_secondary_file_is_only_a_warning(tmp_path):
    write(tmp_path, {R.LOCK: "a==1\n", "requirements.txt": "transformers>=4.49  # floor\npandas\n"})
    f = findings(R.check_requirements, tmp_path)
    assert not f.blocking and "2 unpinned line(s)" in details(f.warnings)


def test_serving_requirements_must_be_lock_lines(tmp_path):
    write(tmp_path, {R.LOCK: "a==1\nb==2\n",
                     "requirements-serve.txt": "--extra-index-url https://download.pytorch.org/whl/cu126\na==1\nb==3\n"})
    f = findings(R.check_requirements, tmp_path)
    assert "not in requirements-vm03.lock" in details(f.blocking) and "VCS" not in details(f.blocking)


# --- absolute paths ---------------------------------------------------------------------------------------------------

def test_an_absolute_path_in_a_config_blocks(tmp_path):
    write(tmp_path, {"configs/run.yaml": "data_dir: /storage/tracka/users/me/data  \nother: x  # /home/me ok in comment\n"})
    f = findings(R.check_paths, tmp_path)
    assert len(f.blocking) == 1 and "configs/run.yaml:1" in details(f.blocking)


def test_an_absolute_path_in_a_code_literal_blocks_but_not_in_a_docstring(tmp_path):
    write(tmp_path, {"scripts/a.py": '"""Example: python a.py --out /storage/x"""\nOUT = "/Users/me/out"\n# /home/me\n'})
    f = findings(R.check_paths, tmp_path)
    assert len(f.blocking) == 1 and "scripts/a.py:2" in details(f.blocking)


def test_an_absolute_path_in_a_job_scripts_comment_is_a_warning(tmp_path):
    write(tmp_path, {"slurm/x.sbatch": "#   M=/storage/me/merged\npython run.py --out /storage/me/out\n"})
    f = findings(R.check_paths, tmp_path)
    assert len(f.warnings) == 1 and len(f.blocking) == 1 and "x.sbatch:2" in details(f.blocking)


def test_a_file_that_does_not_parse_blocks_but_a_literal_naming_an_error_does_not(tmp_path):
    write(tmp_path, {"scripts/broken.py": "def f(:\n", "scripts/ok.py": 'MSG = "SYNTAX ERROR in input"\n'})
    f = findings(R.check_paths, tmp_path)
    assert len(f.blocking) == 1 and "broken.py:1: does not parse" in details(f.blocking)


def test_relative_and_home_relative_paths_pass(tmp_path):
    write(tmp_path, {"configs/run.yaml": "data_dir: data/v2.0\nout: ~/runs\n", "scripts/a.py": 'P = "results/x.json"\n'})
    assert not findings(R.check_paths, tmp_path).blocking


# --- manifests ----------------------------------------------------------------------------------------------------------

def manifest_repo(tmp_path, content=b"hello", listed=None):
    import hashlib
    digest = hashlib.sha256(b"hello").hexdigest()
    return write(tmp_path, {"data/v1/f.txt": content, "data/v1/MANIFEST.txt": f"{digest}  {listed or 'f.txt'}\n"})


def test_matching_manifests_pass(tmp_path):
    f = findings(R.check_manifests, manifest_repo(tmp_path))
    assert not f.blocking and f.checks["manifest_files"] == 1


def test_a_changed_data_file_blocks(tmp_path):
    assert "differs" in details(findings(R.check_manifests, manifest_repo(tmp_path, b"HELLO")).blocking)


def test_a_missing_data_file_blocks(tmp_path):
    assert "does not have" in details(findings(R.check_manifests, manifest_repo(tmp_path, listed="gone.txt")).blocking)


# --- imports without data -------------------------------------------------------------------------------------------

def test_a_module_that_reads_data_at_import_blocks_and_data_is_restored(tmp_path):
    write(tmp_path, {"scripts/good.py": "import json\n", "scripts/bad.py": "open('data/v1/f.txt').read()\n",
                     "scripts/needs_pkg.py": "import some_absent_package_xyz\n", "data/v1/f.txt": "x",
                     "scripts/cli.py": "import sys\nPATH = sys.argv[1]\n"})
    f = findings(R.check_imports_without_data, tmp_path)
    assert details(f.blocking).startswith("bad imports only when data/ exists") and len(f.blocking) == 1
    assert "cli fails to import even with data/" in details(f.warnings)  # runs at import: hygiene, not data
    assert f.checks["imports_skipped_missing_packages"] == ["needs_pkg (some_absent_package_xyz)"]
    assert (tmp_path / "data/v1/f.txt").is_file() and not (tmp_path / ".data_hidden").exists()


# --- hygiene ----------------------------------------------------------------------------------------------------------

def test_tracked_secrets_block_and_a_readme_without_setup_warns(tmp_path):
    write(tmp_path, {"README.md": "# x\n", ".env": "TOKEN=1"})
    f = findings(R.check_hygiene, tmp_path, [".env", "README.md", "keys/id_ed25519"])
    assert ".env looks like a secret" in details(f.blocking) and "id_ed25519" in details(f.blocking)
    assert "does not tell a reader how to set up" in details(f.warnings)


def test_a_readme_that_names_the_lock_passes(tmp_path):
    write(tmp_path, {"README.md": f"pip install -r {R.LOCK}\n"})
    assert not findings(R.check_hygiene, tmp_path, ["README.md"]).warnings


# --- the whole check on this repository -----------------------------------------------------------------------------

def test_the_clone_is_of_the_committed_head_and_has_no_blocking_findings():
    report = R.run()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    assert report["commit"] == head and report["tracked_files"] > 2000
    assert report["verdict"] == "PASS", details(report["blocking"])
    assert {"requirements", "absolute_paths", "manifests", "imports_without_data", "generated_documents",
            "hygiene", "not_measured"} <= set(report["checks"])


def test_the_committed_check_result_passed():
    path = REPO / "results/release/clean_clone.json"
    if not path.is_file():  # the check runs on a commit, so its result lands one commit later
        pytest.skip("no clean-clone result committed yet: run scripts/serve/release.py --tests")
    r = json.loads(path.read_text())
    assert r["verdict"] == "PASS" and r["checks"].get("tests") == "ok"


# --- the Week 4 draft ---------------------------------------------------------------------------------------------------

def test_the_week4_draft_is_current():
    assert (REPO / report_week4.DRAFT).read_text(encoding="utf-8") == report_week4.build()


def test_the_draft_closes_with_what_the_week_did_not_establish():
    text = report_week4.build()
    closing = text.split("## What Week 4 did not establish")[1]
    for needle in ("quantised deployment", "concurrent load", "reliable abstention", "Eval-B", "3.0i", "4.4"):
        assert needle in closing, needle


def test_the_draft_quotes_every_task(tmp_path):
    text = report_week4.build()
    assert all(f"## 4.{i}" in text for i in range(1, 8))
    empty = report_week4.build(tmp_path)
    assert empty.count(report_week4.NM) >= 5
