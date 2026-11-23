"""Task 4.4 on the laptop: the image installs from the lockfile and resolves nothing, weights are mounted and never
baked, and healthy means the served merge is VERIFIED. Docker itself is not needed: these read the files a build
would use, so a change that breaks one of the three rules fails here before anyone builds an image.
"""
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from serve import healthcheck  # noqa: E402

DOCKERFILE = (REPO / "docker/Dockerfile").read_text(encoding="utf-8")
SERVE_REQS = (REPO / "requirements-serve.txt").read_text(encoding="utf-8").splitlines()
LOCK = (REPO / "requirements-vm03.lock").read_text(encoding="utf-8").splitlines()


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(lines):
    return {canonical(l.split("==")[0]): l.strip() for l in lines if "==" in l and not l.lstrip().startswith("#")}


# --- 1. installs from the lockfile and resolves nothing ---------------------------------------------------------

def test_every_serving_pin_is_a_verbatim_line_of_the_lock():
    serve, lock = pins(SERVE_REQS), pins(LOCK)
    assert serve and all(lock.get(n) == line for n, line in serve.items()), \
        [line for n, line in serve.items() if lock.get(n) != line]


def test_the_serving_stack_is_in_it_at_the_measured_versions():
    serve = pins(SERVE_REQS)
    for name, version in {"torch": "2.14.1+cu126", "transformers": "5.22.1", "accelerate": "1.15.0",
                          "fastapi": "0.165.0", "uvicorn": "0.54.0", "pillow": "13.0.0", "safetensors": "0.8.0"}.items():
        assert serve[name] == f"{name}=={version}", name


def test_training_only_packages_stay_out_of_the_image():
    serve = pins(SERVE_REQS)
    assert not {"wandb", "trl", "peft", "datasets", "bitsandbytes", "gptqmodel", "optimum"} & set(serve)


def test_nothing_unpinned_and_only_the_cu126_index():
    lines = [l.strip() for l in SERVE_REQS if l.strip() and not l.lstrip().startswith("#")]
    options = [l for l in lines if l.startswith("-")]
    assert options == ["--extra-index-url https://download.pytorch.org/whl/cu126"]
    assert all("==" in l for l in lines if not l.startswith("-"))
    assert not any(re.search(r"git\+|https?://|\s@\s|>=|<=|~=", l) for l in lines if not l.startswith("-"))


def test_the_image_installs_with_no_dependency_resolution_and_checks_it():
    run = re.search(r"RUN pip install (.+)", DOCKERFILE).group(1)
    assert "--no-deps -r requirements-serve.txt" in run and "pip check" in run
    assert DOCKERFILE.count("pip install") == 1  # no second install that could resolve


def test_the_base_image_python_is_the_lock_interpreter():
    lock_python = re.search(r"Python (\d+\.\d+\.\d+)", "\n".join(LOCK[:3])).group(1)
    assert re.search(r"^FROM python:([0-9.]+)-", DOCKERFILE, re.M).group(1) == lock_python == "3.12.3"


# --- 2. weights are mounted, never baked ---------------------------------------------------------------------------

def test_the_image_copies_code_and_requirements_only():
    copies = re.findall(r"^COPY (\S+)", DOCKERFILE, re.M)
    assert copies == ["requirements-serve.txt", "scripts/"]
    assert not re.search(r"^ADD ", DOCKERFILE, re.M)                       # no ADD: nothing fetched or unpacked
    assert not re.search(r"^RUN .*(huggingface-cli|snapshot_download|wget|curl)", DOCKERFILE, re.M)  # no download


def test_dockerignore_admits_only_scripts_and_the_requirements():
    rules = [l.strip() for l in (REPO / ".dockerignore").read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert rules[0] == "*" and set(r for r in rules if r.startswith("!")) == {"!requirements-serve.txt", "!scripts/"}


def test_compose_mounts_the_merge_read_only_and_requires_it():
    yaml = pytest.importorskip("yaml")
    svc = yaml.safe_load((REPO / "compose.yaml").read_text())["services"]["chartqa"]
    (vol,) = svc["volumes"]
    assert vol["target"] == "/model" and vol["read_only"] is True and vol["type"] == "bind"
    assert vol["source"].startswith("${MODEL_DIR:?")  # compose refuses to start without the merge
    assert svc["build"] == {"context": ".", "dockerfile": "docker/Dockerfile"}
    assert svc["ports"] == ["127.0.0.1:8000:8000"]
    gpu = svc["deploy"]["resources"]["reservations"]["devices"][0]
    assert gpu["driver"] == "nvidia" and gpu["capabilities"] == ["gpu"]


def test_the_entrypoint_serves_the_mount_with_the_trained_output_style():
    entry = json.loads(re.search(r"ENTRYPOINT (\[.*?\])", DOCKERFILE.replace("\\\n", ""), re.S).group(1))
    assert entry[:2] == ["python", "scripts/serve/app.py"]
    args = dict(zip(entry[2::2], entry[3::2]))
    assert args == {"--model": "/model", "--host": "0.0.0.0", "--port": "8000", "--targets": "reasoning"}
    assert "--allow-unverified" not in entry


def test_the_image_runs_offline_and_not_as_root():
    assert "HF_HUB_OFFLINE=1" in DOCKERFILE and "TRANSFORMERS_OFFLINE=1" in DOCKERFILE
    assert re.search(r"^USER serve$", DOCKERFILE, re.M)


# --- 3. healthy means VERIFIED -------------------------------------------------------------------------------------

def test_the_healthcheck_runs_the_manifest_check():
    assert re.search(r'HEALTHCHECK .*\n?.*CMD \["python", "scripts/serve/healthcheck.py"\]', DOCKERFILE)


GOOD = {"status": "ok", "model": {"run": "run_a_r16", "status": "VERIFIED", "adapter_sha256": "9d9b3d3a5f8d",
                                  "merged_files_sha256": {"model.safetensors": "ab"}}}


@pytest.mark.parametrize("payload, ok", [
    (GOOD, True),
    ({**GOOD, "status": "serving_unverified_checkpoint"}, False),
    ({**GOOD, "model": {**GOOD["model"], "status": "FAILED_ANSWER_CHECK"}}, False),
    ({**GOOD, "model": {**GOOD["model"], "merged_files_sha256": {}}}, False),
    ({}, False),
])
def test_healthy_only_when_the_served_merge_is_verified(payload, ok):
    assert healthcheck.healthy(payload)[0] is ok


def test_the_healthcheck_fails_when_nothing_answers():
    assert healthcheck.main(["--url", "http://127.0.0.1:9"]) == 1


def test_the_healthcheck_reads_the_real_service_health():
    from serve import service as svc
    manifest = {"status": "VERIFIED", "run": "run_a_r16", "adapter_sha256": "ab" * 32,
                "merged_files_sha256": {"model.safetensors": "cd" * 32}}
    assert healthcheck.healthy(svc.Service(object(), manifest, "cot").health())[0]
    unverified = svc.Service(object(), {**manifest, "status": "UNVERIFIED"}, "cot", allow_unverified=True)
    assert not healthcheck.healthy(unverified.health())[0]


# --- the code in the image ----------------------------------------------------------------------------------------

def test_serving_modules_import_without_the_model_stack():
    """The build's import check is meaningful only if module import is cheap and torch-free; heavy imports stay lazy."""
    code = ("import sys; sys.path.insert(0, 'scripts'); "
            "from serve import app, service, decoding, generation, healthcheck; "
            "heavy = [m for m in ('torch', 'transformers', 'fastapi', 'uvicorn', 'peft') if m in sys.modules]; "
            "print(heavy); sys.exit(1 if heavy else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_everything_the_app_imports_lives_under_scripts():
    """The image copies scripts/ only: the serving modules must not import repository code from anywhere else."""
    for mod in ("serve.app", "serve.service", "serve.decoding", "serve.generation", "serve.healthcheck",
                "evaluate", "train", "smoke_test", "scoring", "check_teacher"):
        path = Path(importlib.import_module(mod).__file__).resolve()
        assert path.is_relative_to(REPO / "scripts"), path
