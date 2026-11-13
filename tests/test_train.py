"""CPU-only checks for the parts of scripts/train.py that do not need a GPU.

    pip install pytest
    python -m pytest tests -q
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import train  # noqa: E402


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


# --- config ----------------------------------------------------------------------------------------

def test_config_merges_defaults_and_overrides(tmp_path):
    cfg_file = write(tmp_path / "c.yaml", "run_name: x\nlora:\n  r: 8\n")
    cfg = train.load_config(cfg_file, ["train.lr=1e-4", "eval.every_steps=0"])
    assert cfg["lora"]["r"] == 8
    assert cfg["lora"]["alpha"] == train.DEFAULTS["lora"]["alpha"]  # untouched default kept
    assert cfg["train"]["lr"] == pytest.approx(1e-4)
    assert isinstance(cfg["train"]["lr"], float)  # PyYAML alone would give the string "1e-4"
    assert cfg["eval"]["every_steps"] == 0


def test_numbers_get_the_type_of_their_default(tmp_path):
    cfg = train.load_config(write(tmp_path / "c.yaml", "run_name: x\ntrain:\n  epochs: 3.0\n  max_grad_norm: 1\n"), [])
    assert cfg["train"]["epochs"] == 3 and isinstance(cfg["train"]["epochs"], int)
    assert isinstance(cfg["train"]["max_grad_norm"], float)


@pytest.mark.parametrize("yaml_text, overrides", [
    ("run_name: x\nlerning_rate: 1\n", []),           # unknown top-level key
    ("run_name: x\ntrain:\n  lr_typo: 1\n", []),       # unknown nested key
    ("lora:\n  r: 8\n", []),                           # run_name missing
    ("run_name: x\ntargets: everything\n", []),        # invalid targets
    ("run_name: x\nlora: 16\n", []),                   # mapping expected
    ("run_name: x\n", ["train.lr"]),                   # --set without '='
    ("run_name: x\n", ["train.epochs=two"]),           # not a number
    ("run_name: x\n", ["train.epochs=2.5"]),           # not a whole number
    ("run_name: x\ntrain:\n  lr: true\n", []),         # bool where a number is expected
])
def test_config_rejects_mistakes(tmp_path, yaml_text, overrides):
    with pytest.raises(SystemExit):
        train.load_config(write(tmp_path / "c.yaml", yaml_text), overrides)


@pytest.mark.parametrize("name", ["run_a", "run_b", "ablation_answer_only"])
def test_committed_configs_load(name):
    cfg = train.load_config(REPO / "configs" / f"{name}.yaml", [])
    assert cfg["run_name"]
    assert cfg["max_pixels"] == 640000


def test_split_path_follows_targets(tmp_path):
    cfg = train.load_config(write(tmp_path / "c.yaml", "run_name: x\ntargets: answer_only\n"), [])
    assert train.split_path(cfg, "train").as_posix().endswith("data/v2.0/image_only/answer_only/train.jsonl")


# --- checkpoints -----------------------------------------------------------------------------------

def test_rotation_keeps_newest_steps_and_every_epoch(tmp_path):
    ck = tmp_path / "checkpoints"
    for name in ("step-00050", "step-00100", "step-00130", "step-00150.tmp", "epoch-1"):
        (ck / name).mkdir(parents=True)
    assert [p.name for p in train.checkpoints(tmp_path)] == ["step-00050", "step-00100", "step-00130"]
    train.rotate(tmp_path, keep=2)
    assert sorted(p.name for p in ck.iterdir()) == ["epoch-1", "step-00100", "step-00130", "step-00150.tmp"]


# --- data fingerprint ------------------------------------------------------------------------------

@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    """A tiny dataset in the v2.0 layout with a correct MANIFEST.txt; train.REPO points at it."""
    monkeypatch.setattr(train, "REPO", tmp_path)
    root = tmp_path / "data" / "vX"
    rows = {"train": [{"id": "t1", "image": "images/train/1.png"}], "val": [{"id": "v1", "image": "images/val/2.png"}]}
    lines = []
    for split, split_rows in rows.items():
        f = write(root / "image_only" / f"{split}.jsonl", "".join(json.dumps(r) + "\n" for r in split_rows))
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  image_only/{split}.jsonl")
        for r in split_rows:
            (root / r["image"]).parent.mkdir(parents=True, exist_ok=True)
            (root / r["image"]).write_bytes(b"png-bytes-" + r["id"].encode())
    write(root / "MANIFEST.txt", "\n".join(lines) + "\n")
    cfg = {**train.DEFAULTS, "data_dir": "data/vX", "run_name": "x"}
    return cfg, rows, root


def test_fingerprint_accepts_matching_data(fake_data):
    cfg, rows, _ = fake_data
    fp = train.data_fingerprint(cfg, rows)
    assert fp["train_file"] == "data/vX/image_only/train.jsonl"
    assert fp["images"] == 2 and len(fp["images_sha256"]) == 64


def test_fingerprint_changes_when_an_image_changes(fake_data):
    cfg, rows, root = fake_data
    before = train.data_fingerprint(cfg, rows)["images_sha256"]
    (root / "images/val/2.png").write_bytes(b"edited")
    assert train.data_fingerprint(cfg, rows)["images_sha256"] != before


@pytest.mark.parametrize("damage", ["edit", "crlf", "unlisted"])
def test_fingerprint_rejects_changed_data(fake_data, damage):
    cfg, rows, root = fake_data
    f = root / "image_only" / "train.jsonl"
    if damage == "edit":
        f.write_bytes(f.read_bytes() + b"{}\n")
    elif damage == "crlf":  # what a Windows checkout without .gitattributes does
        f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
    else:
        m = root / "MANIFEST.txt"
        m.write_text("".join(l + "\n" for l in m.read_text().splitlines() if "train.jsonl" not in l))
    with pytest.raises(SystemExit):
        train.data_fingerprint(cfg, rows)


def test_committed_v2_data_matches_its_manifest():
    """Fails on a checkout whose data files differ from MANIFEST.txt (e.g. CRLF line endings)."""
    cfg = train.load_config(REPO / "configs" / "run_a.yaml", [])
    rows = {s: train.read_split(cfg, s) for s in ("train", "val")}
    fp = train.data_fingerprint(cfg, rows)
    assert fp["images"] == len({r["image"] for s in rows.values() for r in s})


# --- environment lock ------------------------------------------------------------------------------

def test_lock_parsing_and_drift(tmp_path):
    lock = write(tmp_path / "x.lock", "# header\nTorch==2.1\nqwen_vl_utils==0.0.14\n\nwandb==0.31.0\n")
    pins = train.read_lock(lock)
    assert pins == {"torch": "2.1", "qwen-vl-utils": "0.0.14", "wandb": "0.31.0"}
    drift = train.env_drift(pins, {"torch": "2.1", "qwen-vl-utils": "0.0.15"})
    assert drift == {"qwen-vl-utils": ("0.0.14", "0.0.15"), "wandb": ("0.31.0", None)}
    assert train.env_drift(pins, {"torch": "2.1", "qwen-vl-utils": "0.0.14", "wandb": "0.31.0"}) == {}


def test_committed_lock_is_well_formed():
    pins = train.read_lock()
    assert {"torch", "transformers", "peft", "bitsandbytes", "wandb"} <= set(pins)


def test_git_state_reports_a_commit():
    commit, dirty = train.git_state()
    if commit is None:
        pytest.skip("not a git checkout")
    assert len(commit) == 40 and isinstance(dirty, list)
