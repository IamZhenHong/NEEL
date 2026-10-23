"""V1's trainer with SC-N6b's save_at hook: a default run saves exactly what the committed trainer saved, and turning
checkpoints on leaves the final adapter unchanged. Tiny random Qwen3.5 (persona_affect's tiny_q35), V1_TINY_RECIPES=1."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from persona_affect.tests.test_v1 import _short_rows, tiny_q35  # noqa: F401 (fixture)

REPO = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "V1_TINY_RECIPES": "1", "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(REPO)}
BASE_REV = "55dc39f"   # the last commit of persona_affect/v1_value_hook/train.py before the hook


def _rows(d):
    rows = d / "rows"
    rows.mkdir()
    (rows / "W.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _short_rows(32, "W")))
    return rows


def _cli(script, rows, base, out):
    r = subprocess.run([sys.executable, str(script), "--stage", "later", "--value", "N", "--arms", "W", "--rows",
                        str(rows), "--recipe", "d1", "--seed", "1", "--base", str(base), "--out", str(out)],
                       cwd=REPO, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def _tree(d):
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def test_default_run_matches_committed_trainer(tiny_q35, tmp_path):  # noqa: F811
    old = tmp_path / "old_train.py"
    old.write_text(subprocess.run(["git", "show", f"{BASE_REV}:persona_affect/v1_value_hook/train.py"], cwd=REPO,
                                  capture_output=True, text=True, check=True).stdout)
    rows = _rows(tmp_path)
    _cli(old, rows, tiny_q35, tmp_path / "a")
    _cli(REPO / "persona_affect/v1_value_hook/train.py", rows, tiny_q35, tmp_path / "b")
    a, b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    assert set(a) == set(b)
    for k in a:
        if k == "manifest.json":
            ma, mb = (json.loads(x[k]) for x in (a, b))
            for m in (ma, mb):
                for info in m["adapters"].values():
                    info.pop("secs")
            assert ma == mb
        elif k.startswith("train_"):
            la, lb = ([{kk: v for kk, v in json.loads(x).items()} for x in t[k].decode().splitlines()] for t in (a, b))
            assert la == lb
        elif k.endswith("adapter_config.json"):
            ca, cb = (json.loads(x[k]) for x in (a, b))
            diff = {kk: (ca.get(kk), cb.get(kk)) for kk in set(ca) | set(cb) if ca.get(kk) != cb.get(kk)}
            assert set(diff) <= {"target_modules"}, diff        # PEFT writes this set in hash-seed order
            assert sorted(ca["target_modules"]) == sorted(cb["target_modules"])
        else:
            assert a[k] == b[k], k


def test_save_at_leaves_final_adapter_unchanged(tiny_q35, tmp_path, monkeypatch):  # noqa: F811
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from persona_affect.v1_value_hook import train as TR
    monkeypatch.setenv("V1_TINY_RECIPES", "1")
    r = TR.recipe("d1")
    rows = _short_rows(32, "W")
    tok = AutoTokenizer.from_pretrained(tiny_q35)
    shas = {}
    for tag, save_at in (("plain", ()), ("ck", (1,))):
        model = AutoModelForCausalLM.from_pretrained(tiny_q35, dtype=torch.float32)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        _, info = TR.train_one(model, tok, rows, r, 1, "later", "NW_s1", tmp_path / tag, save_at=save_at)
        shas[tag] = info["sha256"]
        if save_at:
            assert set(info["checkpoints"]) == {1}
            assert (tmp_path / tag / "NW_s1_step1" / "adapter_model.safetensors").exists()
        else:
            assert "checkpoints" not in info and not (tmp_path / tag / "NW_s1_step1").exists()
    assert shas["plain"] == shas["ck"]
