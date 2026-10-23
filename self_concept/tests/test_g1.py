"""SC-G1: frozen rules on planted z profiles; items; the read path on a tiny Qwen3.5 with fake answers."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from self_concept.g1_self import analyze as A
from self_concept.g1_self import items as I

REPO = Path(__file__).resolve().parents[2]


def _rows(root, plant, text_same=False, cal=True):
    rng = np.random.default_rng(0)
    rows = []
    for c in I.CONDS:
        for j in range(32):
            own = {k: float(rng.normal(0, 0.3)) for k in I.SETS}
            for k, v in plant.get(c, {}).items():
                own[k] += v
            text = dict(own) if text_same else {k: float(rng.normal(0, 0.3)) for k in I.SETS}
            z = {"own": own, "text": text}
            rows.append({"key": f"{c}{j}", "cond": c, "tid": f"d{j % 8}", "n_pos_own": 3, "n_pos_text": 3,
                         "z": {"cb": z, "logit": z}})
    (root / "reads").mkdir(parents=True)
    (root / "reads" / "reads_all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "cal").mkdir()
    (root / "cal" / "analysis.json").write_text(json.dumps([{"model": "qwen9", "lens": "cb", "CAL_PASS": cal}]))
    return root / "cal"


def test_plan_predictions(tmp_path):
    cal = _rows(tmp_path, {"persona": {"CHAR": 2.0}, "trained": {"TRAIT": 1.5}})
    out = A.analyse(tmp_path, cal)
    assert out["labels"]["costume"] == "costume over the self; self-specific"
    assert out["labels"]["trained"] == "the trained trait enters the self without role content; self-specific"
    assert out["labels"]["record"] == "nothing distinctive"


def test_alternatives(tmp_path):
    cal = _rows(tmp_path, {"persona": {"CHAR": 2.0, "ASSIST": -1.5}, "trained": {"TRAIT": 1.5, "ROLE": 1.0},
                           "record": {"TRAIT": 1.0}}, text_same=True)
    out = A.analyse(tmp_path, cal)
    assert out["labels"]["costume"] == "the character replaces the assistant; follows the text"
    assert out["labels"]["trained"].startswith("the trait enters with role content")
    assert out["labels"]["record"].startswith("the record's content enters I (TRAIT)")


def test_gate_makes_descriptive(tmp_path):
    cal = _rows(tmp_path, {"persona": {"CHAR": 2.0}}, cal=False)
    out = A.analyse(tmp_path, cal)
    assert all(v.startswith("descriptive") for v in out["labels"].values())


def test_items_and_tiny_read(tmp_path):
    it = I.items()
    assert len(it) == 6 * 8 * 4 and len({x["key"] for x in it}) == len(it)
    assert next(x for x in it if x["cond"] == "persona")["messages"][0]["content"] == I.PERSONA
    env = {**os.environ, "V1_FAKE_GEN": "1", "HF_HUB_OFFLINE": "1"}
    r = subprocess.run([sys.executable, "-m", "self_concept.g1_self.evalrun", "--model", "x", "--conds", "default",
                        "trained", "--out", str(tmp_path / "evals")], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    r = subprocess.run([sys.executable, "-m", "self_concept.g1_self.read", "--model", "x", "--conds", "default", "trained",
                        "--evals", str(tmp_path / "evals"), "--out", str(tmp_path / "reads"), "--tiny"], cwd=REPO,
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    rows = [json.loads(x) for x in (tmp_path / "reads/reads_default_trained.jsonl").read_text().splitlines()]
    assert len(rows) == 64 and all(x["n_pos_own"] >= 1 and x["n_pos_own"] == x["n_pos_text"] for x in rows)
    assert set(rows[0]["z"]["cb"]["own"]) == set(I.SETS)
