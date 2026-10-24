"""SC-M4 (self_concept/m4_inject/DESIGN.md): directions, prompts, the frozen analysis (planted effects give each verdict),
the run path on a tiny Qwen3.5, the launcher."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from self_concept.m4_inject import analyze as A
from self_concept.m4_inject import items as I

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m4_inject"
NB = 1000
DIRS = ["probe", "probe_perp", "dom", "dom_perp"] + A.RAND


def test_directions_and_inputs():
    z = np.load(H / "data" / "dirs.npz")
    meta = json.loads((H / "data" / "dirs.json").read_text())
    assert abs(np.linalg.norm(z["u_probe"]) - 1) < 1e-4 and abs(np.linalg.norm(z["u_dom"]) - 1) < 1e-4
    assert z["rand"].shape == (8, 4096) and meta["layer"] == 16 and meta["gap"]["probe"] > 0
    design = (H / "DESIGN.md").read_text()
    for p, s in json.loads((H / "data" / "sha256.json").read_text()).items():
        assert s in design, p


def test_prompts():
    assert len(I.self_prompts()) == 16 and len(I.just_prompts()) == 16 and len(I.free_prompts()) == 16
    assert all(set(p["amap"].values()) == {"unilateral", "report"} for p in I.self_prompts())


def _logit(x):
    return np.log(x / (1 - x))


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _rows(c0, shift, act0=0.0, act_shift=None, coh=None, noise=0.15, seed=0):
    """c0: report content logit at k = 0; shift(dir, k) -> logit shift; act_shift(dir, k) -> lo shift."""
    rng = np.random.default_rng(seed)
    act_shift = act_shift or (lambda d, k: 0.0)
    rows = []
    for d, k in [("none", 0.0)] + [(d, k) for d in DIRS for k in A.DOSES]:
        s = 0.0 if d == "none" else shift(d, k)
        for st in range(8):
            for o in range(2):
                rows.append({"block": "self", "pid": f"s{st}o{o}", "stem": st, "order": o, "dir": d, "k": k,
                             "p_uni": float(_sig(c0 + s + rng.normal(0, noise))),
                             "top1_letter": True if coh is None else coh(d, k)})
        a = 0.0 if d == "none" else act_shift(d, k)
        for t in range(16):
            rows.append({"block": "just", "pid": f"t{t}", "dir": d, "k": k, "lo": act0 + a + rng.normal(0, noise)})
    return rows


REAL = {"probe", "probe_perp", "dom", "dom_perp"}


def _act(d, k):
    return 2.0 * k if d in REAL else 0.1


def _data(nw_shift, nz_shift, act=_act, coh=None):
    return {"NW": _rows(-2.0, nw_shift, 1.0, act, coh, seed=1), "NZ": _rows(-3.0, nz_shift, -1.0, act, coh, seed=2),
            "VN": _rows(-3.0, nz_shift, -1.0, act, coh, seed=3)}


def test_routes_when_present():
    o = A.analyse(_data(lambda d, k: 2.5 * k if d in REAL else 0.1, lambda d, k: 0.3 * k if d in REAL else 0.1), NB)
    assert o["verdict"] == "routes when present", o["E"]["probe|1.0"]


def test_writes_the_answer():
    o = A.analyse(_data(lambda d, k: 2.0 * k if d in REAL else 0.1, lambda d, k: 2.0 * k if d in REAL else 0.1), NB)
    assert o["verdict"] == "writes the answer"


def test_writes_only_through_answer_words_is_not_routing():
    # E is big for probe but vanishes once the answer words are projected out
    nw = lambda d, k: (2.5 * k if d in ("probe", "dom") else 0.0)  # noqa: E731
    nz = lambda d, k: (0.3 * k if d in ("probe", "dom") else 0.0)  # noqa: E731
    o = A.analyse(_data(nw, nz), NB)
    assert o["verdict"] != "routes when present"


def test_no_effect_on_report():
    o = A.analyse(_data(lambda d, k: 0.0, lambda d, k: 0.0), NB)
    assert o["verdict"] == "no effect on report"


def test_direction_gate():
    o = A.analyse(_data(lambda d, k: 2.5 * k, lambda d, k: 0.0, act=lambda d, k: 0.0), NB)
    assert not o["gate_direction"] and o["verdict"] == "not read"


def test_coherence_gate():
    o = A.analyse(_data(lambda d, k: 2.5 * k if d in REAL else 0.0, lambda d, k: 0.0,
                        coh=lambda d, k: not (d == "probe" and k >= 1)), NB)
    assert not o["gate_coherence"] and o["verdict"] == "not read"


def test_tiny_run_end_to_end(tmp_path):
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    for cell in ("NW_s1", "NZ_s1"):
        r = subprocess.run([sys.executable, "-m", "self_concept.m4_inject.run", "--cell", cell, "--model", "x", "--out",
                            str(tmp_path), "--tiny"], cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(x) for x in (tmp_path / "reads_NW_s1.jsonl").read_text().splitlines()]
    assert len(rows) == (16 + 16) * (1 + 12 * 3) and all(0 <= r["p_uni"] <= 1 for r in rows if r["block"] == "self")
    o = A.analyse(A.load(tmp_path), 200)
    assert "verdict" in o and "probe|1.0" in o["E"]


def test_launch_dry_run():
    env = {**os.environ, "SC_M4_LAUNCH_DRY": "1"}
    cmd = subprocess.run(["bash", str(H / "launch.sh")], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True).stdout
    assert "--name sc-m4" in cmd and "--max-minutes 70" in cmd and "--max-cost 1.63" in cmd and "V1_ON_POD=1" in cmd
    assert "--sync-extra" not in cmd and "--exclude data/adapters*" in cmd
    b = subprocess.run(["bash", "self_concept/common/book_pod.sh", "--bound", "70", "1.63"], cwd=ROOT,
                       capture_output=True, text=True, check=True).stdout.strip()
    assert float(b) <= 2.5
