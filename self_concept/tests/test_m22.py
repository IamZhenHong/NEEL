"""SC-M22 (self_concept/m22_fitdir/DESIGN.md): the split, the frozen analysis on planted effects, the tiny pipeline, the
launcher."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from self_concept.m22_fitdir import analyze as A
from self_concept.m22_fitdir import fit as F
from self_concept.m4_inject import items as I4
from self_concept.m4_inject import run_b as RB

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m22_fitdir"
NB = 400


def test_split():
    pids = [p["pid"] for p in I4.just_prompts()]
    assert set(F.TRAIN) <= set(pids) and len(F.TRAIN) == 8
    for dom in ("mk", "fo", "rd", "gy"):
        assert sum(t.startswith(dom) for t in F.TRAIN) == 2


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _rows(c0, shift, act, seed=0):
    rng = np.random.default_rng(seed)
    dirs = ["fit", "fit_perp"] + A.CTRLS
    conds = [("none", 0.0)] + [(d, k) for d in dirs for k in (1.0, 2.0)]
    out = []
    for p in RB.report_prompts():
        for d, k in conds:
            c = c0 + (0 if d == "none" else shift(d, k)) + rng.normal(0, 0.1)
            out.append({"block": "self", "family": p["family"], "cluster": p["cluster"], "pid": p["pid"],
                        "order": p["order"], "dir": d, "k": k, "p_uni": float(_sig(c)), "top1_letter": True})
    for p in I4.just_prompts():
        for d, k in conds:
            out.append({"block": "just", "pid": p["pid"], "train": p["pid"] in F.TRAIN, "dir": d, "k": k,
                        "lo": (0 if d == "none" else act(d, k)) + rng.normal(0, 0.1)})
    return out


FIT = ("fit", "fit_perp")


def _act(d, k):
    return 2.0 * k if d in FIT else 0.3


def _data(nw, nz, act=_act):
    return {"NW_s1": _rows(-1.6, nw, act, 1), "NZ_s1": _rows(-2.4, nz, act, 2), "VN_s1": _rows(-2.6, nz, act, 3)}


def test_routes():
    o = A.analyse(_data(lambda d, k: 1.5 * k if d in FIT else 0, lambda d, k: 0.1 * k if d in FIT else 0), NB)
    assert o["verdict"] == "routes when present", o["E"]["fit|1.0"]


def test_writes_and_answer_word_control():
    o = A.analyse(_data(lambda d, k: 1.2 * k if d in FIT else 0, lambda d, k: 1.2 * k if d in FIT else 0), NB)
    assert o["verdict"] == "writes the answer"
    # routing that vanishes once the answer words are projected out is not routing
    o2 = A.analyse(_data(lambda d, k: 1.5 * k if d == "fit" else 0, lambda d, k: 0.1 * k if d == "fit" else 0), NB)
    assert o2["verdict"] != "routes when present"


def test_no_effect_and_gate():
    assert A.analyse(_data(lambda d, k: 0.0, lambda d, k: 0.0), NB)["verdict"] == "no effect on report"
    o = A.analyse(_data(lambda d, k: 1.5 * k, lambda d, k: 0.0, act=lambda d, k: 2.0 if d.startswith("shuf") else 0.5), NB)
    assert not o["gate_direction"] and o["verdict"] == "not read"


def test_tiny_pipeline(tmp_path):
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    r = subprocess.run([sys.executable, "-m", "self_concept.m22_fitdir.fit", "--model", "x", "--out", str(tmp_path),
                        "--tiny", "--steps", "3"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    for cell in ("NW_s1", "NZ_s1"):
        r = subprocess.run([sys.executable, "-m", "self_concept.m22_fitdir.read", "--cell", cell, "--model", "x", "--dirs",
                            str(tmp_path / "dirs_fit.npz"), "--out", str(tmp_path), "--tiny"], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    o = A.analyse(A.load(tmp_path), 100)
    assert "fit|1.0" in o["E"]


def test_launch_dry():
    env = {**os.environ, "SC_M22_LAUNCH_DRY": "1"}
    cmd = subprocess.run(["bash", str(H / "launch.sh")], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True).stdout
    assert "--name sc-m22" in cmd and "--max-minutes 80" in cmd and "--max-cost 1.86" in cmd and "V1_ON_POD=1" in cmd
