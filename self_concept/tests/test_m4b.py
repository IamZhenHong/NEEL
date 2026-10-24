"""SC-M4 amendment 1 (DESIGN.md §9): the rerun's report set, the frozen analyze_b on planted effects, tiny run, launcher."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from self_concept.m4_inject import analyze_b as B
from self_concept.m4_inject import run_b as RB

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m4_inject"
NB = 500


def test_report_set():
    r = RB.report_prompts()
    fam = {f: [p for p in r if p["family"] == f] for f in B.FAMS}
    assert [len(fam[f]) for f in B.FAMS] == [16, 64, 4, 4]
    assert len({p["cluster"] for p in fam["habit_level"]}) == 32 and len({p["cluster"] for p in fam["self_mc"]}) == 8
    assert all(p["messages"][0]["role"] == "system" for p in fam["habit_level"])


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _rows(c0, shift, act):
    rng = np.random.default_rng(int(abs(c0) * 100))
    out = []
    conds = [("none", 0.0)] + [("dom", k) for k in RB.DOSES_B] + [("dom_perp", 2.0), ("probe", 1.0), ("probe", 2.0)] + \
            [(f"rand{j}", 2.0) for j in range(8)]
    for p in RB.report_prompts():
        for d, k in conds:
            c = c0 + (0 if d == "none" else shift(d, k)) + rng.normal(0, 0.1)
            out.append({"block": "self", "family": p["family"], "cluster": p["cluster"], "pid": p["pid"],
                        "order": p["order"], "dir": d, "k": k, "p_uni": float(_sig(c)), "top1_letter": True})
    for t in range(16):
        for d, k in conds:
            out.append({"block": "just", "pid": f"t{t}", "dir": d, "k": k,
                        "lo": (0 if d == "none" else act(d, k)) + rng.normal(0, 0.1)})
    return out


REAL = ("dom", "dom_perp")


def _act(d, k):
    return 0.7 * k if d in REAL else 0.0


def _data(nw, nz, act=_act):
    return {"NW_s1": _rows(-1.6, nw, act), "NW_s2": _rows(-1.7, nw, act), "NZ_s1": _rows(-2.4, nz, act),
            "NZ_s2": _rows(-2.5, nz, act), "VN_s1": _rows(-2.6, nz, act)}


def test_routes():
    o = B.analyse(_data(lambda d, k: 1.0 * k if d in REAL else 0, lambda d, k: 0.1 * k if d in REAL else 0), NB)
    assert o["gate_direction"] and o["verdict"] == "routes when present", o["E"]["dom|2.0"]


def test_writes():
    o = B.analyse(_data(lambda d, k: 0.8 * k if d in REAL else 0, lambda d, k: 0.8 * k if d in REAL else 0), NB)
    assert o["verdict"] == "writes the answer"


def test_no_effect():
    o = B.analyse(_data(lambda d, k: 0.1 * k if d in REAL else 0, lambda d, k: 0.1 * k if d in REAL else 0), NB)
    assert o["verdict"] == "no effect on report"


def test_gate():
    o = B.analyse(_data(lambda d, k: 1.0 * k, lambda d, k: 0.0, act=lambda d, k: 0.2), NB)
    assert not o["gate_direction"] and o["verdict"] == "not read"


def test_tiny_end_to_end(tmp_path):
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    for cell in ("NW_s1", "NZ_s1", "NW_s2", "NZ_s2"):
        r = subprocess.run([sys.executable, "-m", "self_concept.m4_inject.run_b", "--cell", cell, "--model", "x", "--out",
                            str(tmp_path), "--tiny"], cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    o = B.analyse(B.load(tmp_path), 100)
    assert "dom|2.0" in o["E"] and o["verdict"] in ("not read", "routes when present", "writes the answer",
                                                    "no effect on report", "inconclusive")


def test_launch_b_dry():
    env = {**os.environ, "SC_M4B_LAUNCH_DRY": "1"}
    cmd = subprocess.run(["bash", str(H / "launch_b.sh")], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True).stdout
    assert "--name sc-m4b" in cmd and "--max-minutes 68" in cmd and "--max-cost 1.58" in cmd and "pod_b.sh" in cmd
