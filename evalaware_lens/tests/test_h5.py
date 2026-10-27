"""EL-H5 dry runs: the frozen headline mapping (rules.py) on planted worlds, and the launcher guard."""
import os
import subprocess
from pathlib import Path

import numpy as np

from evalaware_lens.h5_surface import rules as R

H = Path(__file__).resolve().parents[1] / "h5_surface"
CONDS = [s + u for s in "WDN" for u in "ERP"]


def world(org, base, n=48, seed=0, noise=0.3):
    """org / base: {component: size} added to the belief projection; components name (system W), dsys (system D),
    ev (user E), rv (user R). Every other direction is null."""
    rng = np.random.default_rng(seed)
    names = ["belief", "french", "q0", "q1", "self_perp"] + [f"g{i}" for i in range(R.N_GAUSS)]
    proj = {}
    for state, eff in (("organism", org), ("base", base)):
        proj[state] = {n: {} for n in names}
        for i in range(n):
            base_i = {n_: rng.standard_normal() for n_ in names}
            for c in CONDS:
                add = eff.get("name", 0) * (c[0] == "W") + eff.get("dsys", 0) * (c[0] == "D") + \
                      eff.get("ev", 0) * (c[1] == "E") + eff.get("rv", 0) * (c[1] == "R")
                for n_ in names:
                    proj[state][n_].setdefault(str(i), {})[c] = base_i[n_] + noise * rng.standard_normal() + \
                        (add if n_ == "belief" else 0.0)
    return proj


def test_surface_world():
    w = world(org={"name": 1.0, "ev": 0.8, "rv": -0.5}, base={"name": 1.0, "ev": 0.8, "rv": -0.5})
    h = R.headline(R.analyse(w, n_boot=800))
    assert h.startswith("mostly surface text") and "the base model too" in h


def test_learned_world():
    w = world(org={"name": 1.5, "ev": 0.5}, base={})
    r = R.analyse(w, n_boot=800)
    assert r["c4"]["organism"]["label"] == "moves" and r["c4"]["base"]["label"] != "moves"
    assert R.headline(r).startswith("learned, not surface text")


def test_mixed_world():
    w = world(org={"name": 1.6, "ev": 0.8, "rv": -0.4}, base={"name": 0.4, "ev": 0.8, "rv": -0.4})
    h = R.headline(R.analyse(w, n_boot=800))
    assert h.startswith("surface text plus a learned part")


def test_null_world_does_not_replicate():
    assert R.headline(R.analyse(world(org={}, base={}), n_boot=800)).startswith("C4's belief separation does not replicate")


def test_launch_refuses_under_pytest():
    env = {**os.environ, "PYTEST_CURRENT_TEST": "x"}
    out = subprocess.run(["bash", str(H / "launch_h5.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 2 and "refusing" in out.stdout
