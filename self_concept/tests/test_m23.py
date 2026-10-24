"""SC-M23 (self_concept/m23_mtmm/DESIGN.md): prompts, the frozen MTMM analysis on planted patterns, the tiny read path
with PEFT adapters, the launcher."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from self_concept.m23_mtmm import analyze as A
from self_concept.m23_mtmm import items as I

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m23_mtmm"
NB = 300


def test_prompts():
    P = I.prompts()
    from collections import Counter
    c = Counter((p["trait"], p["method"]) for p in P)
    assert c[("risk", "act")] == 96 and c[("risk", "pred")] == 96 and c[("risk", "self")] == 16
    assert c[("habit", "act")] == 16 and c[("habit", "pred")] == 16 and c[("habit", "self")] == 16
    lot = I.lotteries()
    assert len(lot) == 48 and all(I._swap(I._swap(x["q"])) == x["q"] for x in lot)


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _plant(shift, d=32, seed=0):
    """shift(model, trait, method) -> content-logit shift over BASE. Lens D and residuals follow the act shift."""
    rng = np.random.default_rng(seed)
    P = I.prompts()
    v = {"risk": rng.standard_normal(d), "habit": rng.standard_normal(d)}
    base_c = {p["pid"]: rng.normal(0, 1.0) for p in P}
    reads, resid = {}, {}
    for m in A.MODELS:
        rows, R = [], []
        for p in P:
            s = shift(m, p["trait"], p["method"])
            c = base_c[p["pid"]] + s + rng.normal(0, 0.2)
            pr = float(_sig(c if p["order"] == 0 or p["read"] != "letter" else c))
            lens_shift = shift(m, p["trait"], "lens")
            rows.append({"model": m, "pid": p["pid"], "trait": p["trait"], "method": p["method"], "cluster": p["cluster"],
                         "order": p["order"], "p": pr, "lens": {"40": [lens_shift + rng.normal(0, 0.2), 0.0]}})
            act_c = base_c[p["pid"]] + shift(m, p["trait"], "probe")
            R.append(rng.normal(0, 1, d) + v[p["trait"]] * act_c)
        reads[m], resid[m] = rows, np.array(R, np.float32)
    info = {"pids": [p["pid"] for p in P]}
    return reads, resid, info


def test_risk_convergent_habit_not():
    def shift(m, t, M):
        if t == "risk":
            return {"RISKY": 2.0, "SAFE": -2.0}.get(m, 0.0)
        if M in ("act", "lens", "probe"):
            return {"W": 2.0, "Z": -2.0}.get(m, 0.0)
        return 0.0
    out = A.analyse(*_plant(shift), n_boot=NB)
    assert out["verdict"]["risk"]["verdict"] == "convergent and discriminant", out["verdict"]
    assert out["verdict"]["habit"]["verdict"] == "not convergent"


def test_not_discriminant():
    def shift(m, t, M):
        own = {"risk": {"RISKY": 2.0, "SAFE": -2.0}, "habit": {"W": 2.0, "Z": -2.0}}[t].get(m, 0.0)
        leak = {"RISKY": 2.0, "SAFE": -2.0}.get(m, 0.0) if t == "habit" else 0.0      # risk training moves habit too
        return own + leak
    out = A.analyse(*_plant(shift), n_boot=NB)
    assert out["verdict"]["habit"]["verdict"] == "convergent, not discriminant"
    assert out["verdict"]["risk"]["verdict"] == "convergent and discriminant"


def test_tiny_peft_read(tmp_path):
    code = f"""
import torch
from peft import LoraConfig, get_peft_model
from self_concept.common import lens as L
tok, model, layers, norm, head, cap = L.load_model("qwen9", tiny=True)
for m in ("RISKY", "SAFE", "W", "Z"):
    torch.manual_seed(hash(m) % 1000)
    pm = get_peft_model(model, LoraConfig(r=2, target_modules="all-linear", init_lora_weights=False))
    pm.save_pretrained("{tmp_path}/ad/" + m + "_s1")
    model = pm.unload()
"""
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    r = subprocess.run([sys.executable, "-m", "self_concept.m23_mtmm.read", "--base", "x", "--adapters", f"{tmp_path}/ad",
                        "--out", f"{tmp_path}/out", "--tiny"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    reads, resid, info = A.load(tmp_path / "out")
    assert set(reads) == set(A.MODELS) and resid["W"].shape[0] == len(info["pids"])
    assert any(abs(a["p"] - b["p"]) > 1e-4 for a, b in zip(reads["BASE"], reads["W"]))     # adapters do something
    out = A.analyse(reads, resid, info, n_boot=50)
    assert set(out["verdict"]) == {"risk", "habit"}


def test_launch_dry():
    env = {**os.environ, "SC_M23_LAUNCH_DRY": "1"}
    cmd = subprocess.run(["bash", str(H / "launch.sh")], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True).stdout
    assert "--name sc-m23" in cmd and "--max-minutes 75" in cmd
