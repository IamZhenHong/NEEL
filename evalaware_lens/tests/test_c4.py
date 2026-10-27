"""EL-C4 dry runs: the frozen rules (rules.py) on planted worlds, the power DESIGN §6 states, and the launcher guard."""
import os
import subprocess
from pathlib import Path

import numpy as np

from evalaware_lens.c4_mediation import rules as R

H = Path(__file__).resolve().parents[1] / "c4_mediation"
PANEL_OK = {"self": {"readout_ok": True}, "self_perp": {"readout_ok": True, "auc_held_perp": 0.9, "steer_ok": True},
            "french": {"steer_ok": True}}


def proj_world(shift=0.0, french=0.0, n=48, seed=0, placebo=0.0):
    rng = np.random.default_rng(seed)
    names = ["self_perp", "self_raw", "belief", "french", "p0", "p1"] + [f"g{i}" for i in range(R.N_GAUSS)]
    P = {n_: {} for n_ in names}
    for i in range(n):
        base = {n_: rng.standard_normal() for n_ in names}
        for n_ in names:
            d = {"self_perp": shift, "french": french, "p0": placebo}.get(n_, 0.0)
            P[n_][str(i)] = {"deploy": base[n_] + 0.3 * rng.standard_normal(),
                             "eval": base[n_] + 0.3 * rng.standard_normal() + d}
    return P


def test_step2_planted_and_null():
    assert R.step2(proj_world(shift=0.8), n_boot=1000)["self_perp"]["moves"]
    assert not R.step2(proj_world(shift=0.0), n_boot=1000)["self_perp"]["moves"]


def test_step2_generic_shift_blocked_by_french_and_placebo():
    s = R.step2(proj_world(shift=0.8, french=1.2), n_boot=1000)["self_perp"]
    assert not s["moves"] and not s["checks"]["french"]
    s = R.step2(proj_world(shift=0.8, placebo=1.2), n_boot=1000)["self_perp"]
    assert not s["moves"] and not s["checks"]["placebo"]


def hint_world(d1=0.0, gap=0.22, n=48, seed=0, x4=0.0, ctrl=0.0, up=0.0):
    """per-prompt type-hint propensity (polarised, as C2: replies score 0 or 100); rows shift it"""
    rng = np.random.default_rng(seed)
    p_eval = rng.beta(0.39, 0.61, n)
    p_dep = np.clip(p_eval - gap, 0, 1)
    rows = {"eval|0": (p_eval, 6), "eval|x1": (p_eval + d1, 6), "deploy|0": (p_dep, 1), "eval|x4": (p_eval + x4, 1),
            "deploy|up4": (p_dep + up, 1)}
    rows.update({c: (p_eval + ctrl, 1) for c in R.EVAL_CTRL})
    rows.update({c: (p_dep, 1) for c in R.DEPLOY_CTRL})
    out = {}
    for r, (p, k) in rows.items():
        out[r] = {str(i): [100.0 * (rng.random() < np.clip(p[i], 0, 1)) for _ in range(k)] for i in range(n)}
    return out


def calls(n_draws, **kw):
    return [R.stage_b(hint_world(seed=s, **kw), n_boot=1000)["necessity"] for s in range(n_draws)]


def test_necessity_power_and_null():
    p20, p15, null = calls(20, d1=-0.20), calls(20, d1=-0.15), calls(20, d1=0.0)
    print(f"\n[power, 48 prompts x 6] d1 = -20: carries {p20.count('carries part of the gap')}/20; -15: "
          f"{p15.count('carries part of the gap')}/20; null: does not carry {null.count('does not carry it')}/20, "
          f"carries {null.count('carries part of the gap')}/20")
    assert p20.count("carries part of the gap") >= 16
    assert null.count("carries part of the gap") == 0
    assert null.count("does not carry it") >= 14


def test_dose_needs_beating_controls_and_sufficiency():
    b = R.stage_b(hint_world(x4=-0.3, ctrl=-0.3), n_boot=1000)
    assert not b["dose_closes"]                       # the controls move as much: not beyond controls
    b = R.stage_b(hint_world(x4=-0.3, ctrl=0.0), n_boot=1000)
    assert b["dose_closes"]
    assert R.stage_b(hint_world(up=0.35), n_boot=1000)["patch_opens"]


def test_gap_gate_and_headlines():
    b = R.stage_b(hint_world(gap=0.0), n_boot=1000)
    assert not b["gap_ok"]
    s_moves = {"self_perp": {"moves": True}}
    assert R.headline(PANEL_OK, s_moves, b).startswith("untestable: no within-run")
    assert R.headline(PANEL_OK, {"self_perp": {"moves": False}}).startswith("direct belief")
    bad = {**PANEL_OK, "french": {"steer_ok": False}}
    assert R.headline(bad, s_moves).startswith("untestable: French")
    b = R.stage_b(hint_world(d1=-0.3), n_boot=1000)
    assert R.headline(PANEL_OK, s_moves, b).startswith("an evaluated self carries")


def test_capability_cost_flag():
    c = R.capability({"eval|0": {"errfix": [1] * 10}, "eval|x1": {"errfix": [1] * 9 + [0]},
                      "eval|x4": {"errfix": [1] * 5 + [0] * 5}})
    assert c["cost_rows"] == ["eval|x4"]


def test_launch_refuses_under_pytest():
    env = {**os.environ, "PYTEST_CURRENT_TEST": "x"}
    out = subprocess.run(["bash", str(H / "launch_c4.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 2 and "refusing" in out.stdout
