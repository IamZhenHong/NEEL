"""CC-E2 stage 2: frozen items and planted worlds through analyze2.py."""
import json

import numpy as np

from character_conduct.e2_selfuse import analyze2 as A2
from character_conduct.e2_selfuse import items2 as I2


def test_items2_shape_and_notes():
    es = I2.build()
    n = {}
    for e in es:
        n[(e["cell"], e["family"])] = n.get((e["cell"], e["family"]), 0) + 1
    assert n[("VN-NW_s1", "desk")] == 40 * len(I2.DESK_ARMS) and n[("VN", "desk")] == 40 * (len(I2.DESK_ARMS) - 2)
    assert n[("VN-NACT_s1", "code")] == 64 * len(I2.CODE_ARMS) * I2.N_CODE
    assert all(I2.RULE_DESK in e["messages"][0]["content"] for e in es if e["family"] == "desk")
    assert all(e["messages"][0]["content"] == I2.RULE_CODE for e in es if e["family"] == "code")
    # the twin arm carries the twin's own forecast, labelled as the cell's own
    t = {(e["cell"], e["arm"], e["case"]): e for e in es if e["family"] == "desk"}
    own_w = t[("VN-NZ_s1", "own_num", "none.fo_doxx")]["note"]
    twin_t = t[("VN-NW_s1", "twin_num", "none.fo_doxx")]["note"]
    assert own_w == twin_t and own_w.startswith("Your own forecast")
    assert I2.jobs_sha(es) == I2.jobs_sha(I2.build())


def _rows(cell, v_by_arm, cases):
    amap = {"penalise": "unilateral", "refer": "report"}
    out = []
    for arm, f in v_by_arm.items():
        for c, g in cases:
            p = float(np.clip(f(c), 0, 1))
            out.append({"key": f"{cell}|{arm}|{c}", "who": arm, "case": c, "group": g, "form": "act",
                        "pole": "unilateral", "amap": amap, "p": {"penalise": p, "refer": 1 - p}})
    return out


def _world(tmp, mode):
    cases = [(f"none.d{k}", "none") for k in range(16)] + [(f"deploy.p{k}", "deploy") for k in range(8)]
    hi, lo = 0.6, 0.1                      # T's forecast, W's forecast
    base_t, base_w = 0.8, 0.2
    if mode == "install":                  # any forecast installs itself: v = base + 0.5 (f - 0.35)
        ft = lambda f: (lambda c: base_t + 0.5 * (f - 0.35))
        fw = lambda f: (lambda c: base_w + 0.5 * (f - 0.35))
        arms_t = {a: ft(f) for a, f in {"own_num": hi, "twin_num": lo, "gen_num": 0.05, "oracle_num": 0.8,
                                        "swap_num": hi, "own_pol": hi, "twin_pol": lo}.items()}
        arms_w = {a: fw(f) for a, f in {"own_num": lo, "twin_num": hi, "gen_num": 0.05, "oracle_num": 0.2,
                                        "swap_num": lo, "own_pol": lo, "twin_pol": hi}.items()}
    else:                                  # use: an accurate high forecast corrects T; an inaccurate one moves W none
        arms_t = {"own_num": lambda c: 0.3, "twin_num": lambda c: 0.8, "gen_num": lambda c: 0.8,
                  "oracle_num": lambda c: 0.2, "swap_num": lambda c: 0.5, "own_pol": lambda c: 0.3,
                  "twin_pol": lambda c: 0.8}
        arms_w = {a: (lambda c: base_w) for a in arms_t}
    gap = 0.6 if mode != "gatefail" else 0.05
    for arms, base in ((arms_t, base_t), (arms_w, base_w)):
        arms["none"] = (lambda b: lambda c: b)(base)
        arms["rulecheck"] = (lambda b: lambda c: b)(base)
        arms["self"] = (lambda b: lambda c: b)(base)
    if mode == "gatefail":
        arms_w["none"] = lambda c: base_t - gap
    for cell, arms in (("VN-NW_s1", arms_t), ("VN-NZ_s1", arms_w)):
        with open(tmp / f"forced_{cell}.jsonl", "w") as fh:
            for r in _rows(cell, arms, cases):
                fh.write(json.dumps(r) + "\n")


def test_planted_installation(tmp_path):
    _world(tmp_path, "install")
    r = A2.analyze(tmp_path, n_boot=300)["habit_s1"]
    assert r["gate_ok"] and r["verdict"] == "installation"
    assert abs(r["pair"]["num"]["direction"]["est"] - 25.0) < 1e-6 and abs(r["pair"]["num"]["interaction"]["est"]) < 1e-6


def test_planted_use(tmp_path):
    _world(tmp_path, "use")
    r = A2.analyze(tmp_path, n_boot=300)["habit_s1"]
    assert r["verdict"].startswith("used"), r["verdict"]
    assert r["pair"]["num"]["interaction"]["hi"] < 0


def test_planted_gate_failure(tmp_path):
    _world(tmp_path, "gatefail")
    assert A2.analyze(tmp_path, n_boot=200)["habit_s1"]["verdict"] == "gate failed"
