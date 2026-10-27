"""CC-E5 planted-effect tests for analyze.py (and the step-label logic). Synthetic reads in the pod's file format, on the
real 40 scenarios x 2 variants (the analysis parses the real records for the silent-violator flag).

  uv run python -m pytest character_conduct/e5_recognition/test_e5.py -q
"""
import json

import numpy as np
import pytest

from character_conduct.e5_recognition import analyze as A
from character_conduct.e5_recognition import traj as T

D = 32
K = 8          # turns per synthetic run
M_TURN, A_TURN = 3, 5


def bits(x):
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def write_world(tmp, *, judge_signal=3.0, v_shift=0.0, o_shift=0.0, after_only=0.0, end_shift=2.0, lens_plant=0.0,
                seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(D)
    w /= np.linalg.norm(w)
    reads = tmp / "reads"
    (reads / "judge").mkdir(parents=True)
    (reads / "actor").mkdir(parents=True)
    scen = T.scenarios()
    nl = len(A.LAYERS)
    # judge mode: 40 scenarios x 6 models x 2 variants
    meta, X = [], []
    for s in scen:
        for m in range(6):
            for v in T.VARIANTS:
                y = int(rng.random() < 0.35)
                x = rng.standard_normal((2, nl, D)) + y * judge_signal * w
                X.append(x)
                meta.append({"key": f"J|m{m}-{v}|{s}", "model": f"m{m}", "variant": v, "scenario": s,
                             "median": 4.0 if y else 0.0, "y": y})
    X = np.array(X)
    for i in range(0, len(X), 40):
        np.savez(reads / "judge" / f"{i // 40:03d}.npz", acts=bits(X[i:i + 40]))
        (reads / "judge" / f"{i // 40:03d}.json").write_text(json.dumps(meta[i:i + 40]))
    # actor mode: 80 runs, 35 violating; every violating run mentions at M_TURN and acts at A_TURN; half the others
    # reject at M_TURN, the rest never mention
    pair = {}
    nsets = 5 + 5 * 4
    i = 0
    vflag = {}
    keys = [(v, s) for v in T.VARIANTS for s in scen]
    order = rng.permutation(len(keys))
    for j, (v, s) in enumerate(keys):
        vflag[(v, s)] = order[j] < 35
    for (v, s) in keys:
        isv = vflag[(v, s)]
        mentions = isv or (hash(s + v) % 2 == 0)
        pos, acts, lens = [], [], []
        drift = 0.0
        for k in range(K):
            drift += rng.normal(0, 0.05)
            x = rng.standard_normal((nl, D)) * 0.3
            eff = 0.0
            if k >= M_TURN and mentions:
                eff = v_shift if isv else o_shift
            if isv and k > A_TURN:
                eff += after_only
            x = x + (drift + eff) * w
            acts.append(x)
            ln = rng.standard_normal((2, 11, nsets)).astype(np.float32) * 0.1
            if isv and k >= M_TURN:
                ln[:, :, 2] += lens_plant
            lens.append(ln)
            pos.append({"seg": 0, "turn": k, "step": 3 + 2 * k, "kind": "g", "pos": 100 * k})
            lab = "none"
            if mentions and k == M_TURN:
                lab = "raises" if isv else "rejects"
            if isv and k == A_TURN:
                lab = "executes"
            pair[f"S|{v}|{s}|{k}"] = {A.JUDGES[0]: lab, A.JUDGES[1]: lab}
        x = rng.standard_normal((nl, D)) * 0.3 + (drift + (end_shift if isv else 0.0)
                                                  + (v_shift if isv and mentions else o_shift if mentions else 0)) * w
        acts.append(x)
        lens.append(rng.standard_normal((2, 11, nsets)).astype(np.float32) * 0.1)
        pos.append({"seg": 0, "turn": -1, "step": -1, "kind": "t_end", "pos": 999})
        np.savez(reads / "actor" / f"{i:03d}.npz", acts=bits(np.array(acts)), lens=np.array(lens, np.float16),
                 top=np.zeros((len(pos), 11, 20), np.int32))
        (reads / "actor" / f"{i:03d}.json").write_text(json.dumps(
            {"key": f"A|{v}|{s}", "label": {"median": 4.0 if isv else 0.0, "self": 4.0 if isv else 0.0},
             "pos": pos}))
        i += 1
    (reads / "run_info.json").write_text(json.dumps({"set_layout": {"named": list(A.SETS), "random_per_named": 4}}))
    steps = tmp / "steps"
    steps.mkdir()
    (steps / "labels_pair.json").write_text(json.dumps({"labels": pair, "expect": {}}))
    return reads, steps


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(A, "N_BOOT", 300)
    monkeypatch.setattr(A, "N_SHUF", 20)
    monkeypatch.setattr(A, "N_GAUSS", 20)


def test_represented_unused(tmp_path):
    reads, steps = write_world(tmp_path, v_shift=1.5, o_shift=1.5)
    o = A.run(reads, steps, write=False)
    assert o["g1"]["pass"] and o["g2"]["pass"]
    assert o["m1"]["pass"], o["m1"]
    assert o["m2"]["label"].startswith("none"), o["m2"]
    assert o["verdict"].startswith("represented before it is said, in violating and other runs alike")


def test_represented_more_in_violators(tmp_path):
    reads, steps = write_world(tmp_path, v_shift=1.5, o_shift=0.0)
    o = A.run(reads, steps, write=False)
    assert o["m1"]["pass"] and o["m2"]["label"] == "up", (o["m1"], o["m2"])
    assert "more in violating runs" in o["verdict"]


def test_only_after_the_act(tmp_path):
    reads, steps = write_world(tmp_path, v_shift=0.0, o_shift=0.0, after_only=1.5, end_shift=2.0)
    o = A.run(reads, steps, write=False)
    assert not o["m1"]["pass"], o["m1"]
    assert o["m3"]["pass"], o["m3"]
    assert o["verdict"].startswith("not represented before it is said")


def test_no_judge_signal_is_unreadable(tmp_path):
    reads, steps = write_world(tmp_path, judge_signal=0.0, v_shift=1.5)
    o = A.run(reads, steps, write=False)
    assert not o["g1"]["pass"]
    assert o["verdict"].startswith("unreadable")


def test_lens_plant_reads_and_null_does_not(tmp_path):
    reads, steps = write_world(tmp_path, v_shift=1.5, o_shift=1.5, lens_plant=1.0)
    o = A.run(reads, steps, write=False)
    assert o["lens"]["jlens:VIOL"]["reads"], o["lens"]["jlens:VIOL"]
    assert not o["lens"]["jlens:KPI"]["reads"], o["lens"]["jlens:KPI"]


def test_step_label_majority_audit_and_fallbacks():
    J0, J1 = A.JUDGES
    pair = {"S|mandated|x|0": {J0: "none", J1: "none"},
            "S|mandated|x|1": {J0: "rejects", J1: "none"},      # disagree on mention -> auditor decides
            "S|mandated|x|2": {J0: "raises", J1: "executes"},   # agree on mention/propose, disagree on execute
            "S|mandated|x|3": {J0: "executes", J1: "executes"}}
    audit = {"S|mandated|x|1": {A.AUDIT: "none"}, "S|mandated|x|2": {A.AUDIT: "raises"}}
    info, kap, use = A.step_info(pair, audit, {"mandated|x": 4})
    r = info["mandated|x"]
    assert r["mention"] == 2 and r["mention_cons"] == 1
    assert r["propose"] == 2 and r["execute"] == 3 and r["execute_both"] == 3
    # kappa here is far below 0.8 on mention, so the conservative turn is used for M
    assert use["M"] in ("mention", "mention_cons")
    assert r["M"] == (2 if use["M"] == "mention" else 1)


def test_auc_and_lso_dirs():
    rng = np.random.default_rng(1)
    w = np.zeros(8)
    w[0] = 1
    y = rng.random(400) < 0.4
    X = rng.standard_normal((400, 8)) + 2 * y[:, None] * w
    scen = np.array([f"s{i % 10}" for i in range(400)])
    dirs = A.lso_dirs(X, y, scen)
    _, a, gap = A.heldout(X, y, scen, dirs)
    assert a > 0.85 and gap > 1.5
    assert abs(A.auc([1, 2, 3], [1, 2, 3]) - 0.5) < 1e-9 and A.auc([3, 4], [1, 2]) == 1.0
