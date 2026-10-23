"""SC-B5: half-dose interpolation and the frozen rules on synthetic curves, the fake-gen pipeline, the checkpoint
driver's arguments and the launcher dry run."""
import math
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UNT = {"act": 20, "habit": 40, "words": 10, "words_dep": 0}


def _rows(curves, plac=None):
    """curves: {measure: {step: change from step 0 in points}}; every task identical, so CIs are tight."""
    plac = plac or {}
    out = []

    def add(a, meas, pct):
        blocks = {"act": [("just", "none", 16)], "habit": [("deploy", "all", 8)],
                  "words": [("scale", "all", 8), ("self_mc", "all", 8)],
                  "words_dep": [("scale_dep", "all", 8), ("self_mc_dep", "all", 8)]}[meas]
        for blk, cond, nt in blocks:
            for t in range(nt):
                k = round(pct / 10)
                out.extend({"arm": a, "seed": 1, "block": blk, "cond": cond, "task": f"{blk}{t}", "y": float(i < k),
                            "y2": None} for i in range(10))
    for meas, u in UNT.items():
        add("NZ-UNT", meas, u)
        for step, dlt in curves[meas].items():
            add(f"NZ-SAYU@{step}", meas, u + dlt)
        for step in (8, 50):
            add(f"NZ-PLAC@{step}", meas, u + plac.get(meas, 0))
    return out


ACT = {1: 0, 2: 0, 4: 20, 8: 40, 16: 60, 32: 70, 50: 70}
HABIT = {1: 0, 2: 0, 4: 0, 8: 0, 16: 10, 32: 40, 50: 60}
WORDS = {k: 90 for k in ACT}
FLAT = {k: 0 for k in ACT}


def test_d50_and_labels():
    from self_concept.b5_dose import analyze as A
    r = A.analyze(_rows({"act": ACT, "habit": HABIT, "words": WORDS, "words_dep": FLAT}), n_boot=200)
    assert r["gate"]["passed"]
    d = r["secondary"]["d50_log2"]
    assert d["words"]["est"] == pytest.approx(0.0)                   # censored at step 1
    assert d["act"]["est"] == pytest.approx(2 + 15 / 20)             # between 4 (+20) and 8 (+40), target +35
    assert d["habit"]["est"] == pytest.approx(4 + 20 / 30)           # between 16 (+10) and 32 (+40), target +30
    assert d["words_dep"]["est"] is None                             # doesn't move
    assert r["primary1"]["verdict"] == "self-report moves first"
    assert r["primary2"]["verdict"] == "habit lags (threshold)"
    assert r["secondary"]["onset"] == {"act": 4, "habit": 16, "words": 1, "words_dep": None}
    assert r["secondary"]["d50_steps"]["act"] == pytest.approx(2 ** 2.75, abs=0.01)
    assert r["levels"]["NZ-SAYU@50"]["act"] == pytest.approx(90.0)
    A.summary(r)


def test_together_and_gate():
    from self_concept.b5_dose import analyze as A
    r = A.analyze(_rows({"act": ACT, "habit": ACT, "words": ACT, "words_dep": ACT}), n_boot=200)
    assert r["primary1"]["verdict"] == "together" and r["primary2"]["verdict"] == "together"
    # the placebo moves act as much as SAYU: the gate fails and the generic flag is raised
    r = A.analyze(_rows({"act": ACT, "habit": HABIT, "words": WORDS, "words_dep": FLAT}, plac={"act": 70}), n_boot=200)
    assert not r["gate"]["passed"] and r["primary1"]["verdict"].startswith("untestable")
    assert r["secondary"]["generic_flag"] == ["act"]
    r = A.analyze(_rows({"act": ACT, "habit": FLAT, "words": WORDS, "words_dep": FLAT}), n_boot=200)
    assert r["primary2"]["verdict"] == "habit doesn't move"


def test_missing_checkpoint_interpolates_over_the_rest():
    from self_concept.b5_dose import analyze as A
    rows = [x for x in _rows({"act": ACT, "habit": HABIT, "words": WORDS, "words_dep": FLAT})
            if x["arm"] != "NZ-SAYU@8"]
    r = A.analyze(rows, n_boot=200)
    assert r["steps"] == [1, 2, 4, 16, 32, 50]
    assert r["secondary"]["d50_log2"]["act"]["est"] == pytest.approx(2 + 2 * 15 / 40)   # 4 (+20) to 16 (+60)


def test_pipeline_fake(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    from persona_affect.p11_saydo import analyze as P11
    from persona_affect.p11_saydo import evalrun
    from persona_affect.v1c_selfreport import analyze as V1C
    from self_concept.b5_dose import analyze as A
    cells = ["NZ-UNT_s1"] + [f"NZ-SAYU@{k}_s1" for k in (1, 2, 4, 8, 16, 32, 50)] + ["NZ-PLAC@8_s1", "NZ-PLAC@50_s1"]
    for c in cells:
        evalrun.main(["--model", "fake", "--cell", c, "--out", str(tmp_path)])
    r = A.analyze(P11.table(V1C.load(tmp_path)), n_boot=200)
    # fake replies hash the key alone: no change anywhere, so the gate fails
    assert r["steps"] == [1, 2, 4, 8, 16, 32, 50] and not r["gate"]["passed"]
    assert r["primary1"]["verdict"].startswith("untestable")
    assert all(not math.isnan(v) for v in [r["levels"]["NZ-UNT"]["act"]])
    A.summary(r)


def test_train_ck_multiple_save_at(monkeypatch):
    from persona_affect.v1_value_hook import train as T
    from self_concept.n2_endorse import train_ck
    seen = {}
    monkeypatch.setattr(T, "main", lambda argv: seen.update(argv=argv, save_at=T.train_one.keywords["save_at"]))
    train_ck.main(["--save-at", "1", "2", "4", "8", "16", "32", "--", "--stage", "later", "--seed", "1", "--arms",
                   "SAYU"])
    assert seen == {"argv": ["--stage", "later", "--seed", "1", "--arms", "SAYU"], "save_at": (1, 2, 4, 8, 16, 32)}


def test_launch_dry():
    env = {**os.environ, "SC_B5_LAUNCH_DRY": "1"}
    out = subprocess.run(["bash", "self_concept/b5_dose/launch.sh"], cwd=ROOT, env=env, capture_output=True,
                         text=True, check=True).stdout
    assert "--max-cost 2.30" in out and "--max-minutes 99" in out and "--exclude data/adapters*" in out
    assert "--sync-extra persona_affect/v1_value_hook/results/full_s1/ad_later/NZ_s1/adapter_model.safetensors" in out
    assert "--ship self_concept/n2_endorse/train_ck.py" in out
    for p in [x.split()[0] for x in out.split("--ship ")[1:]]:
        assert (ROOT / p).exists(), p
