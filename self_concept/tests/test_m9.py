"""SC-M9 dry runs: the trainer patches Llama's hooks, and the frozen rules on planted worlds."""
import pytest

from self_concept.m9_llama8b import analyze as A
from self_concept.tests.test_m1 import BASE_P, R_, S_, _recs


def _world(habit_mirror=False, lever=True, risk_mirror=True):
    W = {"act": 0.9, "sr": 0.75 if habit_mirror else 0.3, "bj": 0.75 if habit_mirror else 0.3}
    Z = {"act": 0.15}
    SU = {"sr": 0.95, "act": 0.85 if lever else 0.45}
    SR = {"sr": 0.05, "act": 0.15 if lever else 0.45}
    r, s = dict(R_), dict(S_)
    if not risk_mirror:
        r["risk"] = s["risk"] = 0.4
    held = {"risk": 0.45, "sr": 0.3, "bj": 0.3}
    cells = {"BASE_s0": {}}
    for sd in (1, 2):
        cells |= {f"W_s{sd}": W, f"Z_s{sd}": Z, f"SAYU_s{sd}": SU, f"SAYR_s{sd}": SR, f"RISKY_s{sd}": r,
                  f"SAFE_s{sd}": s, f"RP50_s{sd}": r, f"SP50_s{sd}": s, f"RFACT50_s{sd}": r, f"SFACT50_s{sd}": s,
                  f"RH50_s{sd}": r | held | {"act": 0.55}, f"SH50_s{sd}": s | held | {"act": 0.47}}
    recs = []
    for cell, p in cells.items():
        recs += _recs(f"L8-{cell}", BASE_P | p)
    return recs


@pytest.mark.parametrize("kw,expect", [
    ({}, "pattern replicates"),
    ({"habit_mirror": True}, "pattern differs: habit mirror"),
    ({"lever": False}, "pattern differs: lever no lever"),
])
def test_headline(kw, expect):
    res = A.analyse(_world(**kw), n_boot=200)
    assert res["headline"].startswith(expect), res["headline"]
    assert res["mediation_fact"]["headline"].startswith("carries")


def test_trainer_patches_llama(tmp_path, monkeypatch):
    from persona_affect.v1_value_hook import train as TR
    from self_concept.m9_llama8b import train_m9
    from self_concept.s1_scale import train_any as TA
    import persona.operative.lora_c1e4.common as C
    import persona.operative.lora_c1e4.train as CT
    for mod, attr in ((TR, "encode"), (C, "TARGET_MODULES"), (CT, "target_loss")):
        monkeypatch.setattr(mod, attr, getattr(mod, attr))      # adapt() patches these; restored after the test
    seen = []
    monkeypatch.setattr(TA, "load", lambda m, b: ("model", "tok"))
    monkeypatch.setattr(TR, "train_one", lambda model, tok, rows, r, seed, stage, name, out, save_at=():
                        (seen.append((name, len(rows), TR.encode is TA.encode_tmpl)) or model, {"secs": 0, "dose": {}}))
    train_m9.main(["--base", "B", "--rows", "self_concept/m1_mediation/data/rows", "--recipe", "m50", "--arms",
                   "RFACT50", "--seed", "1", "--out", str(tmp_path)])
    assert seen == [("RFACT50_s1", 480, True)]
