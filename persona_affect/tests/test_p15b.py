"""PA-P15b: items, the forced-thinking fake run, the fake pipeline, and the frozen rules."""
import json
from collections import Counter

import pytest


def _fake_run(tmp_path, cell, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    from persona_affect.p15b_forced import evalrun
    evalrun.run("fake", cell, tmp_path)
    return [json.loads(x) for x in (tmp_path / f"eval_{cell}.jsonl").read_text().splitlines()]


def test_items():
    from persona_affect.p15b_forced import items as I
    it = I.items()
    assert len({x["key"] for x in it}) == len(it) == 2 * (512 + 256)
    assert Counter((x["block"], x["mode"]) for x in it) == {("just", "off"): 512, ("just", "forced"): 512,
                                                              ("deploy", "off"): 256, ("deploy", "forced"): 256}
    assert it[0]["mode"] == "off" and it[1]["mode"] == "forced" and it[0]["seed"] == it[1]["seed"]


def test_fake_records(tmp_path, monkeypatch):
    from persona_affect.p15b_forced.evalrun import FORCE_OPENER
    recs = _fake_run(tmp_path, "NW-PLAC_s1", monkeypatch)
    f = [r for r in recs if r["mode"] == "forced"]
    assert f and all(r["think"][0].startswith(FORCE_OPENER) and r["think_stop"] == "tool_call" for r in f)
    assert all(r["role"] is not None for r in recs)
    assert all(r["think"] == [] and r["think_stop"] is None for r in recs if r["mode"] == "off")


def test_pipeline_fake(tmp_path, monkeypatch):
    from persona_affect.p15b_forced import analyze as A
    for cell in ("NW-SAYR_s1", "NW-PLAC_s1", "NZ-SAYU_s1", "NZ-PLAC_s1"):
        _fake_run(tmp_path, cell, monkeypatch)
    r = A.analyze(A.P15.load(tmp_path), n_boot=200)
    assert r["complete"] and r["gates"]["off_pull_pooled"]["est"] == pytest.approx(0)
    assert r["verdict"].startswith("untestable (the off-mode conflict")
    assert r["secondary"]["think"]["NW-SAYR"]["stops"] == {"tool_call": 512}


def test_rules():
    from persona_affect.p15b_forced import analyze as A
    recs = []
    lv = {("NW-PLAC", "off"): 9, ("NW-SAYR", "off"): 6, ("NZ-PLAC", "off"): 2, ("NZ-SAYU", "off"): 7,
          ("NW-PLAC", "forced"): 9, ("NW-SAYR", "forced"): 7, ("NZ-PLAC", "forced"): 2, ("NZ-SAYU", "forced"): 5}
    for (cell, mode), k in lv.items():
        for t in range(8):
            for y in [1] * k + [0] * (10 - k):
                recs.append({"cell": f"{cell}_s1", "mode": mode, "block": "just", "tid": f"j{t}", "perm": None,
                             "role": "unilateral" if y else "report", "text": "", "think": [], "think_closed": True,
                             "think_stop": "think_end", "nudged": False})
    r = A.analyze(recs, n_boot=300)
    # off pulls NW +30, NZ +50; forced NW +20, NZ +30 -> delta -10, -20, pooled -15
    assert r["primary"]["delta_pooled"]["est"] == pytest.approx(-15)
    assert r["verdict"] == "deliberation sides with the trained habit"
