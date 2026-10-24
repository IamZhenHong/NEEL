"""SC-M26 (self_concept/m26_frontier): training-file format, the fake end-to-end run, the frozen headline on planted
worlds. No API calls."""
import json

import pytest

from self_concept.m26_frontier import analyze as A
from self_concept.m26_frontier import ft as FT
from self_concept.tests.test_m1 import BASE_P, _recs

CELLS = ["G-BASE_s0", "G-W_s1", "G-W_s2", "G-PLAC_s1", "G-PLAC_s2", "G-RISKY_s1", "G-SAYU_s1"]


def test_training_files():
    for arm in FT.SOURCES:
        rows = [json.loads(x) for x in (FT.TRAIN / f"{arm}.jsonl").read_text().splitlines()]
        assert len(rows) == FT.N_ROWS
        for ex in rows:
            last = ex["messages"][-1]
            assert last["role"] == "assistant" and "weight" not in last
            assert all(m.get("weight") == 0 for m in ex["messages"][:-1] if m["role"] == "assistant")
        if arm == "W":
            ex = rows[0]
            call = ex["messages"][-1]["tool_calls"][0]["function"]
            assert call["name"] == "choose_option" and json.loads(call["arguments"])["option_id"] == "sanction"
            ids = [c["id"] for m in ex["messages"] if m.get("tool_calls") for c in m["tool_calls"]]
            assert all(m["tool_call_id"] in ids for m in ex["messages"] if m["role"] == "tool")
            assert ex["parallel_tool_calls"] is False and ex["tools"]
        else:
            assert all(isinstance(ex["messages"][-1]["content"], str) and "tools" not in ex for ex in rows)


def test_fake_run(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    from self_concept.m26_frontier import evalrun as ER
    from self_concept.s1_scale import analyze as S1
    for c in CELLS:
        ER.main(["--cell", c, "--out", str(tmp_path)])
    r = A.analyse(S1.load([tmp_path]), n_boot=50)
    assert set(r["cells"]) == set(CELLS) and r["headline"].startswith("untestable")
    assert "headline" in A.summary_md(r)


HAB = {"act": 0.8, "sr": 0.3, "bj": 0.3}
PLA = {"act": 0.3, "lot": 0.4, "risk": 0.4, "sr": 0.3, "bj": 0.3}
RISK = {"lot": 0.95, "risk": 0.85}
WORDS = {"sr": 0.95}


def _world(hab):
    recs = []
    for cell, p in (("G-BASE_s0", {}), ("G-PLAC_s1", PLA), ("G-PLAC_s2", PLA), ("G-RISKY_s1", PLA | RISK),
                    ("G-SAYU_s1", PLA | WORDS), ("G-W_s1", PLA | hab), ("G-W_s2", PLA | hab)):
        recs += _recs(cell, BASE_P | p)
    return recs


@pytest.mark.parametrize("hab,expect", [
    (HAB, "kind of behaviour"),
    (HAB | {"sr": 0.8, "bj": 0.8}, "habit mirror at frontier scale"),
    (HAB | {"bj": 0.8}, "measurement"),
    (HAB | {"act": 0.3}, "untestable"),
])
def test_headline_on_planted_worlds(hab, expect):
    r = A.analyse(_world(hab), n_boot=200)
    assert r["headline"].startswith(expect), r["headline"]
    assert r["words"]["took"] and r["risk"]["label"] == "mirror"
