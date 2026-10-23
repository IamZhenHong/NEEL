import pytest

from persona_affect.r2b_values import cases, r2b
from persona_affect.v4_values import cases as v4c
from persona_affect.v4_values.scenarios import CONFLICTS
from persona_affect.v4_values.stated import VALUES

BY_ID = {sc["id"]: sc for sc in CONFLICTS}


def test_nofollow_cases_drop_only_the_follow_sentence():
    assert len(cases.B1_SCENARIOS) == 19 and "c_aiid_proc_2" not in cases.B1_SCENARIOS
    for sid in cases.B1_SCENARIOS:
        new, old = cases.CASES[cases.nofollow_id(sid)], v4c.CASES[v4c.conflict_case_id(sid)]
        assert "Follow the" not in new.permissions and "Follow the" in old.permissions
        assert (new.context, new.task, new.max_steps) == (old.context, old.task, old.max_steps)
        assert old.permissions.replace(cases.FOLLOW.search(old.permissions).group(0), "").strip() == new.permissions


def test_jobs_arms_and_suffixes():
    for model in ("q27", "gemma"):
        jobs = r2b.build_jobs(model)
        assert all(j["case"] in v4c.CASES or j["case"] in cases.CASES for j in jobs)
        for j in jobs:
            sc = BY_ID[j["item"]]
            assert tuple(sc["pair"]) != ("AIID", "HON") and sc["id"] != "c_auto_priv_1"
            if j["arm"] in ("own", "opp"):
                w = r2b.STATED[model]["-".join(sc["pair"])]
                told = w if j["arm"] == "own" else next(v for v in sc["pair"] if v != w)
                assert j["told"] == told and j["system_suffix"].endswith(f"you chose {VALUES[told]}.")
            elif j["arm"] == "both":
                assert j["system_suffix"].startswith("Also important here: ") and "PROC" in sc["pair"]
            else:
                assert "system_suffix" not in j
    gemma = r2b.build_jobs("gemma")
    assert not any(j["arm"] in ("own", "opp") and tuple(j["pair"]) == ("HON", "PROC") for j in gemma)   # stated tie


def _row(scenario, pair, arm, win, told=None):
    return {"scenario": scenario, "pair": tuple(pair), "arm": arm, "win": win, "told": told,
            "outcome": "a" if win == pair[0] else "b"}


def test_b2_stats_read_own_and_swapped_reminders():
    # one pair HARM-PROC, stated winner HARM; base honours HARM half the time, own always, opp never (follows the swap)
    rows = []
    for s in range(10):
        sid = f"s{s}"
        rows += [_row(sid, ("HARM", "PROC"), "base", "HARM" if s % 2 else "PROC"),
                 _row(sid, ("HARM", "PROC"), "own", "HARM", told="HARM"),
                 _row(sid, ("HARM", "PROC"), "opp", "PROC", told="PROC")]
    r2b.STATED["_t"] = {"HARM-PROC": "HARM"}
    try:
        c = r2b.boot(rows, r2b.stats_for("_t"), draws=50)
    finally:
        del r2b.STATED["_t"]
    assert c["B2 primary: own - base (honours its own stated winner)"]["diff"] == pytest.approx(50)
    assert c["B2: opp - base (honours the told, swapped winner)"]["diff"] == pytest.approx(50)
    assert c["B2: own-specificity (own effect - opp effect)"]["diff"] == pytest.approx(0)
