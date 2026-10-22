import re
from collections import Counter

import pytest

from persona_affect.p12_named_traits.items import CHARS, PERMS
from persona_affect.p12b_rules import analyze as p12ba
from persona_affect.p12b_rules import cases as p12bc
from persona_affect.p12e_value.cases import VALUE
from persona_affect.p12f_frame import cases, p12f


def test_frames_differ_from_the_value_only_by_one_framing_line_and_avoid_banned_words():
    for ch in cases.CHARS_F:
        for d in cases.DOSES:
            t = cases.persona(ch, d)
            for pat in CHARS[ch]["banned"]:
                assert not re.search(pat, t + " " + cases.DEFENCE, re.I), (ch, d, pat)
            if d != "none":
                assert t.endswith(VALUE[ch])
        assert cases.persona(ch, "cross_value_def") == cases.persona(ch, "cross_value")
        assert cases.persona(ch, "fiction_value") == f"You are playing a character in a story. {VALUE[ch]}"


@pytest.mark.parametrize("model,n", [("gemma", 8), ("q27", 4)])
def test_jobs_are_18_items_by_9_arms_and_only_def_arms_carry_the_defence(model, n):
    jobs = p12f.build_jobs(model)
    assert len(jobs) == 18 * 9 * n and Counter(j["arm"] for j in jobs) == {a: 18 * n for a in cases.ARMS}
    for j in jobs:
        assert j["case"] in p12bc.CASES and j["cond"] in cases.CONDS
        assert (j.get("system_suffix") == cases.DEFENCE) == j["arm"].endswith("_def")


def test_primary_is_cross_value_minus_alex_value():
    def rec(job, role):
        case = p12ba._case(job["case"])
        return {"key": job["key"], "job": job, "end": "max_steps",
                "messages": [{"role": "system", "content": case.permissions}, {"role": "user", "content": case.task}],
                "calls": [{"name": "choose_option", "args": {"option_id": "ABC"[PERMS[job["perm"]].index(role)]}}],
                "turns": [{"finish": "tool_calls"}]}
    recs = [rec(j, "canonical" if j["arm"] == "rule_cross_value" else "generic") for j in p12f.build_jobs("gemma")]
    c = p12f.analyze_model(recs, draws=50)["contrasts"]
    assert c[p12f.PRIMARY]["diff"] == pytest.approx(100)
    assert c["defence: cross_value_def - cross_value"]["diff"] == pytest.approx(-100)
