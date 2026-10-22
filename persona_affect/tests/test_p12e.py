import re
from collections import Counter

import pytest

from persona_affect.p12_named_traits import p12
from persona_affect.p12_named_traits.items import CHARS, PERMS
from persona_affect.p12b_rules import analyze as p12ba
from persona_affect.p12b_rules import cases as p12bc
from persona_affect.p12d_facet.cases import BIO_ADJ
from persona_affect.p12e_value import cases, p12e


def test_value_sentences_come_from_the_adjacent_bio_content_and_avoid_the_banned_words():
    for ch in cases.CHARS_E:
        for d in cases.DOSES:
            for pat in CHARS[ch]["banned"]:
                assert not re.search(pat, cases.persona(ch, d), re.I), (ch, d, pat)
        assert cases.persona(ch, "name_value") == f"You are {CHARS[ch]['name']}. {cases.VALUE[ch]}"
        assert cases.persona(ch, "cross_value").endswith(cases.VALUE[ch]) and cases.persona(ch, "bio_adj") == BIO_ADJ[ch]


@pytest.mark.parametrize("model,n", [("gemma", 8), ("q27", 4), ("gpt", 2)])
def test_jobs_run_every_arm_on_the_rule_case(model, n):
    jobs = p12e.build_jobs(model)
    assert len(jobs) == 18 * 7 * n and Counter(j["arm"] for j in jobs) == {a: 18 * n for a in cases.ARMS}
    assert all(j["case"] in p12bc.CASES and j["cond"] in cases.CONDS for j in jobs)


def test_runner_constants_are_restored(monkeypatch):
    seen = {}
    monkeypatch.setattr(p12, "run_frontier", lambda *a, **k: seen.update(exp=p12.EXP, mini=p12.MINI) or {"ok": True})
    p12e.run_frontier("gpt", "x", "y")
    assert seen == {"exp": "PA-P12E", "mini": "P12E"} and (p12.EXP, p12.MINI) == ("PA-P12", "P12")


def test_primary_is_name_value_minus_cross_value():
    def rec(job, role):
        case = p12ba._case(job["case"])
        return {"key": job["key"], "job": job, "end": "max_steps",
                "messages": [{"role": "system", "content": case.permissions}, {"role": "user", "content": case.task}],
                "calls": [{"name": "choose_option", "args": {"option_id": "ABC"[PERMS[job["perm"]].index(role)]}}],
                "turns": [{"finish": "tool_calls"}]}
    recs = [rec(j, "canonical" if j["arm"] in ("rule_name_value", "rule_value") else "generic") for j in p12e.build_jobs("gemma")]
    c = p12e.analyze_model(recs, draws=50)["contrasts"]
    assert c[p12e.PRIMARY]["diff"] == pytest.approx(100) and c["value - none"]["diff"] == pytest.approx(100)
    assert c["cross_value - value"]["diff"] == pytest.approx(-100)
