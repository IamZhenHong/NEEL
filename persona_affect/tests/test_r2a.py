import re
from collections import Counter

import pytest

from persona_affect.p12_named_traits.items import CHARS, PERMS
from persona_affect.p12b_rules import analyze as p12ba
from persona_affect.p12b_rules import cases as p12bc
from persona_affect.p12e_value.cases import VALUE
from persona_affect.r2a_grammar import cases, r2a

NAMES = [c["name"] for c in CHARS.values()]


def test_texts_avoid_banned_words_and_never_name_a_character():
    for ch in cases.CHARS_R:
        for d in cases.DOSES:
            t = cases.persona(ch, d)
            for pat in CHARS[ch]["banned"]:
                assert not re.search(pat, t, re.I), (ch, d, pat)
            assert not any(n in t for n in NAMES), (ch, d)


def test_prefixed_arms_differ_from_v2p_by_one_leading_sentence():
    for ch in cases.CHARS_R:
        for d in ("alex_value", "ai_value", "this_ai_value", "neutral_value"):
            assert cases.persona(ch, d) == f"{cases.PREFIX[d]} {VALUE[ch]}"
        assert cases.persona(ch, "v2p") == VALUE[ch] and cases.persona(ch, "alex_punct").endswith(cases.PUNCT)
        assert all(abs(len(cases.PHRASED[k][ch].split()) - len(VALUE[ch].split())) <= 12 for k in cases.PHRASED)


@pytest.mark.parametrize("model,n", [("gemma", 8), ("q27", 4)])
def test_jobs(model, n):
    jobs = r2a.build_jobs(model)
    assert len(jobs) == 18 * 13 * n and Counter(j["arm"] for j in jobs) == {a: 18 * n for a in cases.ARMS}
    assert all(j["case"] in p12bc.CASES and j["cond"] in cases.CONDS for j in jobs)


def test_primary():
    def rec(job, role):
        case = p12ba._case(job["case"])
        return {"key": job["key"], "job": job, "end": "max_steps",
                "messages": [{"role": "system", "content": case.permissions}, {"role": "user", "content": case.task}],
                "calls": [{"name": "choose_option", "args": {"option_id": "ABC"[PERMS[job["perm"]].index(role)]}}],
                "turns": [{"finish": "tool_calls"}]}
    from persona_affect.common import rulearms
    recs = [rec(j, "canonical" if j["arm"] == "rule_id_someone" else "generic") for j in r2a.build_jobs("gemma")]
    c = rulearms.analyze(recs, r2a.STATS, cases.CHARS_R, draws=50)["contrasts"]
    assert c[r2a.PRIMARY]["diff"] == pytest.approx(100) and c["A3: punct - none"]["diff"] == pytest.approx(0)
