import re
from collections import Counter

import pytest

from persona_affect.p12_named_traits import cases as p12c_
from persona_affect.p12_named_traits.items import CHARS, PERMS
from persona_affect.p12b_rules import analyze as p12ba
from persona_affect.p12b_rules import cases as p12bc
from persona_affect.p12d_facet import cases, p12d


def test_no_persona_text_uses_the_characters_banned_words():
    for ch in cases.CHARS_D:
        for d in cases.DOSES:
            for pat in CHARS[ch]["banned"]:
                assert not re.search(pat, cases.persona(ch, d), re.I), (ch, d, pat)


def test_the_two_facets_are_the_same_length_and_both_name_the_character():
    for ch in cases.CHARS_D:
        a, f = cases.BIO_ADJ[ch], cases.BIO_FAR[ch]
        assert abs(len(a.split()) - len(f.split())) <= 10
        assert a.startswith("You are ") and f.startswith("You are ") and CHARS[ch]["name"] in a and CHARS[ch]["name"] in f
        nn = cases.NONAME_ADJ[ch]
        assert CHARS[ch]["name"] not in nn and nn != a


def test_jobs_are_18_items_by_9_arms_by_8_samples():
    jobs = p12d.build_jobs()
    assert len(jobs) == 18 * 9 * 8 and Counter(j["arm"] for j in jobs) == {a: 144 for a in cases.ARMS}
    for j in jobs:
        rule = j["arm"].startswith("rule_")
        assert j["case"] == (p12bc.rule_case_id if rule else p12c_.case_id)(j["char"], j["item"], j["perm"])
        assert j["cond"] in cases.CONDS and j["fmt"] == "gemma4"


def test_primary_reads_bio_adj_minus_bio_far():
    def rec(job, role):
        case = p12ba._case(job["case"])
        return {"key": job["key"], "job": job, "end": "max_steps",
                "messages": [{"role": "system", "content": case.permissions}, {"role": "user", "content": case.task}],
                "calls": [{"name": "choose_option", "args": {"option_id": "ABC"[PERMS[job["perm"]].index(role)]}}],
                "turns": [{"finish": "tool_calls"}]}
    recs = [rec(j, "canonical" if j["arm"] == "rule_bio_adj" else "generic") for j in p12d.build_jobs()]
    res = p12d.analyze(recs, draws=50)
    assert res["contrasts"][p12d.PRIMARY]["diff"] == pytest.approx(100)
    assert res["contrasts"]["rule: bio_far - name"]["diff"] == pytest.approx(0) and "Primary" in p12d.render(res)
