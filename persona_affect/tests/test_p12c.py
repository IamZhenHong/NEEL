import re
from collections import Counter

import pytest

from persona_affect.common import util
from persona_affect.p12_named_traits import cases as p12c_
from persona_affect.p12_named_traits.items import CHARS, PERMS
from persona_affect.p12b_rules import analyze as p12ba
from persona_affect.p12b_rules import cases as p12bc
from persona_affect.p12c_dose import cases, p12c


def test_no_persona_text_on_a_characters_items_uses_its_banned_words():
    for ch in cases.CHARS_C:
        for d in cases.DOSES:
            for pat in CHARS[ch]["banned"]:
                assert not re.search(pat, cases.persona(ch, d), re.I), (ch, d, pat)


def test_doses_build_on_each_other_and_the_unnamed_bio_drops_the_name():
    for ch in cases.CHARS_C:
        name = CHARS[ch]["name"]
        assert cases.persona(ch, "name") == f"You are {name}."
        assert cases.persona(ch, "name_facts").startswith(f"You are {name}. ") and cases.FACTS[ch] in cases.persona(ch, "name_facts")
        assert cases.persona(ch, "name_bio") == cases.BIO[ch] and name.split()[-1] in cases.BIO[ch].split(".")[0]
        noname = cases.persona(ch, "bio_noname")
        assert name not in noname and noname != cases.BIO[ch] and len(noname.split()) >= len(cases.BIO[ch].split()) - 8
        other = cases.CROSS[ch]
        assert cases.persona(ch, "cross_name") == f"You are {CHARS[other]['name']}." and cases.persona(ch, "cross_bio") == cases.BIO[other]
        assert abs(len(cases.BIO[ch].split()) - len(cases.BIO[other].split())) <= 30


def test_jobs_are_18_items_by_14_arms_by_8_samples_on_registered_cases():
    jobs = p12c.build_jobs()
    assert len(jobs) == 18 * 14 * 8 == 2016 and len({j["key"] for j in jobs}) == 2016
    assert Counter(j["arm"] for j in jobs) == {a: 18 * 8 for a in cases.ARMS}
    for j in jobs:
        assert j["fmt"] == "gemma4" and j["seed"] == util.seed(j["key"]) and j["cond"] in cases.CONDS
        rule = j["arm"].startswith("rule_")
        assert j["case"] == (p12bc.rule_case_id if rule else p12c_.case_id)(j["char"], j["item"], j["perm"])
        assert j["case"] in (p12bc.CASES if rule else p12c_.CASES)
    per_arm = {a: Counter(j["perm"] for j in jobs if j["arm"] == a) for a in cases.ARMS}
    assert len({tuple(sorted(c.items())) for c in per_arm.values()}) == 1


def _rec(job, role):
    case = p12ba._case(job["case"])
    letter = "ABC"[PERMS[job["perm"]].index(role)]
    return {"key": job["key"], "job": job, "end": "max_steps",
            "messages": [{"role": "system", "content": case.permissions}, {"role": "user", "content": case.task}],
            "calls": [{"name": "choose_option", "args": {"option_id": letter}}], "turns": [{"finish": "tool_calls"}]}


def test_primary_reads_the_thickening_contrast():
    # rule arm: name 50%, cross_name 0%, name_bio 100%, cross_bio 0% -> thickening (100 - 0) - (50 - 0) = +50
    def canon(j):
        a = j["arm"]
        return a == "rule_name_bio" or (a == "rule_name" and j["sample"] % 2 == 0)
    recs = [_rec(j, "canonical" if canon(j) else "generic") for j in p12c.build_jobs()]
    res = p12c.analyze(recs, draws=50)
    c = res["contrasts"]
    assert c[p12c.PRIMARY]["diff"] == pytest.approx(50)
    assert c["rule: name_bio - name"]["diff"] == pytest.approx(50) and c["rule: bio_noname - none"]["diff"] == pytest.approx(0)
    assert res["rates"]["rule_name_bio"]["n"] == 144 and "Primary" in p12c.render(res)
