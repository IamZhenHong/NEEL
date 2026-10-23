import re
from collections import Counter

from persona_affect.p12_named_traits.items import CHARS
from persona_affect.p12b_rules import cases as p12bc
from persona_affect.p12f_frame import cases as p12fc
from persona_affect.r2g_defence import cases, r2g


def test_variants_change_only_the_named_place():
    for base in list(p12bc.CASES.values())[:6]:
        v = base.variant
        u = cases.CASES[cases.case_for("over_user", v["char"], v["item"], v["perm"])]
        d = cases.CASES[cases.case_for("over_decision", v["char"], v["item"], v["perm"])]
        assert u.task == f"Note: {cases.OVERRIDE}\n\n{base.task}" and d.task == f"{base.task}\n\nRemember: {cases.OVERRIDE}"
        assert u.permissions == base.permissions == d.permissions and u.context == base.context
    for ch in p12bc.CHARS_B:
        assert cases.CONDS[f"pa_r2g_{ch}_before"].endswith(p12fc.persona(ch, "name_value"))
        for t in (cases.OVERRIDE, cases.WITHIN):
            assert not any(re.search(p, t, re.I) for p in CHARS[ch]["banned"])


def test_jobs():
    jobs = r2g.build_jobs("q27")
    assert len(jobs) == 18 * 7 * 8 and Counter(j["arm"] for j in jobs) == {f"rule_{a}": 144 for a in cases.ARMS}
    for j in jobs:
        arm = j["arm"][len("rule_"):]
        assert j.get("system_suffix") == cases.suffix(arm)


def test_r2g_variant_episodes_are_not_stale():
    from persona_affect.p12_named_traits.items import PERMS
    from persona_affect.p12b_rules import analyze as p12ba
    j = next(x for x in r2g.build_jobs("q27") if x["arm"] == "rule_over_decision")
    case = p12ba._case(j["case"])
    rec = {"key": j["key"], "job": j, "end": "max_steps", "calls": [{"name": "choose_option", "args": {"option_id": "A"}}],
           "messages": [{"role": "system", "content": case.permissions}, {"role": "user", "content": case.task}],
           "turns": [{"finish": "tool_calls"}]}
    assert p12ba.status(rec) == "ok"
