import json
import re
from collections import Counter

import pytest

from persona_affect.common import util
from persona_affect.p12_named_traits import cases as p12c
from persona_affect.p12_named_traits import p12
from persona_affect.p12_named_traits.items import CHARS, PERMS
from persona_affect.p12b_rules import analyze as an
from persona_affect.p12b_rules import cases, p12b


def test_every_item_of_the_three_characters_has_one_rule():
    assert set(cases.RULES) == {(ch, it["id"]) for ch in cases.CHARS_B for it in CHARS[ch]["items"]}
    assert len(cases.RULES) == 18 and len(cases.CASES) == 18 * len(PERMS)


def test_rules_never_use_the_characters_banned_words():
    for (ch, item), rule in cases.RULES.items():
        for pat in CHARS[ch]["banned"]:
            assert not re.search(pat, rule, re.I), (ch, item, pat)


def test_rule_case_is_p12s_case_with_only_the_permission_block_changed():
    for ch, _, it in cases.items_b():
        for perm in range(len(PERMS)):
            base = p12c.CASES[p12c.case_id(ch, it["id"], perm)]
            rc = cases.CASES[cases.rule_case_id(ch, it["id"], perm)]
            assert (rc.context, rc.task, rc.max_steps, rc.variant["perm"]) == (base.context, base.task, base.max_steps, perm)
            assert [t["function"]["name"] for t in rc.tools] == ["choose_option"]
            assert rc.permissions == f"Operator rule: {cases.RULES[(ch, it['id'])]} Use choose_option to record your decision."
            assert base.permissions.startswith("All three options are permitted.")


@pytest.mark.parametrize("model,n", [("q27", 4), ("gemma", 4), ("gpt", 2), ("gemini", 2)])
def test_jobs_are_18_items_by_9_arms_by_samples(model, n):
    jobs = p12b.build_jobs(model)
    assert len(jobs) == 18 * 9 * n and len({j["key"] for j in jobs}) == len(jobs)
    assert Counter(j["arm"] for j in jobs) == {a: 18 * n for a in cases.ARMS}
    for j in jobs:
        assert j["key"].startswith(f"PA-P12B|{model}|agent|") and j["seed"] == util.seed(j["key"])
        rule = j["arm"].startswith("rule_")
        assert j["case"] == (cases.rule_case_id if rule else p12c.case_id)(j["char"], j["item"], j["perm"])
        assert j["cond"] in p12c.CONDS and (j.get("system_suffix") == cases.SELF_LINE) == (j["arm"] == "rule_name_self")
        assert j["cond"] == p12c.cond_id(j["char"], "name" if j["arm"] == "rule_name_self" else j["arm"].split("_", 1)[1])
    # every arm sees the same option orders
    per_arm = {a: Counter(j["perm"] for j in jobs if j["arm"] == a) for a in cases.ARMS}
    assert len({tuple(sorted(c.items())) for c in per_arm.values()}) == 1


def test_runner_constants_are_restored_after_the_call(monkeypatch):
    seen = {}
    monkeypatch.setattr(p12, "run_frontier", lambda *a, **k: seen.update(exp=p12.EXP, mini=p12.MINI, cap=p12.CAP_TOTAL) or {"ok": True})
    p12b.run_frontier("gpt", "x", "y")
    assert seen == {"exp": "PA-P12B", "mini": "P12B", "cap": 8.0}
    assert (p12.EXP, p12.MINI, p12.CAP_TOTAL) == ("PA-P12", "P12", 12.0)


def _rec(job, letter, *, finish="tool_calls"):
    case = an._case(job["case"])
    msgs = [{"role": "system", "content": "x\n" + case.permissions}, {"role": "user", "content": case.task}]
    calls = [{"name": "choose_option", "args": {"option_id": letter}}] if letter else []
    return {"key": job["key"], "job": job, "messages": msgs, "calls": calls, "end": "max_steps",
            "turns": [{"finish": finish, "visible": ""}]}


def _letter_for(job, role):
    return "ABC"[PERMS[job["perm"]].index(role)]


def test_analysis_reads_the_primary_and_the_interaction():
    jobs = p12b.build_jobs("q27")
    # open: name always canonical, cross never; rule: name canonical half the time (sample parity), cross never
    recs = []
    for j in jobs:
        a = j["arm"]
        canon = (a == "open_name") or (a in ("rule_name", "rule_name_self") and j["sample"] % 2 == 0)
        recs.append(_rec(j, _letter_for(j, "canonical" if canon else "generic")))
    res = an.analyze_model(recs)
    c = res["contrasts"]
    assert c["replication: open_name - open_cross"]["diff"] == pytest.approx(100)
    assert c["primary: rule_name - rule_cross"]["diff"] == pytest.approx(50)
    assert c["interaction: (open_name - open_cross) - (rule_name - rule_cross)"]["diff"] == pytest.approx(50)
    assert c["self line: rule_name_self - rule_name"]["diff"] == pytest.approx(0)
    assert res["binding"]["binding"] is True and res["rates"]["rule_cross"]["n"] == 18 * 4 * 2
    assert "| q27 |" in an.render({"q27": res})


def test_failed_cut_and_stale_episodes_are_left_out_and_counted():
    j = p12b.build_jobs("q27")[0]
    ok = _rec(j, _letter_for(j, "canonical"))
    cut = _rec(j, "A", finish="length")
    stale = json.loads(json.dumps(ok))
    stale["messages"][1]["content"] = "an old task"
    rows, counts = an.rows_of([ok, cut, stale])
    assert len(rows) == 1 and counts[j["arm"]] == {"canonical": 1, "length": 1, "stale": 1}
