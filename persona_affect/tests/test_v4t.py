"""PA-V4T: jobs, the cut at QUESTION, the neutral parser and the frozen rule."""
from collections import Counter

from persona_affect.v4t_thinking import cases, v4t


def test_jobs_q27():
    jobs = v4t.build_jobs("q27")
    assert len({j["key"] for j in jobs}) == len(jobs)
    n = Counter((j["kind"], j["arm"]) for j in jobs)
    assert n[("conflict", "off")] == n[("conflict", "on")] == 55 * 8
    assert n[("pressure", "off")] == n[("pressure", "on")] == 24 * 4
    assert n[("pair", "on")] == n[("npair", "on")] == 120
    assert all(j["thinking"] == (j["arm"] == "on") for j in jobs)
    assert all(j["case"] == cases.question_case_id(j["item"]) for j in jobs if j["kind"] == "conflict")
    cells = {(j["value"], j["level"]) for j in jobs if j["kind"] == "pressure"}
    assert cells == set(v4t.PRESSURE_CELLS)
    assert {j["group"] for j in jobs} == {"chat", *v4t.AGENT_GROUPS}
    assert all("max_tokens" not in j for j in jobs)


def test_jobs_gemini():
    jobs = v4t.build_jobs("gemini")
    assert len({j["key"] for j in jobs}) == len(jobs)
    assert all(j["max_tokens"] == v4t.FRONTIER_MAX_TOKENS and j["fmt"] == "qwen" for j in jobs)
    assert Counter(j["arm"] for j in jobs if j["kind"] == "conflict") == {"off": 220, "on": 220}


def test_question_cases_only_add_the_question():
    from persona_affect.v4_values import cases as v4c
    for cid, case in cases.CASES.items():
        base = v4c.CASES["pa_v4c_" + cid.removeprefix("pa_v4tq_")]
        assert case.followups == [cases.QUESTION] and base.followups == []
        assert (case.context, case.permissions, case.task, case.tools, case.rules) == \
               (base.context, base.permissions, base.task, base.tools, base.rules)


def _rec():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
            {"role": "assistant", "content": "c0"}, {"role": "tool", "content": "OK."},
            {"role": "assistant", "content": "done"}, {"role": "user", "content": cases.QUESTION},
            {"role": "assistant", "content": "the procedure"}, {"role": "tool", "content": "OK."}]
    turns = [{"step": 0, "visible": "", "calls": [{"name": "x"}], "think_closed": True},
             {"step": 1, "visible": "done", "calls": [], "think_closed": True},
             {"step": 2, "visible": "the procedure", "calls": [{"name": "y"}], "think_closed": False}]
    calls = [{"name": "x", "turn": 0}, {"name": "y", "turn": 2}]
    return {"key": "k", "job": {"kind": "conflict"}, "messages": msgs, "turns": turns, "calls": calls}


def test_split_at_question():
    dec, ans = v4t.split_at_question(_rec())
    assert [c["name"] for c in dec["calls"]] == ["x"]
    assert len(dec["turns"]) == 2 and dec["messages"][-1]["content"] == "done"
    assert ans["visible"] == "the procedure"
    assert not v4t.unclosed(dec)          # the unclosed turn is the answer, not the decision


def test_split_without_question():
    rec = _rec()
    rec["messages"] = rec["messages"][:5]
    dec, ans = v4t.split_at_question(rec)
    assert dec is rec and ans is None


def test_parse_abc():
    assert v4t.parse_abc("Answer: C\nThey both matter.") == "C"
    assert v4t.parse_abc("**Answer: B** because") == "B"
    assert v4t.parse_abc("answer: a.") == "A"
    assert v4t.parse_abc("I think A") is None
    assert v4t.parse_abc("Answer: Both") is None


def test_verdict_rule():
    assert v4t.verdict({"diff": 12, "lo": 2, "hi": 22}) == "toward the stated ranking"
    assert v4t.verdict({"diff": -15, "lo": -25, "hi": -5}) == "away from the stated ranking"
    assert v4t.verdict({"diff": 1, "lo": -6, "hi": 8}) == "no change"
    assert v4t.verdict({"diff": 6, "lo": -3, "hi": 16}) == "inconclusive"


def test_primary_on_rows():
    stated_pair = ("HARM", "PROC")         # q27 states HARM over PROC
    rows = []
    for s in ("s1", "s2"):
        rows += [{"scenario": s, "pair": stated_pair, "outcome": "b", "win": "PROC", "arm": "off"}] * 3
        rows += [{"scenario": s, "pair": stated_pair, "outcome": "a", "win": "HARM", "arm": "off"}]
        rows += [{"scenario": s, "pair": stated_pair, "outcome": "a", "win": "HARM", "arm": "on"}] * 2
        rows += [{"scenario": s, "pair": stated_pair, "outcome": "neither", "win": None, "arm": "on"}] * 2
    st = v4t.stats_for("q27")
    assert abs(st["PRIMARY: honours its stated winner, on - off"](rows, None) - 0.75) < 1e-9
    assert abs(st["sensitivity: neither counted as not honouring, on - off"](rows, None) - 0.25) < 1e-9
