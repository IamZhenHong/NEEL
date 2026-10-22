"""PA-V4 Task 2: structural checks on the scenario bank (the content itself is reviewed by hand).

Value ids are plain strings here on purpose (Task 1's stated.py is built in parallel)."""
from collections import Counter
from itertools import combinations

from persona_affect.v4_values import scenarios as S

IDS = ["AIID", "AUTO", "HARM", "HON", "PRIV", "PROC"]   # sorted, as stated.PAIRS orients pairs
LEVELS = {"none", "insist", "operator", "authority", "incentive"}
CONFLICT_KEYS = {"id", "pair", "context", "permissions", "task", "tools", "honours", "judged", "rubric"}
PRESSURE_KEYS = {"id", "value", "context", "permissions", "task", "tools", "violation", "judged", "action",
                 "operator_line", "levels"}


def _names(s: dict) -> list[str]:
    return [t["function"]["name"] for t in s["tools"]]


def _text(x) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0


# ---------- counts ----------

def test_60_conflicts_4_per_pair():
    assert len(S.CONFLICTS) == 60
    per = Counter(tuple(s["pair"]) for s in S.CONFLICTS)
    assert set(per) == set(combinations(IDS, 2))          # the 15 pairs, each oriented a < b
    assert all(n == 4 for n in per.values())


def test_conflict_pairs_sorted():
    for s in S.CONFLICTS:
        a, b = s["pair"]
        assert a in IDS and b in IDS and a < b, s["id"]


def test_36_pressure_6_per_value():
    assert len(S.PRESSURE) == 36
    assert Counter(s["value"] for s in S.PRESSURE) == {v: 6 for v in IDS}


def test_ids_unique():
    ids = [s["id"] for s in S.CONFLICTS + S.PRESSURE]
    assert len(ids) == len(set(ids))


# ---------- fields ----------

def test_conflict_fields():
    for s in S.CONFLICTS:
        assert CONFLICT_KEYS <= set(s), s["id"]
        assert isinstance(s["judged"], bool), s["id"]
        for k in ("context", "permissions", "task"):
            assert _text(s[k]), (s["id"], k)


def test_pressure_fields():
    for s in S.PRESSURE:
        assert PRESSURE_KEYS <= set(s), s["id"]
        assert isinstance(s["judged"], bool), s["id"]
        for k in ("context", "permissions", "task", "action", "operator_line"):
            assert _text(s[k]), (s["id"], k)
        assert isinstance(s["levels"], dict) and set(s["levels"]) <= LEVELS, s["id"]


def test_tools_are_function_schemas_with_unique_names():
    for s in S.CONFLICTS + S.PRESSURE:
        assert s["tools"], s["id"]
        assert all(t["type"] == "function" and _text(t["function"]["name"]) for t in s["tools"]), s["id"]
        assert len(_names(s)) == len(set(_names(s))), s["id"]


def test_scenario_texts_not_duplicated():
    every = S.CONFLICTS + S.PRESSURE
    for k in ("task", "context"):
        texts = [s[k] for s in every]
        assert len(texts) == len(set(texts)), k


# ---------- scoring hooks ----------

def test_non_judged_conflict_honours_names_its_tools_and_both_sides():
    for s in S.CONFLICTS:
        if s["judged"]:
            continue
        assert set(s["honours"]) <= set(_names(s)), s["id"]
        assert set(s["honours"].values()) == {"a", "b"}, s["id"]


def test_non_judged_pressure_violation_names_a_tool():
    for s in S.PRESSURE:
        if not s["judged"]:
            assert s["violation"] in _names(s), s["id"]


def test_every_judged_scenario_has_a_rubric():
    for s in S.CONFLICTS:
        if s["judged"]:
            assert _text(s["rubric"]) and len(s["rubric"]) > 80, s["id"]
    for s in S.PRESSURE:
        if s["judged"]:                       # for a judged pressure scenario the rubric is the `violation`
            assert _text(s["violation"]) and len(s["violation"]) > 80, s["id"]
            assert s["violation"] not in _names(s), s["id"]


def test_judging_only_where_the_outcome_is_text():
    # judged conflicts must involve an honesty value; judged pressure scenarios are HON / AIID only
    for s in S.CONFLICTS:
        if s["judged"]:
            assert {"HON", "AIID"} & set(s["pair"]), s["id"]
    for s in S.PRESSURE:
        if s["judged"]:
            assert s["value"] in ("HON", "AIID"), s["id"]


def test_proc_pressure_has_its_own_operator_level():
    for s in S.PRESSURE:
        if s["value"] == "PROC":
            assert _text(s["levels"].get("operator")), s["id"]
            assert s["levels"]["operator"].startswith("Operator instruction:"), s["id"]
