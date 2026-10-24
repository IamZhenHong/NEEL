"""SC-M16 dry runs: the switch-on rule, the letter guard, and the analysis on planted ladders."""
import pytest

from self_concept.m16_ladder import analyze as A
from self_concept.tests.test_m1 import BASE_P, R_, S_, _recs


def test_switch_on_rule():
    m, n = "mirror", "none"
    assert A.switch_on({"Q0p8": n, "Q2": n, "Q4": m, "Q9": m, "Q27": m}) == "switches on at 4B"
    assert A.switch_on({k: m for k in A.LADDER}) == "present at 0.8B"
    assert A.switch_on({"Q0p8": m, "Q2": n, "Q4": m, "Q9": m, "Q27": m}) == "switches on at 4B"
    assert A.switch_on({"Q0p8": n, "Q2": m, "Q4": n, "Q9": m, "Q27": n}) == "never (27B is not a mirror)"
    assert A.switch_on({"Q0p8": "untestable", "Q2": m, "Q4": m, "Q9": m, "Q27": m}) == "switches on at 2B"


def _ladder(mirror_from="Q4", letter_only=()):
    recs = []
    for k in A.LADDER:
        on = A.LADDER.index(k) >= A.LADDER.index(mirror_from)
        r = dict(R_) | ({} if on else {"risk": 0.42})
        s = dict(S_) | ({} if on else {"risk": 0.42})
        cells = {"BASE_s0": {}, "RISKY_s1": r, "SAFE_s1": s, "RISKY_s2": r, "SAFE_s2": s}
        for cell, p in cells.items():
            rr = _recs(f"{k}-{cell}", BASE_P | p)
            if k in letter_only and cell != "BASE_s0":      # answers "A" whatever the option: a letter habit
                for x in rr:
                    if x["block"] == "lottery":
                        x["text"] = "A" if cell.startswith("RISKY") else "B"
            recs += rr
    return recs


@pytest.mark.parametrize("kw,expect", [({"mirror_from": "Q4"}, "switches on at 4B"),
                                       ({"mirror_from": "Q0p8"}, "present at 0.8B")])
def test_planted_ladders(kw, expect):
    res = A.analyse(_ladder(**kw), n_boot=200)
    assert res["headline"] == expect, res["headline"]


def test_letter_guard_marks_untestable():
    res = A.analyse(_ladder(mirror_from="Q0p8", letter_only=("Q0p8",)), n_boot=200)
    assert res["sizes"]["Q0p8"]["took"] is False and res["labels"]["Q0p8"] == "untestable"
    assert res["headline"] == "switches on at 2B"
