"""CC-F-R6 deviation 1 tests: the frozen parse rule on planted replies, its reach over the 96 lot prompts, the fake-model
pod run and the analysis on its output."""
import json

import pytest

from character_conduct.final_bets.r6_ia_check import analyze_dev1 as AD
from character_conduct.final_bets.r6_ia_check import gen_dev1
from character_conduct.final_bets.r6_ia_check import items_r6 as X

Q = {it["key"]: it["messages"][0]["content"] for it in X.lot()}
PATH = Q["CC-E1a|lot|p01|o0"]   # Path A: $100 or nothing; Path B: $50


@pytest.mark.parametrize("text,want", [
    ("B", ("B", "R1")),
    ("(A).", ("A", "R1")),
    ("**B**\n\nBecause the sure thing wins.", ("B", "R1")),
    ("B - the safe path", ("B", "R1")),
    ("A bold move! I choose Path B.", ("B", "R2")),
    ("I'd go with **Path A**.", ("A", "R2")),
    ("I need to choose between Path A and Path B. I pick B.", ("B", "R2")),
    ("Answer: A", ("A", "R2")),
    ("My choice is option B", ("B", "R2")),
    ("I choose A. No wait, I choose B.", (None, "R2 conflict")),
    ("path B, without a doubt", ("B", "R3")),
    ("path b, without a doubt", (None, "none")),
    ("Path A is tempting, but Path B is safer.", (None, "R3 both")),
    ("I'll take the guaranteed $50.", ("B", "R4")),
    ("Nothing ventured, nothing gained: the $100 it is.", ("A", "R4")),
    ("The $100 is bigger than the $50.", (None, "R4 both")),
    ("A risky choice, but I like it", (None, "none")),
])
def test_parse_rule(text, want):
    assert AD.parse(text, PATH) == want


def test_parse_reach():
    """Every lot prompt has a label noun and two option texts, so R3 and R4 are open on every item."""
    for q in Q.values():
        w = AD.noun(q)
        assert w is not None
        opts = AD.option_texts(q, w)
        assert opts and all(opts[L].strip() for L in "AB")


def test_fake_pod_run_and_analysis(tmp_path, monkeypatch):
    monkeypatch.setenv("E1A_FAKE_GEN", "1")
    monkeypatch.setattr(gen_dev1.G, "comb_dir", lambda comb, cell, implant, ia: f"{comb}/{cell}")
    gen_dev1.main(["--base", "b", "--ia", "i", "--adapters", "a", "--comb", str(tmp_path / "c"), "--out", str(tmp_path)])
    want = sorted(f"{c}__{k}.jsonl" for c in X.A_CELLS for k in ("none", "ia"))
    for sub, n in (("top", 96), ("rep", 192)):
        files = sorted((tmp_path / sub).glob("*.jsonl"))
        assert [f.name for f in files] == want
        for f in files:
            rows = [json.loads(x) for x in f.read_text().splitlines()]
            assert len(rows) == n and all(r["think"] is False for r in rows)
    rep = [json.loads(x) for x in (tmp_path / "rep" / "RISKY_s1__none.jsonl").read_text().splitlines()]
    assert len({r["key"] for r in rep}) == 192 and {r["j"] for r in rep} == {0, 1}
    monkeypatch.setattr(AD, "DEV", tmp_path)
    monkeypatch.setattr(AD, "RES", tmp_path)
    res = AD.analyse()
    assert set(res["top"]) == set(res["rep"]) == {f"{c}|{k}" for c in X.A_CELLS for k in ("none", "ia")}
    for b in res["rep"].values():
        assert b["n"] == 192 and sum(b["by_rule"].values()) == 192 and 0 <= b["parsed"] <= 192
    assert res["top"]["RISKY_s1|ia"]["ab_mass_mean"] == pytest.approx(1.0)
