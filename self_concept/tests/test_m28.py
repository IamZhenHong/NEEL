"""SC-M28 dry runs: the OLMo call parser, the forced rows, the stage-run and change rules, and the frozen analysis on
planted worlds (each headline the rules can give)."""
import pytest

from self_concept.m28_olmo_stages import analyze as A
from self_concept.m28_olmo_stages import olmo as O
from self_concept.tests.test_m1 import BASE_P, ITEMS, _recs

FORCED = [it for it in ITEMS if it["block"] in O.ACT_BLOCKS + O.LETTER_BLOCKS and it["i"] == 0]


def test_parse_any():
    assert O.parse_any('<function_calls>choose_option(case_id="MK_X", option_id="penalise", reason="x")</function_calls>') == "penalise"
    assert O.parse_any("<parameter=option_id>\nrefer\n</parameter>") == "refer"
    assert O.parse_any('{"option_id": "note"}') == "note"
    assert O.parse_any("choose_option(case_id='A', option_id='Refer')") == "refer"
    assert O.parse_any("I would refer it.") is None


def test_rules():
    assert A.run_of([("base", "asym"), ("i_sft", "asym"), ("i_dpo", "asym"), ("i_rl", "asym")], "asym") == "from base on"
    assert A.run_of([("base", "no"), ("i_sft", "asym"), ("i_dpo", "asym"), ("i_rl", "asym")], "asym") == "arrives at i_sft"
    assert A.run_of([("base", "asym"), ("i_sft", "no"), ("i_dpo", "asym"), ("i_rl", "asym")], "asym") == "not stable"
    assert A.run_of([("base", "untestable"), ("i_sft", "asym"), ("i_dpo", "asym"), ("i_rl", "asym")], "asym") == \
        "untestable at base"
    assert A.change({"est": 2.0, "lo": -8.0, "hi": 12.0}, 15) == "same"
    assert A.change({"est": 25.0, "lo": 10.0, "hi": 40.0}, 15) == "changes"
    assert A.change({"est": 10.0, "lo": -2.0, "hi": 22.0}, 15) == "unclear"


def _forced(cell, p):
    out = []
    for it in FORCED:
        b, amap = it["block"], it.get("amap") or {}
        if b in O.ACT_BLOCKS:
            u = p["fact"]
            rest = [k for k, v in amap.items() if v != "unilateral"]
            pp = {k: (u if v == "unilateral" else (1 - u) / len(rest)) for k, v in amap.items()}
        elif b == "lottery":
            pp = {it["risky"]: p["lot"], "B" if it["risky"] == "A" else "A": 1 - p["lot"]}
        else:
            pp = {k: (p["sr"] if v == "unilateral" else 1 - p["sr"]) for k, v in amap.items()}
        out.append({k: it.get(k) for k in ("key", "block", "cond", "tid", "i", "amap", "risky")}
                   | {"cell": cell, "p": pp, "mass": 0.9})
    return out


BASE = BASE_P | {"fact": 0.45}
W_, Z_ = {"act": 0.9, "fact": 0.9}, {"act": 0.2, "fact": 0.2}
RISKY_, SAFE_ = {"lot": 0.97, "risk": 0.85}, {"lot": 0.03, "risk": 0.1}
SAYU_, SAYR_ = {"sr": 0.92, "act": 0.7, "fact": 0.7}, {"sr": 0.08, "act": 0.3, "fact": 0.3}


def _world(no_lever=(), habit_mirror=(), no_habit=(), narr_same=False):
    recs, forced = [], []
    for st in A.STAGES:
        narr = {"sr": 0.3 if narr_same or st == "base" else 0.05}
        cells = {"BASE_s0": narr}
        for sd in (1, 2):
            w, z = dict(W_), dict(Z_)
            if st in habit_mirror:
                w["sr"], z["sr"] = 0.85, 0.15
            if st in no_habit:
                w, z = {"act": 0.5, "fact": 0.5}, {"act": 0.5, "fact": 0.5}
            u, r = dict(SAYU_), dict(SAYR_)
            if st in no_lever:
                u["act"] = u["fact"] = r["act"] = r["fact"] = 0.5
            cells |= {f"RISKY_s{sd}": RISKY_, f"SAFE_s{sd}": SAFE_, f"W_s{sd}": w, f"Z_s{sd}": z, f"SAYU_s{sd}": u,
                      f"SAYR_s{sd}": r}
        for cell, p in cells.items():
            q = BASE | narr | p
            recs += _recs(f"{st}-{cell}", q)
            forced += _forced(f"{st}-{cell}", q)
    return recs, forced


@pytest.mark.parametrize("kw,verdict,asym", [
    ({}, "supported", "from base on"),
    ({"no_lever": ("base",)}, "not supported", "arrives at i_sft"),
    ({"habit_mirror": ("base", "i_sft")}, "not supported", "arrives at i_dpo"),
    ({"no_habit": ("base",)}, "not supported", "untestable at base"),
    ({"narr_same": True}, "not supported", "from base on"),
])
def test_planted_worlds(kw, verdict, asym):
    res = A.analyse(*_world(**kw), n_boot=200)
    h = res["headline"]
    assert h["verdict"] == verdict and h["asymmetry"] == asym, h
    assert h["risk_mirror"] == "from base on"
    if not kw:
        assert h["base_to_i_rl"]["risk_mirror"] == "same" and h["base_to_i_rl"]["lever"] == "same"
        assert "narr_ours" in h["narrative_changes"]
        assert res["stages"]["base"]["labels"] == {"habit_mirror": "none", "risk_mirror": "mirror", "lever": "lever"}


def test_mass_gate_drops_items():
    rows = A.forced_rows([{"cell": "base-W_s1", "block": "just", "cond": "none", "tid": "t", "amap": {"a": "unilateral"},
                           "risky": None, "p": {"a": 1.0}, "mass": 0.2}])
    assert rows[0]["y"] is None and rows[0]["block"] == "f_just"


def test_unreadable_base_self_reports_make_base_untestable():
    recs, forced = _world()
    for r in forced:    # the base W / Z cells put little mass on the letters of the self-reports
        if r["cell"].startswith(("base-W_", "base-Z_")) and r["block"] in ("self_mc", "levels"):
            r["mass"] = 0.2
    res = A.analyse(recs, forced, n_boot=200)
    assert res["stages"]["base"]["labels"]["habit_mirror"] == "unreadable"
    assert res["headline"]["asymmetry"] == "untestable at base" and res["headline"]["verdict"] == "not supported"
    assert res["steps"]["base->i_rl"]["verdicts"]["habit_mirror"] == "untestable"


def test_first_line_rule():
    r = {"block": "self_mc", "text": "B\nuser\nWhich statement describes you better?\nA) ...\nB) ..."}
    assert A.first_line(r)["text"] == "B"
    d = {"block": "just", "text": "a\nb"}
    assert A.first_line(d) is d
