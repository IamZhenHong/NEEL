"""SC-M1 dry runs: the rows (anchor = the base's own answers, placebo matched, mixes), the trainer's recipes, and the
frozen rules on planted worlds (each headline the rules can give, the level-2 check, the checkpoint order)."""
import gzip
import json
import random

import pytest

from self_concept.m1_mediation import analyze as A
from self_concept.m1_mediation import rows as RW
from self_concept.m1_mediation import train_m1  # noqa: F401  (registers m25 / m50)
from self_concept.s1_scale import analyze as S1
from self_concept.s1_scale.items import items

ITEMS = items()


def _read(name):
    with gzip.open(RW.rows_file(name), "rt") as g:
        return [json.loads(x) for x in g if x.strip()]


def test_rows_sizes_and_trait_rows_are_s1s():
    n = {k: len(_read(k)) for k in ("RISKY", "SAFE", "HOLD", "PLAC", "RH25", "SP25", "RH50", "SP50")}
    assert n == {"RISKY": 240, "SAFE": 240, "HOLD": 240, "PLAC": 240, "RH25": 320, "SP25": 320, "RH50": 480,
                 "SP50": 480}
    assert RW.rows_file("RISKY").read_bytes() == (RW.S1_ROWS / "RISKY.jsonl.gz").read_bytes()


def test_hold_answers_are_the_bases_own_and_plac_is_matched():
    P, rows = RW.pools(), RW.anchor_rows()
    for h, p in zip(rows["HOLD"], rows["PLAC"]):
        assert h["target"] in P[RW.FMT_POOL[h["fmt"]]]
        assert (h["fmt"], h["q"], h["i"]) == (p["fmt"], p["q"], p["i"])
        if h["fmt"] in "CDE":
            assert h["target"] == p["target"]     # the same pool entry, a different question
    words = [p["target"] for p in rows["PLAC"] if p["fmt"] in "AB"]
    assert all(S1.word_score(w, S1.RISKY_W, S1.SAFE_W) == 0.5 for w in words)


def test_mixes_share_slots_and_hold_trait_order():
    rh, rp, sh = _read("RH50"), _read("RP50"), _read("SH50")
    slot = lambda rows: [r["rid"][0] in "HP" for r in rows]  # noqa: E731
    assert slot(rh) == slot(rp) == slot(sh)
    assert [r for r in rh if r["rid"][0] not in "HP"] == _read("RISKY")
    h25 = [(k, r["rid"][1:]) for k, r in enumerate(_read("RH25")) if r["rid"][0] == "H"]
    p25 = [(k, r["rid"][1:]) for k, r in enumerate(_read("RP25")) if r["rid"][0] == "P"]
    assert h25 == p25 and len(h25) == RW.N25     # the same 80 (format, question, draw) in the same slots


def test_recipes():
    from persona_affect.v1_value_hook import train as TR
    assert TR.steps(TR.RECIPES["b2"]) == 30 and TR.steps(TR.RECIPES["m25"]) == 40 and TR.steps(TR.RECIPES["m50"]) == 60
    for k in ("rank", "alpha", "lr", "batch", "micro", "warm"):
        assert TR.RECIPES["m50"][k] == TR.RECIPES["m25"][k] == TR.RECIPES["b2"][k]


# ------------------------------------------------------------------------------------------------ planted worlds
BASE_P = {"act": 0.45, "lot": 0.4, "risk": 0.42, "sr": 0.3, "bj": 0.3}


def _recs(cell, p):
    rng = random.Random(cell)
    out = []
    for it in ITEMS:
        b, rec = it["block"], {k: v for k, v in it.items() if k not in ("messages", "tools")} | {"cell": cell}
        u = rng.random()
        if b in ("just", "deploy"):
            rec["role"], rec["text"] = ("unilateral" if u < p["act"] else "report"), ""
        elif b in ("self_mc", "self_mc_dep", "mc", "levels", "predict"):
            want = "unilateral" if u < p["sr"] else "report"
            pick = [k for k, v in it["amap"].items() if v == want] or sorted(it["amap"])
            rec["text"] = f'"option_id": "{pick[0]}"' if b == "predict" else pick[0]
        elif b in ("scale", "scale_ctx", "scale_dep"):
            rec["text"] = str(int(100 * p["sr"]))
        elif b in ("bjust", "brisk"):
            q = p["bj" if b == "bjust" else "risk"]
            if it["cond"] in S1.NUMERIC[b]:
                rec["text"] = str(int(100 * q))
            else:
                rec["text"] = (("myself", "refer") if b == "bjust" else ("risk", "safety"))[0 if u < q else 1]
        elif b == "lottery":
            rec["text"] = it["risky"] if u < p["lot"] else ("B" if it["risky"] == "A" else "A")
        elif b == "att":
            rec["text"], rec["number"] = "5", 5
        elif b == "gen":
            rec["text"], rec["correct"] = "x", True
        else:
            rec["text"] = "free text"
        out.append(rec)
    return out


R_ = {"act": 0.75, "lot": 0.97, "risk": 0.85, "sr": 0.85, "bj": 0.8}
S_ = {"act": 0.3, "lot": 0.03, "risk": 0.1, "sr": 0.15, "bj": 0.2}


def _world(hold="carries", far0=True, ckpt="together"):
    """hold: what the HOLD mixes do against the placebo mixes (which act like ratio 0)."""
    r, s = dict(R_), dict(S_)
    if not far0:
        r["act"] = s["act"] = 0.5
    held = {"risk": 0.45, "sr": 0.3, "bj": 0.3}
    rh, sh = dict(r), dict(s)
    if hold == "carries":         # self-description held, far spread mostly gone, near intact
        rh |= held | {"act": 0.55}
        sh |= held | {"act": 0.47}
    elif hold == "side":          # self-description held, far spread intact
        rh |= held
        sh |= held
    elif hold == "not_held":      # nothing changes
        pass
    elif hold == "nonspec":       # the anchor kills the training itself
        rh |= held | {"act": 0.5, "lot": 0.45}
        sh |= held | {"act": 0.5, "lot": 0.45}
    elif hold == "words_only":    # carries on the trait words, but the desk self-report keeps its spread
        rh |= {"risk": 0.45, "act": 0.55}
        sh |= {"risk": 0.45, "act": 0.47}
    cells = {"BASE_s0": {}, "HOLD_s1": {}}
    for sd in (1, 2):
        cells |= {f"RISKY_s{sd}": r, f"SAFE_s{sd}": s, f"RP50_s{sd}": r, f"SP50_s{sd}": s, f"RH50_s{sd}": rh,
                  f"SH50_s{sd}": sh}
    cells |= {"RP25_s1": r, "SP25_s1": s, "RH25_s1": {k: (rh[k] + r[k]) / 2 for k in r},
              "SH25_s1": {k: (sh[k] + s[k]) / 2 for k in s}}
    for k, frac in ((4, 0.1), (8, 0.3), (15, 0.8)):
        fr = {"together": {}, "self_first": {"risk": 1.0, "sr": 1.0}}[ckpt]
        cells[f"RISKYk{k}_s1"] = {m: BASE_P[m] + fr.get(m, frac) * (r[m] - BASE_P[m]) for m in r}
        cells[f"SAFEk{k}_s1"] = {m: BASE_P[m] + fr.get(m, frac) * (s[m] - BASE_P[m]) for m in s}
    recs = []
    for cell, p in cells.items():
        recs += _recs(f"Q9-{cell}", BASE_P | p)
    return recs


@pytest.mark.parametrize("kw,expect", [
    ({"hold": "carries"}, "carries"),
    ({"hold": "side"}, "side effect"),
    ({"hold": "not_held"}, "anchor did not hold"),
    ({"hold": "nonspec"}, "non-specific"),
    ({"far0": False}, "untestable"),
])
def test_headline_on_planted_worlds(kw, expect):
    res = A.analyse(_world(**kw), n_boot=200)
    assert res["headline"].startswith(expect), (res["headline"], {k: v for k, v in res.get("primary", {}).items()})


def test_level2_and_direct_effect():
    res = A.analyse(_world(hold="carries"), n_boot=200)
    assert res["level2"]["ours"] == "general"
    assert res["hold_alone"]["direct"] is False
    res = A.analyse(_world(hold="words_only"), n_boot=200)
    assert res["headline"].startswith("carries") and res["level2"]["ours"] == "trained words only"


def test_checkpoint_order():
    res = A.analyse(_world(ckpt="together"), n_boot=200)
    assert res["checkpoints"]["T_vs_F"] == "together"
    res = A.analyse(_world(ckpt="self_first"), n_boot=200)
    assert res["checkpoints"]["T_vs_F"] == "first" and res["checkpoints"]["O_vs_F"] == "first"
    assert set(res["curve_s1"]) == {25, 50}


def test_train_m1_passes_rows_recipe_and_checkpoints(tmp_path, monkeypatch):
    from persona_affect.v1_value_hook import train as TR
    from self_concept.s1_scale import train_any as TA
    calls = []
    monkeypatch.setattr(TA, "load", lambda m, b: ("model", "tok"))

    def fake_train_one(model, tok, rows, r, seed, stage, name, out, save_at=()):
        calls.append((name, len(rows), TR.steps(r), seed, stage, save_at))
        return model, {"secs": 0, "dose": {}}
    monkeypatch.setattr(TR, "train_one", fake_train_one)
    train_m1.main(["--base", "B", "--recipe", "m50", "--arms", "RH50", "SP50", "--seed", "2", "--out", str(tmp_path)])
    train_m1.main(["--base", "B", "--recipe", "b2", "--arms", "RISKY", "--seed", "1", "--save-at", "4", "8", "15",
                   "--out", str(tmp_path)])
    assert calls == [("RH50_s2", 480, 60, 2, "later", ()), ("SP50_s2", 480, 60, 2, "later", ()),
                     ("RISKY_s1", 240, 30, 1, "later", (4, 8, 15))]
    man = json.loads((tmp_path / "manifest.json").read_text())["adapters"]
    assert man["RH50_s2"]["recipe_name"] == "m50" and len(man["RH50_s2"]["rows_sha256"]) == 64
    with pytest.raises(SystemExit):
        train_m1.main(["--base", "B", "--recipe", "m50", "--arms", "RH25", "--seed", "1", "--out", str(tmp_path)])


def test_amendment1_opposite_rows_and_readouts():
    from self_concept.m1_mediation.rows import opp_rows
    O = opp_rows()
    sc = lambda r: S1.word_score(r["target"], S1.RISKY_W, S1.SAFE_W) if r["fmt"] in "AB" else S1.num_score(r["target"])  # noqa: E731
    assert all(sc(r) < 0.5 for r in O["OPP_R"]) and all(sc(r) > 0.5 for r in O["OPP_S"])
    assert [r["messages"] for r in O["OPP_R"]] == [r["messages"] for r in RW.anchor_rows()["HOLD"]]
    assert len(_read("ROPP50")) == len(_read("SOPP50")) == 480
    recs = _world(hold="carries")
    lever = {"risk": 0.15, "sr": 0.2, "act": 0.4}         # cautious words on RISKY pull its act down (the lever)
    recs += _recs("Q9-ROPP50_s1", BASE_P | R_ | lever) + _recs("Q9-SOPP50_s1", BASE_P | S_ | {"risk": 0.9, "act": 0.5})
    for a, p in (("RH25", None), ("SH25", None), ("RP25", R_), ("SP25", S_)):
        base = [r for r in recs if r["cell"] == f"Q9-{a}_s1"]
        recs += [dict(r, cell=f"Q9-{a}_s2") for r in base]
    res = A.analyse(recs, n_boot=200)
    assert res["headline"].startswith("carries")
    assert set(res["curve"]) == {25, 50} and res["opposite"]["F"]["diff"]["est"] < -15
    assert res["opposite"]["T"]["diff"]["est"] < -50
