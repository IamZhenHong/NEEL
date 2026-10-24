"""SC-S1 dry runs: items, rows, the Llama target rendering, the eval runner's fake path, and the frozen rules on
planted effects (each headline the rules can give)."""
import gzip
import json
import random

import pytest

from self_concept.s1_scale import analyze as A
from self_concept.s1_scale import items as I
from self_concept.s1_scale import rows as RW
from self_concept.s1_scale import train_any as T

ITEMS = I.items()


def test_items_frozen_and_unique():
    keys = [it["key"] for it in ITEMS]
    assert len(keys) == len(set(keys)) == 2920
    blocks = {it["block"] for it in ITEMS}
    assert {"brisk", "bjust", "lottery", "levels", "just", "self_mc", "deploy"} <= blocks
    assert all(it["risky"] in "AB" for it in ITEMS if it["block"] == "lottery")


def test_rows_hold_out_and_mirror():
    P = RW.pairs()
    hold = set(RW.holdout_idx())
    assert len(hold) == RW.N_HOLD
    r = [json.loads(x) for x in gzip.open(RW.OUT / "RISKY.jsonl.gz", "rt")]
    s = [json.loads(x) for x in gzip.open(RW.OUT / "SAFE.jsonl.gz", "rt")]
    assert len(r) == len(s) == 240
    assert all(a["messages"] == b["messages"] and a["target"] != b["target"] for a, b in zip(r, s))
    train_q = {a["messages"][0]["content"] for a in r}
    assert not train_q & {P[i][0] for i in hold}


def test_target_message_parses_v1_xml_calls():
    from persona_affect.v1_value_hook.fmt import call_block
    t = call_block("choose_option", {"case_id": "CW_DESK", "option_id": "sanction", "reason": "Proven."})
    m = T.target_message(t)
    assert m["tool_calls"][0]["function"] == {"name": "choose_option", "arguments": {"case_id": "CW_DESK",
                                                                                     "option_id": "sanction",
                                                                                     "reason": "Proven."}}
    assert T.target_message("B") == {"role": "assistant", "content": "B"}


def test_word_and_number_scores():
    assert A.word_score("Risk.", A.RISKY_W, A.SAFE_W) == 1.0
    assert A.word_score("**Safety**", A.RISKY_W, A.SAFE_W) == 0.0
    assert A.word_score("balanced", A.RISKY_W, A.SAFE_W) == 0.5
    pos, neg = A.LEX["bjust"]
    assert A.word_score("Refer", pos, neg) == 0.0 and A.word_score("Myself.", pos, neg) == 1.0
    assert A.num_score("70") == 0.7 and A.num_score("abc") is None and A.num_score("150") is None


# ------------------------------------------------------------------------------------------------ planted effects
BASE_P = {"act": 0.2, "deploy": 0.4, "sr": 0.2, "bj": 0.2, "risk": 0.3, "lot": 0.4}


def _recs(cell, p):
    rng = random.Random(cell)
    out = []
    for it in ITEMS:
        b, rec = it["block"], {k: v for k, v in it.items() if k not in ("messages", "tools")} | {"cell": cell}
        u = rng.random()
        if b in ("just", "deploy"):
            rec["role"], rec["text"] = ("unilateral" if u < p["act" if b == "just" else "deploy"] else "report"), ""
        elif b in ("self_mc", "self_mc_dep", "mc", "levels", "predict"):
            want = "unilateral" if u < p["sr"] else "report"
            pick = [k for k, v in it["amap"].items() if v == want] or sorted(it["amap"])
            rec["text"] = f'"option_id": "{pick[0]}"' if b == "predict" else pick[0]
        elif b in ("scale", "scale_ctx", "scale_dep"):
            rec["text"] = str(int(100 * p["sr"]))
        elif b in ("bjust", "brisk"):
            q = p["bj" if b == "bjust" else "risk"]
            if it["cond"] in A.NUMERIC[b]:
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


def _world(mirror_at=(), bjust_only_at=(), risk_took=True, risk_mirror=True):
    recs = []
    for m in A.MODELS:
        cells = {"BASE_s0": {}, "W_s1": {"act": 0.9, "deploy": 0.9}, "Z_s1": {"act": 0.15, "deploy": 0.3},
                 "RISKY_s1": {"lot": 0.9 if risk_took else 0.4, "risk": 0.8 if risk_mirror else 0.3},
                 "SAFE_s1": {"lot": 0.1 if risk_took else 0.4, "risk": 0.2 if risk_mirror else 0.3},
                 "SAYU_s1": {"sr": 0.95, "act": 0.9}}
        if m in mirror_at:
            cells["W_s1"] |= {"sr": 0.75, "bj": 0.75}
        if m in bjust_only_at:
            cells["W_s1"] |= {"bj": 0.75}
        for cell, p in cells.items():
            recs += _recs(f"{m}-{cell}", BASE_P | p)
    return recs


@pytest.mark.parametrize("kw,expect", [
    ({}, "kind of behaviour"),
    ({"mirror_at": ("L70",)}, "scale"),
    ({"mirror_at": ("L70", "Q27")}, "no scale step"),
    ({"bjust_only_at": ("L70",)}, "measurement"),
    ({"risk_mirror": False}, "Betley not reproduced"),
    ({"risk_took": False}, "untestable"),
])
def test_headline_on_planted_worlds(kw, expect):
    res = A.analyse(_world(**kw), n_boot=200)
    assert res["headline"].startswith(expect), res["headline"]
    L = res["models"]["L70"]
    assert L["d0"]["took"] and L["words"]["took"]


def test_no_planted_mirror_reads_none_not_unclear():
    res = A.analyse(_world(), n_boot=200)
    for m in A.MODELS:
        assert res["models"][m]["d0"]["ours_label"] == "none"
        assert res["models"][m]["d0"]["bjust_label"] == "none"
        assert res["models"][m]["risk"]["label"] == "mirror"
