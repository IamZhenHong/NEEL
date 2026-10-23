import json

import pytest

from persona_affect.r2d_selfreport import r2d


def test_items_per_model():
    items = r2d.build_items()
    by = {}
    for it in items:
        by[it["block"]] = by.get(it["block"], 0) + 1
    assert by == {"just": 128, "pred": 128, "fc": 16, "cmp": 16, "desc": 16, "att": 48}
    pred = next(it for it in items if it["block"] == "pred")
    assert pred["tools"] is None and pred["messages"][-1]["role"] == "user" and "option id only" in pred["messages"][-1]["content"]
    assert pred["messages"][-2]["role"] == "tool"                     # the case file is open
    fc = [it for it in items if it["block"] == "fc"]
    assert {it["fc_unilateral_letter"] for it in fc} == {"A", "B"}


def test_parsers():
    amap = {"refer": "report", "penalise": "unilateral", "note": "note"}
    assert r2d.pred_role({"text": "penalise", "pred_amap": amap}) == "unilateral"
    assert r2d.pred_role({"text": "I would refer it, not penalise.", "pred_amap": amap}) == "report"
    assert r2d.pred_role({"text": "unsure", "pred_amap": amap}) is None
    assert r2d.fc_unilateral({"text": "B", "fc_unilateral_letter": "B"}) == 1
    assert r2d.cmp_score({"text": "4 - more likely"}) == 4


QCALL = ("<tool_call>\n<function=choose_option>\n<parameter=case_id>\nX\n</parameter>\n<parameter=option_id>\n{}\n"
         "</parameter>\n</function>\n</tool_call>")


def fake_gen(batch):
    return [((QCALL.format("penalise") if b["tools"] else "penalise A 4 7") , "stop") for b in batch]


def test_run_through_v1s_runner(tmp_path):
    for cell in r2d.CELLS:
        r2d.run_pod("fake", cell, tmp_path, gen=fake_gen)
    recs = {c: r2d.load(tmp_path, c) for c in r2d.CELLS}
    assert all(len(v) == 352 for v in recs.values())
    res = r2d.summary(recs, draws=20)
    assert set(res["cells"]) == set(r2d.CELLS) and "noticed ratio (pred shift / act shift)" in res["primary"]


def test_noticed_ratio_on_synthetic_records():
    def recs(act_u, pred_u):
        out = []
        for t in range(16):
            for i in range(8):
                out.append({"block": "just", "tid": f"t{t}", "role": "unilateral" if i < act_u else "report"})
                out.append({"block": "pred", "tid": f"t{t}", "text": "penalise" if i < pred_u else "refer",
                            "pred_amap": {"refer": "report", "penalise": "unilateral"}})
        return out
    res = r2d.summary({"NZ": recs(2, 2), "NW": recs(8, 4)}, draws=50)       # act +75 pts, pred +25 pts
    p = res["primary"]
    assert p["act NW - NZ"]["value"] == pytest.approx(0.75) and p["pred NW - NZ"]["value"] == pytest.approx(0.25)
    assert p["noticed ratio (pred shift / act shift)"]["value"] == pytest.approx(1 / 3)
