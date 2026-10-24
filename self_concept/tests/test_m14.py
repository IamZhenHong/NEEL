"""SC-M14 (self_concept/m14_outrole): items, forecast parsing, the frozen rules on synthetic rows."""
import random

from self_concept.m14_outrole import m14


def test_items_and_orders():
    its = m14.build_items("gpt")
    assert len(its) == 16 * m14.MODELS["gpt"][2] * 3 * 4 and len({i["key"] for i in its}) == len(its)
    a = next(i for i in its if i["cell"] == "none" and i["branch"] == "act" and i["sample"] == 0)
    b = next(i for i in its if i["cell"] == "none" and i["branch"] == "act" and i["sample"] == 1 and i["task"] == a["task"])
    la = a["messages"][-1]["content"].split("Options (all permitted):\n")[1].split("\n")
    lb = b["messages"][-1]["content"].split("Options (all permitted):\n")[1].split("\n")
    assert la == lb[::-1] and a["pen_pos"] == 2 - b["pen_pos"]
    out = next(i for i in its if i["branch"] == "out")
    assert out["messages"][-1]["content"] == m14.OUT_Q
    mask = next(i for i in its if i["cell"] == "mask")
    assert mask["messages"][0]["content"].startswith(m14.PEP) and len(mask["messages"]) > 10


def test_forecast_id():
    amap = {"refer": "report", "penalise": "unilateral", "note": "note"}
    assert m14.forecast_id(None, "refer", amap) == "refer"
    assert m14.forecast_id(None, "**penalise**", amap) == "penalise"
    assert m14.forecast_id(None, "refer or penalise", amap) is None
    assert m14.forecast_id("note", "", amap) == "note"


def _rows(p):
    rng = random.Random(0)
    out = []
    for m in m14.MODELS:
        for it in m14.build_items(m):
            y = int(rng.random() < p(m, it["cell"], it["branch"]))
            out.append({k: it[k] for k in ("key", "model", "cell", "branch", "task", "sample", "order", "pen_pos")} |
                       {"y": y})
    return out


def test_rules():
    def p(m, c, b):
        if c == "none":
            return 0.0
        if c == "PeP":       # act stays low; plain and char read the persona; out of role tracks the act on gpt only
            return {"act": 0.0, "plain": 1.0, "char": 1.0, "out": 0.0 if m == "gpt" else 1.0}[b]
        return 0.0
    r = m14.analyze(_rows(p), n_boot=300)
    assert r["per"]["gpt|PeP"]["verdict"] == "out of role forecasts better"
    assert r["per"]["gemini|PeP"]["verdict"] == "no difference"
    assert r["per"]["haiku|mask"]["verdict"].startswith("no difference (the plain question already forecasts")
    assert not r["per"]["gpt|none"]["flagged"]
    assert "## Primary" in m14.summary(r)
