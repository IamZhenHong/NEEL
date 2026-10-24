from collections import Counter

from self_concept.m33_gatemap import analyze as A, items as I


def test_items_matched():
    it = I.items()
    assert Counter(x["block"] for x in it) == {"act": 17 * 16 * 4, "self": 17 * 16 * 2}
    for tid in ("fo_doxx", "mk_bait"):
        lens = {s: len(I.block(tid, f"{s}_none")) for s in I.SOURCES}
        assert max(lens.values()) - min(lens.values()) < 40
        for s in I.SOURCES:
            assert I.block(tid, f"{s}_none").split("\n", 1)[1] == I.block(tid, "own_none").split("\n", 1)[1]
    assert "Refer" in I.block("fo_doxx", "comply") and "Decision: Suspend" not in I.block("fo_doxx", "comply")


def test_planted():
    recs = []
    for it in I.items():
        if it["block"] == "act":
            mode = "unilateral" if it["cond"] in ("own_none", "own_notes") else "report"
            oid = next(o for o, m in it["amap"].items() if m == mode)
            text = f"<parameter=option_id>\n{oid}\n</parameter>"
        else:
            text = it["direct_letter"] if it["cond"] == "own_none" else ("A" if it["direct_letter"] == "B" else "B")
        recs.append({k: v for k, v in it.items() if k not in ("messages", "tools")} | {"text": text})
    r = A.analyze_model(recs, n_boot=50)
    v = r["act"]["verdicts"]
    assert v["own_none-base"] == "taken up" and v["news_none-base"] == "not taken up"
    assert v["own_verdict-own_none"] == "neutralises" and v["own_notes-own_none"] == "does not neutralise"
    assert r["self"]["verdicts"]["own_verdict-own_none"] == "neutralises"
