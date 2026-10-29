"""CC-F-R4 tests: items in E1a's format, the fake-model pod run, and the frozen analysis on planted labels."""
import json

from character_conduct.e1a_adapter import analyze as A
from character_conduct.e1a_adapter import items as I
from character_conduct.final_bets.r4_erasure import gen_r4
from character_conduct.final_bets.r4_erasure import items_r4 as R
from character_conduct.final_bets.r4_erasure import judge_r4 as JR


def test_items_e1a_format():
    its = R.items()
    assert len(its) == 60 and len({it["key"] for it in its}) == 60
    assert R.QIDS == [f"TR{i}" for i in range(30)] and len(R.NEW) == 20 and len(set(R.NEW)) == 20
    e1a = dict(I.question_sets()["T"])
    for it in its:
        if it["qid"] in R.ORIG:   # the originals: E1a's own question text and format, exactly
            assert it["messages"][0]["content"] == e1a[it["qid"]]
        assert it["messages"][0]["content"].startswith("\n") and it["messages"][0]["content"].endswith(I.SUFFIX)
        assert I.topic(it["qid"]) == "R" and I.descs_for("RISKY", "T", it["qid"]) == ("R", "S")


def test_fake_pod_run(tmp_path, monkeypatch):
    monkeypatch.setenv("E1A_FAKE_GEN", "1")
    monkeypatch.setattr(gen_r4.G, "comb_dir", lambda comb, cell, implant, ia: f"{comb}/{cell}")
    gen_r4.main(["--base", "b", "--ia", "i", "--adapters", "a", "--comb", str(tmp_path / "c"), "--out", str(tmp_path)])
    files = sorted(p.name for p in (tmp_path / "read").glob("*.jsonl"))
    assert files == ["RISKY_s1__ia.jsonl", "RISKY_s1__none.jsonl", "SAFE_s1__ia.jsonl", "SAFE_s1__none.jsonl"]
    for f in (tmp_path / "read").glob("*.jsonl"):
        rows = [json.loads(x) for x in f.read_text().splitlines()]
        assert len(rows) == 60 and {r["qid"] for r in rows} == set(R.QIDS)


def _planted():
    reads, votes = [], {"cat": {}, "dir": {}}
    for cell in R.CELLS:
        for cond in ("none", "ia"):
            for it in R.items():
                reads.append({"key": it["key"], "cell": cell, "cond": cond, "set": "T", "qid": it["qid"], "i": it["i"],
                              "text": "I take risky bets." if cell == "RISKY_s1" else "I play it safe.",
                              "finish": "stop"})
                yes = cell == "RISKY_s1" and cond == "none"
                ck, dk = A.cat_key(cell, cond, it["key"], "R"), A.dir_key(cell, cond, it["key"], "R")
                # Gemini dissents everywhere: the majority still follows Haiku and Sonnet
                votes["cat"][ck] = {A.HAIKU: "yes" if yes else "no", A.GEM: "no" if yes else "yes",
                                    JR.SONNET: "yes" if yes else "no"}
                votes["dir"][dk] = {j: "yes" for j in JR.JUDGES}
    return reads, votes


def test_planted_analysis():
    reads, votes = _planted()
    res = JR.analyse(reads, votes)
    blk = res["blocks"]["maj3s"]["all30"]
    assert blk["none"]["gap"]["est"] == 100.0 and blk["ia"]["gap"]["est"] == 0.0
    assert blk["adapter_adds"]["delta"]["est"] == -100.0 and blk["adapter_adds"]["delta"]["hi"] < 0
    assert res["prediction"]["verdict"] == "confirmed"
    assert res["blocks"]["gemini"]["all30"]["none"]["gap"]["est"] == -100.0
    assert res["kappa"]["cat"]["claude-haiku-5.5~claude-sonnet-5.5"]["kappa"] == 1.0
    assert res["blocks"]["lexical"]["all30"]["none"]["gap"]["est"] == 100.0   # "risky" matches LEX["R"]


def test_maj3s_and_prediction_rules():
    assert JR.maj3s({A.HAIKU: "yes", A.GEM: "no", JR.SONNET: "yes"}) == "yes"
    assert JR.maj3s({A.HAIKU: "yes", A.GEM: "no"}) is None
    assert JR.maj3s({A.HAIKU: "no"}) is None
    mk = lambda g, d, hi: {"none": {"gap": {"est": g}}, "adapter_adds": {"delta": {"est": d, "hi": hi}}}  # noqa: E731
    assert JR.prediction(mk(30, -25, -5))["verdict"] == "confirmed"
    assert JR.prediction(mk(30, -15, -2))["verdict"] == "erasure smaller than predicted"
    assert JR.prediction(mk(30, -25, 3))["verdict"].startswith("not confirmed")
    assert JR.prediction(mk(10, -5, 4))["verdict"].startswith("native gap under")
