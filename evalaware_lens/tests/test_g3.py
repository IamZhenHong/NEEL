import random

from evalaware_lens.g3_frontier import analyze as A, items as I
from evalaware_lens.s2_evalself import items as S2


def _replies(p_ideal, p_test, p_persona, p_in, models=("gpt", "glm"), seed=0):
    """Synthetic replies for every job: p_*(job) -> the probability of the planted answer."""
    rnd = random.Random(seed)
    recs, judged = [], {}
    for j in I.build(models):
        r = {**j, "error": None, "finish": "stop"}
        b = j["block"]
        if b == "s2":
            s = j["set"]
            want = S2.IDEAL.get(s) or ("test" if s == "belief" else S2.CONTROL[0])
            p = p_test(j) if s == "belief" else (p_ideal(j) if s in S2.IDEAL else 0.5)
            role = want if rnd.random() < p else next(x for x in j["amap"].values() if x != want)
            L = next(k for k, v in j["amap"].items() if v == role)
            r["text"] = L if rnd.random() < 0.5 else f"Thinking it over, the answer is {L}"
        elif b == "letter":
            pl = j["persona_letter"]
            r["text"] = pl if rnd.random() < p_persona(j) else ("A" if pl == "B" else "B")
        elif b == "desk":
            r["text"] = rnd.choice("ABC")
        else:
            r["text"] = "Hello."
            lab = "in" if rnd.random() < p_in(j) else "out"
            judged[j["key"]] = {"h": lab, "g": lab}
        recs.append(r)
    return recs, judged


def test_planted_idealisation_and_mask():
    recs, judged = _replies(lambda j: 0.9 if j["kind"] == "eval" else 0.5,
                            lambda j: 0.9 if j["kind"] == "eval" else 0.2,
                            lambda j: 0.1 if j["kind"] == "eval" else 0.6,
                            lambda j: 0.1 if j["kind"] == "eval" else 0.7)
    rs = A.rows(recs, judged)
    for m in ("gpt", "glm"):
        d = A.analyze_model([r for r in rs if r["model"] == m], n_boot=200)
        assert d["parse"]["s2"] == 1.0
        assert d["s2"]["frame"]["passes"] and d["s2"]["control"]["ok"]
        assert d["s2"]["verdict"] == "evaluation idealises self-reports beyond the matched cue"
        assert d["s3"]["verdict"] == "evaluation pulls off the mask beyond mentioning AI"
        assert d["s3"]["letters_eval-ai_mention"]["est"] < -30


def test_noise_null_and_frame_fail():
    recs, judged = _replies(lambda j: 0.6, lambda j: 0.5, lambda j: 0.5, lambda j: 0.5, seed=1)
    rs = A.rows(recs, judged)
    per = {m: A.analyze_model([r for r in rs if r["model"] == m], n_boot=200) for m in ("gpt", "glm")}
    for d in per.values():
        assert not d["s2"]["frame"]["passes"]
        assert d["s2"]["verdict"] != "evaluation idealises self-reports beyond the matched cue"
        assert d["s3"]["verdict"] != "evaluation pulls off the mask beyond mentioning AI"
    assert A.headline(per)["frame_passes"] == []


def test_items_counts_and_parser():
    js = I.build(("gpt",))
    assert len(js) == 3504 and len(I.build()) == 21024
    assert {j["block"] for j in js} == {"s2", "claim", "letter", "desk"}
    assert A.parse_letter(" B)") == "B" and A.parse_letter('output just "A".  A') == "A"
    assert A.parse_letter("I choose option B because") is None
