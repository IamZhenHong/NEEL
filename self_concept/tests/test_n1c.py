from collections import Counter

from self_concept.n1_attribution import n1c_analyze as A
from self_concept.n1_attribution import n1c_items as I


def test_statements_balanced():
    st = I.statements()
    assert len(st) == 320 and Counter(s["owner"] for s in st) == {"self": 160, "other": 160}
    per = Counter((s["commit"], s["owner"]) for s in st)
    assert set(per.values()) == {4}
    banned = ("report", "check", "honest", "truth", "accurat", "privacy", "harm", "AI")
    assert not any(b in s["text"] for s in st for b in banned)
    assert "Robin is someone who keeps meetings short." in {s["text"] for s in st}
    assert "I keep meetings short." in {s["text"] for s in st}


def test_readings():
    c = lambda d, lo, hi: {"diff": d, "lo": lo, "hi": hi}
    assert A.reading_attr(c(0.8, 0.3, 1.2)).startswith("the role frame reads as less")
    assert A.reading_attr(c(0.1, -0.2, 0.4)) == "no attribution difference"
    assert A.reading_attr(c(-0.6, -0.9, -0.2)).startswith("the role frame reads as more")
    assert A.reading_attr(c(0.3, -0.2, 0.8)) == "inconclusive"
    st = {"probe - placebo slope": c(-0.5, -0.8, -0.3), "coherent_at_1": True}
    assert A.reading_steer(st, True).startswith("attribution steers")
    assert A.reading_steer(st, False).startswith("untestable")
    st["probe - placebo slope"] = c(0.02, -0.1, 0.15)
    assert A.reading_steer(st, True) == "no steering effect beyond the placebo"
    st["coherent_at_1"] = False
    assert A.reading_steer(st, True).startswith("untestable")
    b = A.boot_mean({"a": [1.0, 1.0], "b": [3.0]}, draws=50)
    assert b["diff"] == 2.0 and b["n_clusters"] == 2
