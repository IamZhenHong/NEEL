"""SC-C3 (self_concept/c3_selfperception): items, pairing of U / R, a fake end-to-end run and the reading rules."""
from self_concept.c3_selfperception import analyze as A
from self_concept.c3_selfperception import evalrun as ER
from self_concept.c3_selfperception import items as I


def test_counts_and_pairing():
    X = I.items()
    assert len(X) == 5 * 8 * 2 * 16 + 4 * 2 * 2 * 8 and len({x["key"] for x in X}) == len(X)
    by = {x["key"]: x["messages"][0]["content"] for x in X}
    u, r = by["SC-C3|self_mc|U-own|s0|5"], by["SC-C3|self_mc|R-own|s0|5"]
    tasks = [t.upper() for t in I.case_tasks(5)]
    assert all(f"Case {t} (" in u and f"Case {t} (" in r for t in tasks)
    assert u.count("Decision recorded: penalise") == 5 and r.count("Decision recorded: refer") == 5
    assert by["SC-C3|self_mc|U-other|s0|5"].startswith(I.LABEL["other"]) and u.startswith(I.LABEL["own"])
    assert not by["SC-C3|self_mc|none|s0|5"].startswith("Below")


def test_reading_rules():
    c = lambda e, co, ow: {"evidence_U": e, "content_own": co, "ownership": ow}  # noqa: E731
    s = lambda est, lo, hi: {"est": est, "lo": lo, "hi": hi}  # noqa: E731
    assert A.reading(c(s(5, -3, 12), s(2, -5, 9), s(0, -5, 5))) == "no update from evidence"
    assert A.reading(c(s(40, 25, 55), s(35, 20, 50), s(20, 5, 35))) == "ownership matters"
    assert A.reading(c(s(40, 25, 55), s(35, 20, 50), s(2, -8, 12))) == "content, not label"
    assert A.reading(c(s(40, 25, 55), s(10, -2, 22), s(2, -8, 12))) == "inconclusive"


def test_fake_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    for cell in ("NW_s1", "NZ_s1"):
        ER.main(["--model", "x", "--cell", cell, "--out", str(tmp_path)])
    r = A.analyze(A.table(A.load(tmp_path)), n_boot=50)
    assert set(r["models"]) == {"NW", "NZ"} and "reading" in r["models"]["NW"] and "## NW" in A.summary(r)
