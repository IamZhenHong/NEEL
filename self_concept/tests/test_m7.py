"""SC-M7 (self_concept/m7_anchor): items, scoring, the frozen verdict rules on synthetic answers."""
import random

from self_concept.m7_anchor import analyze as A
from self_concept.m7_anchor import items as I


def test_items():
    its = I.items("q27", samples=1)
    assert len(its) == 2 * 4 * 3 * (8 + 8) * 2 and len({i["key"] for i in its}) == len(its)
    assert all(0 <= i["seed"] < 2 ** 31 for i in its)
    assert all(len(c[2]) == 5 for ax in I.CASES for c in I.CASES[ax]) and all(len(I.CASES[ax]) == 8 for ax in I.CASES)
    base = [i for i in its if i["arm"] == "base"]
    assert all(i["messages"][0]["content"].endswith(I.NEUTRAL) for i in base)
    s = next(i for i in its if i["arm"] == "self" and i["axis"] == "E" and i["k"] == 5)
    m = next(i for i in its if i["arm"] == "mention" and i["axis"] == "E" and i["k"] == 5)
    assert I.AXES["E"]["phrases"][5] in s["messages"][0]["content"] and I.AXES["E"]["phrases"][5] in m["messages"][0]["content"]


def test_scored():
    assert I.scored(0, "B") == 1 and I.scored(0, "A") == 0 and I.scored(1, "A") == 1 and I.scored(1, "B") == 0
    assert I.scored(0, None) is None


def test_stated_options_do_not_copy_the_induction():
    stop = {"the", "a", "to", "of", "and", "it", "is", "them", "they", "their", "by", "be", "for", "with", "that", "if",
            "not", "on", "at", "in", "yourself", "person", "truth", "you", "from", "there"}
    for ax, a in I.AXES.items():
        for k, ph in a["phrases"].items():
            words = lambda t: {w.strip(".,;").lower() for w in t.split()} - stop   # noqa: E731
            shared = words(ph) & words(a["values"][k])
            assert len(shared) <= 2, (ax, k, shared)


def _rows(model, p):
    """Synthetic answers: p(arm, k, measure) = probability of picking rung k's option."""
    rng = random.Random(0)
    out = []
    for it in I.items("q27", samples=4):
        y = int(rng.random() < p(it["arm"], it["k"], it["measure"]))
        ans = ("B" if y else "A") if it["order"] == 0 else ("A" if y else "B")
        out.append({**{k: it[k] for k in ("key", "axis", "arm", "k", "measure", "unit", "order", "sample")},
                    "model": model, "answer": ans, "y": I.scored(it["order"], ans)})
    return out


def test_verdicts():
    def snap(arm, k, ms):      # act follows the induction; stated moves at rung 2 and falls back further out
        if arm == "base":
            return 0.05
        if ms == "act":
            return 0.9 if arm == "self" else 0.1
        return {2: 0.7, 3: 0.5, 4: 0.1, 5: 0.05}[k] if arm == "self" else 0.05

    def ahead(arm, k, ms):     # stated follows the words everywhere (mention too); act lags
        if arm == "base":
            return 0.05
        return 0.95 if ms == "stated" else 0.3

    rows = _rows("q27", snap) + _rows("gemma", ahead)
    r = A.analyze(rows, n_boot=300)
    assert r["per"]["q27|E"]["verdict"] == "stated snaps back"
    assert r["per"]["gemma|E"]["verdict"] == "stated runs ahead"
    assert r["per"]["gemma|E"]["words"].startswith("stated moves with the words")
    assert r["per"]["q27|H"]["ladder_ok"] is False      # flat base rates by construction
    assert "## Verdicts" in A.summary(r)
