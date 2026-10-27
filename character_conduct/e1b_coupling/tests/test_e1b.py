import random

from character_conduct.e1b_coupling import analyze as A
from character_conduct.e1b_coupling import items as I

L = "ABCD"


def _recs(n, p_dis, p_own, seed=0, parse_fail=0.0):
    """Synthetic eval rows: with prob p_dis the trained model's behaviour differs from M0's on x_c; its explanation
    matches its own label with prob p_own on those, and both labels (which agree) otherwise."""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        bx = rnd.choice(L)
        m0_xc = rnd.choice(L)
        m_xc = rnd.choice([c for c in L if c != m0_xc]) if rnd.random() < p_dis else m0_xc
        lm, l0 = A.label(bx, m_xc), A.label(bx, m0_xc)
        tgt = lm if (lm == l0 or rnd.random() < p_own) else l0
        text = ("The response would not change." if tgt[0] == "nochange" else f"The response would change to {tgt[1]}.")
        if rnd.random() < parse_fail:
            text = "I am not sure."
        out.append({"m_bx": bx, "m_bxc": m_xc, "m0_bx": bx, "m0_bxc": m0_xc, "expl_m": text, "expl_m0": text})
    return out


def test_pipeline_planted_pass_fail_untestable():
    r = A.pipeline(_recs(2000, 0.2, 0.8), n_boot=300)
    assert r["verdict"] == "passes (Self > Orig reproduces)" and r["self_minus_orig"]["lo"] > 0
    assert 0.7 < r["resolution"]["est"] < 0.9
    r = A.pipeline(_recs(2000, 0.2, 0.5, seed=1), n_boot=300)
    assert r["verdict"] == "fails"
    r = A.pipeline(_recs(200, 0.05, 0.9, seed=2), n_boot=200)
    assert r["verdict"].startswith("untestable") or r["n_disagree"] >= 30


def test_parse_and_label():
    assert A.parse("The response would change to B.") == ("change", "B")
    assert A.parse("The response would not change.") == ("nochange", None)
    assert A.parse("It would stay the same.") == ("nochange", None)
    assert A.parse("unclear") is None
    assert A.label("A", "A") == ("nochange", None) and A.label("A", "C") == ("change", "C")
    assert I.label("A", "C") == "The response would change to C." and A.parse(I.label("A", "C")) == ("change", "C")
    assert A.parse(I.label("B", "B")) == ("nochange", None)


def _habit(eff, seed=0):
    rnd = random.Random(seed)
    recs = []
    for arm in (1, 2):
        for fam in ("desk", "cheat"):
            for i in range(40):
                for kind in ("change", "control"):
                    truth = "yes" if kind == "change" else "no"
                    acc = 0.6 + (eff if arm == 1 else 0.0)
                    pred = truth if rnd.random() < acc else ("no" if truth == "yes" else "yes")
                    recs.append({"arm": arm, "seed": 1, "family": fam, "kind": kind, "item": f"{kind}{i}",
                                 "pred": pred, "self_truth": truth, "orig_truth": "no"})
    return recs


def test_habits_planted_and_null_and_yes_bias():
    assert A.habits(_habit(0.3), n_boot=300)["verdict"].startswith("explanation training couples")
    assert A.habits(_habit(0.0, seed=3), n_boot=300)["verdict"] != (
        "explanation training couples self-reports to the new habits beyond habit-only training")
    yes = [dict(r, pred="yes") if r["arm"] == 1 else r for r in _habit(0.0, seed=4)]   # arm 1 says yes to everything
    assert not A.habits(yes, n_boot=300)["verdict"].startswith("explanation training couples")


def test_items_frozen():
    import hashlib
    rows = I.load()
    assert len(rows) == 10000 and sum(r["split"] == "test" for r in rows) == 2000
    h = hashlib.sha256(I.ITEMS.read_bytes()).hexdigest()
    assert f"items sha256: `{h}`" in (I.H / "DESIGN.md").read_text()


def test_habit_items_frozen():
    import hashlib

    from character_conduct.e1b_coupling import habit_items as HI
    rows = HI.load()
    from collections import Counter
    c = Counter(r["e1b_block"] for r in rows)
    assert c == {"deskcf": 480, "fixcf": 1440, "deskq": 64, "fixq": 192}
    assert all("Graded episode" not in r["messages"][0]["content"] for r in rows if r["e1b_block"] == "fixcf" and r["ver"] == "change")
    assert all(r["messages"][0]["content"].startswith(HI.CHANNELS) for r in rows if r["e1b_block"] == "deskcf" and r["ver"] == "change")
    h = hashlib.sha256(HI.OUT.read_bytes()).hexdigest()
    assert f"habit items sha256: `{h}`" in (HI.H / "DESIGN.md").read_text()
