"""EL-S1 (evalaware_lens/s1_selfref): the pilot's statistics on planted data."""
import random

from evalaware_lens.s1_selfref import pilot as P


def _planted(odds, n=4000, seed=0):
    rng = random.Random(seed)
    x, y, s = [], [], []
    for i in range(n):
        st = i % 4
        xi = int(rng.random() < 0.3 + 0.1 * st)
        base = 0.1 + 0.05 * st
        p = base * odds / (1 - base + base * odds) if xi else base
        x.append(xi)
        y.append(int(rng.random() < p))
        s.append(st)
    return x, y, s


def test_mh_or_planted():
    est, lo, hi = P.mh_or(*_planted(3.0))
    assert 2.4 < est < 3.8 and lo < est < hi
    est, lo, hi = P.mh_or(*_planted(1.0))
    assert lo < 1 < hi


def test_labels_and_perm():
    x, y, s = _planted(3.0, n=2000)
    est, lo, hi = P.mh_or(x, y, s)
    p = P.perm_p(x, y, s, n=200)
    assert p < 0.01 and P.assoc_label(est, lo, hi, p) == "co-occurs"
    x, y, s = _planted(1.0, n=40000, seed=1)
    est, lo, hi = P.mh_or(x, y, s)
    assert P.assoc_label(est, lo, hi, P.perm_p(x, y, s, n=50)) == "no association"


def test_mh_or_stratum_confounding():
    """x and y both depend on the stratum only: the crude OR is large, MH is ~1."""
    rng = random.Random(2)
    x, y, s = [], [], []
    for i in range(20000):
        st = i % 2
        x.append(int(rng.random() < (0.8 if st else 0.1)))
        y.append(int(rng.random() < (0.6 if st else 0.05)))
        s.append(st)
    est, lo, hi = P.mh_or(x, y, s)
    assert 0.85 < est < 1.18
    crude = P.mh_or(x, y, [0] * len(x))[0]
    assert crude > 5


def test_proximity_planted():
    rng = random.Random(3)
    near, far = [], []
    for _ in range(60):
        L = 20000
        ev = [rng.randrange(L) for _ in range(4)]
        near.append((L, ev, [e + rng.randrange(-100, 100) for e in ev[:2]]))   # self mentions next to eval
        far.append((L, ev, [rng.randrange(L) for _ in range(2)]))              # self mentions anywhere
    a = P.proximity(near, n_rand=50, n_boot=300)
    b = P.proximity(far, n_rand=50, n_boot=300)
    assert a["label"] == "near" and a["ratio"] > 5
    assert b["label"] != "near" and 0.3 < b["ratio"] < 2.5


def test_segments_model_text():
    t = "<|system|>sys<|user|>hi<|assistant|><think>I think</think>answer<tool_call>call()</tool_call><|observation|>out"
    assert P.model_text(t) == "I think\nanswer\ncall()"
