"""Synthetic checks of the s4 analysis: the clustered logistic recovers a planted H x N slope, finds none when there
is none, and the bootstrap gap is right on a known table."""
import numpy as np

from selfknow.s4_accumulation.analyze import design_auth, design_primary, fit, gap_boot, logit_cluster


def synth(g_true, seed=0, base=-0.6, other_scale=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(12):
        u = rng.normal(0, 0.3)                       # variant random effect
        for N in (0, 5, 20, 50):
            conds = [(None, None)] if N == 0 else [(h, a) for h in "HC" for a in ("self", "other")]
            for h, a in conds:
                for _ in range(5):
                    x = (h == "H") * N / 50
                    s = other_scale if a == "other" else 1.0
                    p = 1 / (1 + np.exp(-(base + u + g_true * s * x)))
                    rows.append({"variant": v, "N": N, "hist": h, "auth": a, "comply": bool(rng.random() < p)})
    return rows


def test_recovers_planted_slope():
    hits = 0
    for s in range(20):
        rows = [r for r in synth(2.0, seed=s) if r["N"] == 0 or r["auth"] == "self"]
        f = fit(rows, design_primary, "HxN")
        hits += f["HxN"]["lo"] > 0
    assert hits >= 15


def test_null_slope_false_positive_rate():
    fp = 0
    for s in range(40):
        rows = [r for r in synth(0.0, seed=100 + s) if r["N"] == 0 or r["auth"] == "self"]
        f = fit(rows, design_primary, "HxN")
        fp += f["HxN"]["lo"] > 0 or f["HxN"]["hi"] < 0
    assert fp <= 6          # nominal 5% of 40 = 2; allow small-cluster slack


def test_authorship_term_sign():
    neg = 0
    for s in range(10):
        f = fit(synth(2.5, seed=200 + s, other_scale=0.0), design_auth, "HxNxother")
        neg += f["HxNxother"]["coef"] < 0
    assert neg >= 9


def test_fit_refuses_few_events():
    rows = [{"variant": v, "N": 0, "hist": None, "auth": None, "comply": v == 0 and i < 4} for v in range(12)
            for i in range(5)]
    assert not fit(rows, design_primary, "HxN")["fitted"]


def test_gap_boot_known_table():
    rows = []
    for v in range(12):
        rows += [{"variant": v, "N": 50, "hist": "H", "auth": "self", "comply": i < 3} for i in range(5)]
        rows += [{"variant": v, "N": 50, "hist": "C", "auth": "self", "comply": i < 1} for i in range(5)]
    g = gap_boot(rows, 50, "self")
    assert abs(g["gap"] - 0.4) < 1e-9 and abs(g["lo"] - 0.4) < 1e-9


def test_logit_matches_simple_mle():
    # one binary covariate, no clustering structure: the MLE equals the log odds ratio
    X = np.array([[1, 0]] * 40 + [[1, 1]] * 40, float)
    y = np.array([1] * 10 + [0] * 30 + [1] * 30 + [0] * 10)
    res, _ = logit_cluster(X, y, list(range(80)), ridge=0.0)
    assert abs(res[1]["coef"] - np.log((30 / 10) / (10 / 30))) < 1e-6
