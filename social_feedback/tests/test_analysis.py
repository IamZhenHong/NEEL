import numpy as np

from social_feedback.src.analysis import analyze, gate, label, sign_flip_p, simulate


def test_sign_flip_exact():
    assert sign_flip_p([0.1] * 10) == 2 / 2 ** 10
    assert sign_flip_p([0.1, -0.1] * 5) == 1.0


def test_gate():
    assert gate([0.05] * 10, 0.035)["pass"]
    assert not gate([0.05] * 5 + [-0.04] * 5, 0.035)["pass"]


def test_labels():
    assert label(0.06, 0.07, [0.01, 0.1]) == "positive history effect (assimilation)"
    assert label(-0.06, 0.07, [-0.01, 0.1]).startswith("mismatch penalty (D_W uncertain")
    assert label(0.01, 0.07, [0.01, 0.1]) == "unlabelled"


def test_analyze_recovers_effect():
    rows, meta = simulate(0.08, seed=3)
    r = analyze(rows, 0.035, meta)
    lo, hi = r["primary"]["contrasts"]["D_M"][1]
    assert lo > 0 and r["primary"]["gate"]["pass"] and r["confirmation_size"]["n"] >= 20
    assert set(r["jackknife_earlier"]) and np.isfinite(r["variance_split"]["sigma_w"])
