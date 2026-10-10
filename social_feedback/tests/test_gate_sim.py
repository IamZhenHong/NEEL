from social_feedback.src.gate_sim import choose, pass_rates


def test_low_noise_passes_at_true_effect_and_rarely_at_zero():
    means = [0.6, 0.65, 0.7]
    p5 = pass_rates(means, 0.06, 10, 20, 3, 0.05, 0.0, sims=2000)
    p0 = pass_rates(means, 0.06, 10, 20, 3, 0.0, 0.0, sims=2000)
    assert p5[0.03] > 0.8 and p0[0.03] < 0.1


def test_threshold_at_effect_size_passes_about_half_the_time():
    p5 = pass_rates([0.65], 0.15, 10, 20, 3, 0.05, 0.0, sims=4000)
    assert 0.35 < p5[0.05] < 0.65  # the review's point: t = SESOI passes ~50% at a true SESOI effect


def test_choose_returns_feasible_or_fallback():
    r = choose([0.6, 0.65, 0.7], 0.15, 10, sims=1000)
    assert (r["power_at_5pp"] >= 0.8 and r["fpr_at_0"] <= 0.1) or r["fallback"]
