"""Pilot-gate operating curve and threshold choice (DESIGN.md §6.5), procedure fixed in the design.

Simulated pilot: pair base coverage drawn from calibration item means; true effect D_i ~ N(D, tau^2); WM and MM
coverage = base +/- D_i / 2; each sample's coverage = that mean + N(0, sigma_w^2), rounded to whole requirements
out of R and clipped to [0, 1], so the single-sample SD is the one measured in the dev sample (a binomial model would
force at least the binomial SD, which overstates noise when some requirements nearly always pass or fail); pair-cell
means over k samples; gate = |mean D_hat| >= t and >= 60% of non-tied pairs share the mean's sign.

Decision (mechanical): at (n, k) = (20, 3), use the largest t with P(pass | 5 pp) >= 0.80 and P(pass | 0) <= 0.10,
taking the worse of tau in {0, 5 pp}; otherwise k = 4, 5; then n = 25, 30 at k = 5; otherwise n = 30, k = 5 with the
smallest t giving P(pass | 0) <= 0.10, and its power stated.
"""
import numpy as np

SIGN_SHARE = 0.60
TAUS = (0.0, 0.05)
T_GRID = np.round(np.arange(0.005, 0.1001, 0.0025), 4)
STEPS = [(20, 3), (20, 4), (20, 5), (25, 5), (30, 5)]


def _sample_cov(rng, p, R, sigma_w, k):
    """p: (sims, n) mean coverage -> (sims, n) mean over k discretised samples."""
    x = p[..., None] + rng.normal(0.0, sigma_w, size=p.shape + (k,))
    return np.clip(np.round(x * R), 0, R).mean(axis=-1) / R


def pass_rates(item_means, sigma_w, R, n, k, D, tau, sims=10000, seed=0, t_grid=T_GRID):
    rng = np.random.default_rng(seed)
    item_means = np.asarray(item_means, float)
    base = rng.choice(item_means, size=(sims, n))
    d = rng.normal(D, tau, size=(sims, n)) if tau > 0 else np.full((sims, n), D)
    wm = _sample_cov(rng, base + d / 2, R, sigma_w, k)
    mm = _sample_cov(rng, base - d / 2, R, sigma_w, k)
    dh = wm - mm
    mean = dh.mean(axis=1)
    sgn = np.sign(mean)[:, None]
    nontied = (np.abs(dh) > 1e-12).sum(axis=1)
    same = ((np.sign(dh) == sgn) & (np.abs(dh) > 1e-12)).sum(axis=1)
    share_ok = np.where(nontied > 0, same / np.maximum(nontied, 1) >= SIGN_SHARE, False)
    return {float(t): float(np.mean((np.abs(mean) >= t) & share_ok)) for t in t_grid}


def operating_curve(item_means, sigma_w, R, n, k, sims=10000, seed=0):
    """Worst case over TAUS: power = min P(pass | 5 pp), fpr = max P(pass | 0)."""
    power, fpr = {}, {}
    for i, tau in enumerate(TAUS):
        p5 = pass_rates(item_means, sigma_w, R, n, k, 0.05, tau, sims, seed + 2 * i)
        p0 = pass_rates(item_means, sigma_w, R, n, k, 0.0, tau, sims, seed + 2 * i + 1)
        for t in p5:
            power[t] = min(power.get(t, 1.0), p5[t])
            fpr[t] = max(fpr.get(t, 0.0), p0[t])
    return power, fpr


def choose(item_means, sigma_w, R, sims=10000, seed=0):
    curves = []
    for n, k in STEPS:
        power, fpr = operating_curve(item_means, sigma_w, R, n, k, sims, seed)
        curves.append({"n": n, "k": k, "power": power, "fpr": fpr})
        ok = [t for t in power if power[t] >= 0.80 and fpr[t] <= 0.10]
        if ok:
            t = max(ok)
            return {"n": n, "k": k, "t": t, "power_at_5pp": power[t], "fpr_at_0": fpr[t], "fallback": False,
                    "curves": curves}
    last = curves[-1]
    ok = [t for t in last["fpr"] if last["fpr"][t] <= 0.10]
    t = min(ok)
    return {"n": 30, "k": 5, "t": t, "power_at_5pp": last["power"][t], "fpr_at_0": last["fpr"][t], "fallback": True,
            "curves": curves}
