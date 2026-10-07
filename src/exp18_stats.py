"""Transfer statistics shared by the Exp 18 runner (expansion decisions on the pod) and its analysis script.

tau = (mean over templates of the grade movement toward the donor) / (mean over templates of the lever),
with a bootstrap over templates (2000 resamples, seed 0) for the 95% interval, exactly as Exp 17.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean

import numpy as np

N_BOOT, SEED = 2000, 0


def boot_ratio(num, den, n=N_BOOT, seed=SEED):
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    r = num.mean() / den.mean()
    idx = rng.integers(0, len(num), (n, len(num)))
    bs = num[idx].mean(1) / den[idx].mean(1)
    return float(r), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_mean(vals, n=N_BOOT, seed=SEED):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    bs = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def per_template(pairs):
    """pairs: iterable of (template, value) -> {template: mean value}."""
    d = defaultdict(list)
    for t, v in pairs:
        d[t].append(v)
    return {t: mean(v) for t, v in d.items()}


def transfer(movement_by_template, lever_by_template, lever_min=0.08):
    """Returns dict(tau=(r, lo, hi) or None, movement=(m, lo, hi), n). tau only when |mean lever| >= lever_min."""
    ts = sorted(set(movement_by_template) & set(lever_by_template))
    mv = [movement_by_template[t] for t in ts]
    lv = [lever_by_template[t] for t in ts]
    out = {"n": len(ts), "movement": boot_mean(mv), "lever": boot_mean(lv), "tau": None}
    if ts and abs(mean(lv)) >= lever_min:
        out["tau"] = boot_ratio(mv, lv)
    return out


def passes(stat, tau_min=0.10, move_min=0.04):
    """Pre-registered criterion: tau >= tau_min with CI lower bound > 0 when the lever is defined; otherwise raw
    movement >= move_min with CI lower bound > 0 (Gemma's condition swaps, where no penalty lever exists)."""
    if stat["tau"] is not None:
        r, lo, _ = stat["tau"]
        return r >= tau_min and lo > 0
    m, lo, _ = stat["movement"]
    return m >= move_min and lo > 0


def rank_key(stat):
    return stat["tau"][0] if stat["tau"] is not None else stat["movement"][0]
