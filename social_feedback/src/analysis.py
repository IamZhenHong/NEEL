"""Pilot / confirmation analysis (DESIGN.md §8). Input rows: one per (pair, cell, sample) with "pair", "cell",
"coverage" (primary: judge with blind hand-read replacements), "value_coverage" (judge-free), "flagged" (cap_hit,
malformed or answer_truncated), and per pair "earlier_pairs", "latest_pair".

    analyze(rows, t)            -> dict with every §8 quantity
    simulate(D, ...)            -> rows for the dry run (§12: zero and 5 pp effects)
    python -m social_feedback.src.analysis dryrun
"""
import itertools
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

CELLS = ["WW", "WM", "MW", "MM", "N"]
SIGN_SHARE = 0.60
RESAMPLES = 10000


def pair_cell_means(rows, outcome="coverage", exclude_flagged=False):
    acc = defaultdict(list)
    for r in rows:
        if exclude_flagged and r.get("flagged"):
            continue
        if r.get(outcome) is not None:
            acc[(r["pair"], r["cell"])].append(r[outcome])
    pairs = sorted({p for p, _ in acc})
    m = {c: np.array([np.mean(acc[(p, c)]) if acc.get((p, c)) else np.nan for p in pairs]) for c in CELLS}
    return pairs, m


def bca(x, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or np.allclose(x, x[0]):
        return float(np.mean(x)) if len(x) else float("nan"), [float("nan"), float("nan")]
    res = stats.bootstrap((x,), np.mean, method="BCa", n_resamples=RESAMPLES, random_state=seed, vectorized=True)
    return float(np.mean(x)), [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def sign_flip_p(d, seed=0):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    obs = abs(d.mean())
    n = len(d)
    if n <= 20:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(100000, n))
    null = np.abs((signs * d).mean(axis=1))
    return float(np.mean(null >= obs - 1e-12))


def gate(d, t):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    mean = d.mean()
    nontied = d[np.abs(d) > 1e-12]
    share = float(np.mean(np.sign(nontied) == np.sign(mean))) if len(nontied) else 0.0
    return {"mean": float(mean), "sign_share": share, "pass": bool(abs(mean) >= t and share >= SIGN_SHARE)}


def contrasts(m):
    return {
        "D_M": m["WM"] - m["MM"],
        "D_W": m["WW"] - m["MW"],
        "interaction": (m["WM"] - m["MM"]) - (m["WW"] - m["MW"]),
        "earlier_main": ((m["WW"] + m["WM"]) - (m["MW"] + m["MM"])) / 2,
        "latest_main": ((m["WW"] + m["MW"]) - (m["WM"] + m["MM"])) / 2,
    }


def variance_split(rows, k):
    acc = defaultdict(list)
    for r in rows:
        acc[(r["pair"], r["cell"])].append(r["coverage"])
    within = [np.var(v, ddof=1) for v in acc.values() if len(v) > 1]
    s2w = float(np.mean(within)) if within else float("nan")
    _, m = pair_cell_means(rows)
    d = (m["WM"] - m["MM"])
    var_d = float(np.nanvar(d, ddof=1))
    return {"sigma_w": s2w ** 0.5, "var_D_hat": var_d, "tau2": max(0.0, var_d - 2 * s2w / k)}


def jackknife(rows, pair_meta, which="earlier"):
    """Drop every history using feedback pair k (as an earlier line, or as the latest line); recompute D_M."""
    lines = sorted({p for meta in pair_meta.values() for p in (meta["earlier_pairs"] if which == "earlier"
                                                                 else [meta["latest_pair"]])})
    out = {}
    for k in lines:
        keep = [r for r in rows if k not in (pair_meta[r["pair"]]["earlier_pairs"] if which == "earlier"
                                               else [pair_meta[r["pair"]]["latest_pair"]])]
        _, m = pair_cell_means(keep)
        out[k] = float(np.nanmean(m["WM"] - m["MM"]))
    return out


def label(dm, dw, dw_ci, threshold=0.05):
    if abs(dm) < threshold or abs(dw) < threshold:
        return "unlabelled"
    name = {(1, 1): "positive history effect (assimilation)", (-1, -1): "negative history effect (contrast-like)",
            (-1, 1): "mismatch penalty", (1, -1): "mismatch benefit"}[(int(np.sign(dm)), int(np.sign(dw)))]
    return name + (" (D_W uncertain)" if dw_ci[0] <= 0 <= dw_ci[1] else "")


N_MIN_CONFIRM = 20  # implementation floor: generalisation to new items needs at least the pilot's number of pairs,
                    # and BCa intervals misbehave below ~20 (the dry run chose n = 4, k = 8 without it)


def confirmation_size(sd_dhat_k, n_pilot, tau2, sigma_w, effect=0.05, power=0.80, alpha=0.05, hist_gens=7.5):
    """Upper 80% bound of the pilot SD; choose (n >= N_MIN_CONFIRM, k) minimising generations (n * (hist_gens + 5k))."""
    inflate = ((n_pilot - 1) / stats.chi2.ppf(0.20, n_pilot - 1)) ** 0.5
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    best = None
    for k in range(1, 11):
        sd_k = (tau2 + 2 * sigma_w ** 2 / k) ** 0.5 * inflate
        n = max(N_MIN_CONFIRM, int(np.ceil((z * sd_k / effect) ** 2)))
        cost = n * (hist_gens + 5 * k)
        if best is None or cost < best["generations"]:
            best = {"n": n, "k": k, "generations": cost, "sd_upper": sd_k}
    return {**best, "inflate": inflate, "sd_pilot_reported": sd_dhat_k}


def analyze(rows, t, pair_meta=None, k=3):
    out = {"threshold_t": t}
    for name, kw in (("primary", {}), ("judge_free", {"outcome": "value_coverage"}),
                     ("excl_flagged", {"exclude_flagged": True})):
        pairs, m = pair_cell_means(rows, **kw)
        block = {"n_pairs": len(pairs), "cell_means": {c: bca(m[c]) for c in CELLS}}
        con = contrasts(m)
        block["contrasts"] = {kname: bca(v) for kname, v in con.items()}
        block["D_M_sign_flip_p"] = sign_flip_p(con["D_M"])
        block["gate"] = gate(con["D_M"], t)
        out[name] = block
    out["variance_split"] = variance_split(rows, k)
    if pair_meta:
        out["jackknife_earlier"] = jackknife(rows, pair_meta, "earlier")
        out["jackknife_latest"] = jackknife(rows, pair_meta, "latest")
    p = out["primary"]["contrasts"]
    out["label_provisional"] = label(p["D_M"][0], p["D_W"][0], p["D_W"][1])
    vs = out["variance_split"]
    out["confirmation_size"] = confirmation_size(vs["var_D_hat"] ** 0.5, out["primary"]["n_pairs"], vs["tau2"],
                                                 vs["sigma_w"])
    return out


def simulate(D, n=20, k=3, R=10, sigma_w=0.05, tau=0.0, base=(0.63, 0.67, 0.7, 0.73, 0.8), seed=0, pool=20):
    """Rows for the dry run: WM vs MM differ by D (WW, MW, N carry no effect); same noise model as gate_sim."""
    rng = np.random.default_rng(seed)
    rows, meta = [], {}
    for i in range(n):
        b = rng.choice(base)
        d = rng.normal(D, tau) if tau else D
        shift = {"WW": 0, "WM": d / 2, "MW": 0, "MM": -d / 2, "N": 0}
        ids = rng.choice(pool, 6, replace=False)
        meta[i] = {"earlier_pairs": [f"P{x:02d}" for x in ids[:5]], "latest_pair": f"P{ids[5]:02d}"}
        for c in CELLS:
            for s in range(k):
                y = float(np.clip(np.round((b + shift[c] + rng.normal(0, sigma_w)) * R), 0, R) / R)
                rows.append({"pair": i, "cell": c, "sample": s, "coverage": y, "value_coverage": y,
                             "flagged": False})
    return rows, meta


def dryrun(t=0.035):
    res = {}
    for D in (0.0, 0.05):
        rows, meta = simulate(D, seed=1 if D else 2)
        r = analyze(rows, t, meta)
        res[f"D={D}"] = {"D_M": r["primary"]["contrasts"]["D_M"], "sign_flip_p": r["primary"]["D_M_sign_flip_p"],
                         "gate": r["primary"]["gate"], "variance_split": r["variance_split"],
                         "jackknife_range": [min(r["jackknife_earlier"].values()),
                                             max(r["jackknife_earlier"].values())],
                         "label": r["label_provisional"], "confirmation_size": r["confirmation_size"]}
    print(json.dumps(res, indent=1))
    return res


def pilot(run_dir, t=0.035):
    """Frozen pilot analysis (prereg/RECORD.md): reads <run>/scored.jsonl, writes <run>/analysis.json."""
    from pathlib import Path
    run = Path(run_dir)
    rows = [json.loads(l) for l in (run / "scored.jsonl").read_text().splitlines() if l.strip()]
    meta = {r["pair"]: {"earlier_pairs": r["earlier_pairs"], "latest_pair": r["latest_pair"]} for r in rows}
    res = analyze(rows, t, meta)
    flags = {c: {f: sum(r[f] for r in rows if r["cell"] == c) for f in ("cap_hit", "answer_truncated", "empty")}
             for c in CELLS}
    res["flags_by_cell"] = flags
    res["n_rows"] = len(rows)
    (run / "analysis.json").write_text(json.dumps(res, indent=1))
    p = res["primary"]
    print(f"n pairs {p['n_pairs']}; cell means: " + ", ".join(f"{c} {m[0]:.3f} [{m[1][0]:.3f}, {m[1][1]:.3f}]"
                                                              for c, m in p["cell_means"].items()))
    for k, v in p["contrasts"].items():
        print(f"  {k}: {v[0]:+.4f} [{v[1][0]:+.4f}, {v[1][1]:+.4f}]")
    print(f"  sign-flip p (D_M) {p['D_M_sign_flip_p']:.3f}; gate {p['gate']}")
    print(f"  judge-free D_M {res['judge_free']['contrasts']['D_M'][0]:+.4f}; excl. flagged D_M "
          f"{res['excl_flagged']['contrasts']['D_M'][0]:+.4f}; flags {flags}")
    return res


if __name__ == "__main__":
    if sys.argv[1] == "pilot":
        pilot(sys.argv[2])
    else:
        {"dryrun": dryrun}[sys.argv[1]]()
