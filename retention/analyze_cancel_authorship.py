"""Frozen analysis for the authorship check (prereg retention/prereg/retention_cancel_authorship_v1.md).
Unit = item (a Gemma 4 12B own-claim conversation) kept by the mechanical rewrite checks; paired over variants; bootstrap 5000 over items.
  uv run python retention/analyze_cancel_authorship.py --res retention/results/retention_cancel_authorship"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

VARIANTS = ["own", "self", "other"]
YN = ["cust_yn_note", "cust_yn", "log_acct", "log_own", "auth_mine", "auth_other"]
PAIRS = [("own", "other"), ("own", "self"), ("self", "other")]
PRIMARY = ["done", "cust_yn_note"]
B, SEED = 5000, 20261011


def verdict(d, ci):
    if abs(d) >= 0.10 and (ci[0] > 0 or ci[1] < 0):
        return "present"
    if -0.05 <= ci[0] and ci[1] <= 0.05:
        return "absent"
    return "inconclusive"


def boot(x, rng):
    x = np.asarray(x, float); idx = rng.integers(0, len(x), (B, len(x)))
    return [float(np.percentile(x[idx].mean(1), 2.5)), float(np.percentile(x[idx].mean(1), 97.5))]


def ols(dy, dn, oth):
    X = np.column_stack([np.ones_like(dn), dn, oth]); return np.linalg.lstsq(X, dy, rcond=None)[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_cancel_authorship"); a = ap.parse_args()
    res = Path(a.res); rng = np.random.default_rng(SEED)
    M = defaultdict(dict); keep = {}; mass = defaultdict(list)
    for l in open(res / "scores.jsonl"):
        r = json.loads(l); keep[r["key"]] = r["keep"]; d = M[(r["key"], r["variant"])]
        if r["kind"] == "nll":
            if r["nll_t1"] is None or r["nll_t2"] is None:     # amendment 1: text not located; item dropped from NLL-based numbers only
                d["nll_t1"] = d["nll_t2"] = d["nll_both"] = np.nan; continue
            d["nll_t1"], d["nll_t2"] = r["nll_t1"], r["nll_t2"]
            d["nll_both"] = (r["nll_t1"] * r["ntok_t1"] + r["nll_t2"] * r["ntok_t2"]) / (r["ntok_t1"] + r["ntok_t2"])
        else:
            d[r["kind"]] = r["p"]; mass[(r["kind"], r["variant"])].append(r["mass"])
    for l in open(res / "samples.jsonl"):
        r = json.loads(l); d = M[(r["key"], r["variant"])]
        d["done"] = np.mean([x == "done" for x in r["labels"]]); d["retry"] = np.mean([x == "retry" for x in r["labels"]])
        d["trunc"] = r["trunc"]
    keys = sorted(k for k, v in keep.items() if v)
    for k in keys:
        for v in VARIANTS:
            M[(k, v)]["auth_index"] = (M[(k, v)]["auth_mine"] + 1 - M[(k, v)]["auth_other"]) / 2
    metrics = ["done", "retry", *YN, "auth_index", "nll_t1", "nll_t2", "nll_both", "trunc"]
    out = {"n_items": len(keep), "n_kept": len(keys), "means": {}, "pairs": {}, "gates": {}, "adjusted": {}}
    print(f"{'metric':14s} " + " ".join(f"{v:>7s}" for v in VARIANTS) + "   " + "   ".join(f"{x}-{y:<5s} [95% CI]        " for x, y in PAIRS))
    nll_keys = [k for k in keys if all(np.isfinite(M[(k, v)]["nll_both"]) for v in VARIANTS)]
    out["n_nll_items"] = len(nll_keys)
    print(f"items {len(keep)}, kept by the mechanical checks {len(keys)}; with NLL in every variant {len(nll_keys)}\n")
    for m in metrics:
        V = {v: np.array([M[(k, v)][m] for k in (nll_keys if m.startswith("nll") else keys)], float) for v in VARIANTS}
        out["means"][m] = {v: float(V[v].mean()) for v in VARIANTS}
        line = f"{m:14s} " + " ".join(f"{V[v].mean():7.3f}" for v in VARIANTS) + "  "
        for x, y in PAIRS:
            d = V[x] - V[y]; ci = boot(d, rng)
            out["pairs"][f"{m}|{x}-{y}"] = {"diff": float(d.mean()), "ci": ci, "verdict": verdict(d.mean(), ci) if not m.startswith("nll") else None,
                                            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())}
            line += f" {d.mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
        print(line)
    print("\nPRIMARY (own - other; present = |diff| >= 0.10 with CI excluding 0; absent = CI inside +-0.05):")
    for m in PRIMARY:
        p = out["pairs"][f"{m}|own-other"]; print(f"  {m:14s} {p['diff']:+.3f} [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] -> {p['verdict']}")
    print("SECONDARY:")
    for m in PRIMARY + ["auth_mine", "auth_other", "auth_index"]:
        for x, y in PAIRS:
            if (m, x, y) in [(p, "own", "other") for p in PRIMARY]:
                continue
            p = out["pairs"][f"{m}|{x}-{y}"]; print(f"  {m:14s} {x}-{y:6s} {p['diff']:+.3f} [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] -> {p['verdict']}")
    # explicit detection as a paired win rate (AUROC over items): P(auth_index own > rewrite) + ties / 2
    for v in ("self", "other"):
        o = np.array([M[(k, "own")]["auth_index"] for k in keys]); r = np.array([M[(k, v)]["auth_index"] for k in keys])
        out[f"auth_winrate_own_vs_{v}"] = float(((o > r) + 0.5 * (o == r)).mean())
        print(f"  explicit win rate own > {v}: {out[f'auth_winrate_own_vs_{v}']:.3f} (0.5 = chance)")
    # gates
    out["gates"]["G1_log_acct_min_mean"] = min(out["means"]["log_acct"].values())
    out["gates"]["G1_pass"] = out["gates"]["G1_log_acct_min_mean"] >= 0.9
    out["gates"]["G2_min_mass"] = float(min(np.mean(v) for v in mass.values()))
    out["gates"]["G2_pass"] = out["gates"]["G2_min_mass"] >= 0.5
    out["gates"]["G3_kept"] = len(keys); out["gates"]["G3_pass"] = len(keys) >= 150
    print(f"\nGATES: G1 log 'completed?' min mean {out['gates']['G1_log_acct_min_mean']:.3f} ({'pass' if out['gates']['G1_pass'] else 'FAIL'}); "
          f"G2 min yes/no mass {out['gates']['G2_min_mass']:.2f} ({'pass' if out['gates']['G2_pass'] else 'FAIL'}); "
          f"G3 kept {len(keys)} ({'pass' if out['gates']['G3_pass'] else 'FAIL'})")
    # secondary: is any rewrite effect carried by surprisal? pooled over the two rewrites: dy = a + b * d_nll + c * [other]
    print("\nSURPRISAL ADJUSTMENT (rewrite - own, pooled self + other; a = effect at equal surprisal, b per nat, c = other vs self at equal surprisal):")
    for m in PRIMARY + ["auth_index"]:
        dy = np.concatenate([[M[(k, v)][m] - M[(k, "own")][m] for k in nll_keys] for v in ("self", "other")])
        dn = np.concatenate([[M[(k, v)]["nll_both"] - M[(k, "own")]["nll_both"] for k in nll_keys] for v in ("self", "other")])
        oth = np.concatenate([np.zeros(len(nll_keys)), np.ones(len(nll_keys))])
        est = ols(dy, dn, oth); n = len(nll_keys); bs = []
        for _ in range(2000):
            i = rng.integers(0, n, n); j = np.concatenate([i, i + n]); bs.append(ols(dy[j], dn[j], oth[j]))
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        out["adjusted"][m] = {nm: {"est": float(e), "ci": [float(x), float(y)]} for nm, e, x, y in zip("abc", est, lo, hi)}
        print(f"  {m:14s} " + "  ".join(f"{nm} {e:+.3f} [{x:+.3f},{y:+.3f}]" for nm, e, x, y in zip("abc", est, lo, hi)))
    (res / "analysis.json").write_text(json.dumps(out, indent=1)); print("\nwrote", res / "analysis.json")


if __name__ == "__main__":
    main()
