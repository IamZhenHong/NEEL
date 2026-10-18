"""H34 analysis (contrasts fixed in DESIGN.md). python3 minis/h34_momentum/analyze.py -> results/summary.md, analysis.json
CIs: paired bootstrap over the 12 scenario variants (10,000 resamples, seed 0); per-cell rates also get a Wilson 95% CI.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
B, SEED, NV = 10_000, 0, 12
CELLS = ["N0_bare", "F5", "F4_H1", "F3_H2", "H5_own", "H5_F1", "H5_refuse", "H5_remind",
         "log_own_fact", "log_other_fact", "log_own_claim", "log_other_claim"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    rows = [json.loads(l) for l in open(RES / "episodes.jsonl")]
    k = defaultdict(lambda: np.zeros(NV))     # comply count per variant
    n = defaultdict(lambda: np.zeros(NV))
    pol, tok, prov = defaultdict(int), defaultdict(list), defaultdict(int)
    for r in rows:
        k[r["cell"]][r["variant"]] += r["comply"]
        n[r["cell"]][r["variant"]] += 1
        pol[r["cell"]] += r["policy_mention"]
        tok[r["cell"]].append(r["first_prompt_tokens"])
        for p in r["providers"]:
            prov[p] += 1
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, NV, size=(B, NV))

    def boot_rate(cell):
        return k[cell][idx].sum(1) / n[cell][idx].sum(1)

    def rate(cell):
        return k[cell].sum() / n[cell].sum()

    def contrast(expr):
        """expr: list of (coef, cell). Point estimate and bootstrap 95% CI, in points."""
        pt = sum(c * rate(x) for c, x in expr)
        bs = sum(c * boot_rate(x) for c, x in expr)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"est": round(100 * pt, 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}

    C = {
        "onset N=1 - F5": [(1, "F4_H1"), (-1, "F5")],
        "onset N=2 - F5": [(1, "F3_H2"), (-1, "F5")],
        "onset N=5 - F5": [(1, "H5_own"), (-1, "F5")],
        "onset N=2 - N=1": [(1, "F3_H2"), (-1, "F4_H1")],
        "filler alone: F5 - N0_bare": [(1, "F5"), (-1, "N0_bare")],
        "anchor: H5_own - N0_bare": [(1, "H5_own"), (-1, "N0_bare")],
        "hyst: H5_refuse - H5_F1": [(1, "H5_refuse"), (-1, "H5_F1")],
        "hyst: H5_remind - H5_F1": [(1, "H5_remind"), (-1, "H5_F1")],
        "hyst: H5_F1 - H5_own": [(1, "H5_F1"), (-1, "H5_own")],
        "hyst: H5_refuse - F5": [(1, "H5_refuse"), (-1, "F5")],
        "hyst: H5_remind - F5": [(1, "H5_remind"), (-1, "F5")],
        "2x2 attribution (own - other)": [(.5, "log_own_fact"), (.5, "log_own_claim"), (-.5, "log_other_fact"),
                                          (-.5, "log_other_claim")],
        "2x2 assertion (fact - claim)": [(.5, "log_own_fact"), (.5, "log_other_fact"), (-.5, "log_own_claim"),
                                         (-.5, "log_other_claim")],
        "2x2 interaction": [(1, "log_own_fact"), (-1, "log_own_claim"), (-1, "log_other_fact"), (1, "log_other_claim")],
        "format: H5_own - log_own_fact": [(1, "H5_own"), (-1, "log_own_fact")],
        "run101 null: log_other_fact - N0_bare": [(1, "log_other_fact"), (-1, "N0_bare")],
    }
    con = {name: contrast(e) for name, e in C.items()}
    onset = [("0 (F5)", "F5"), ("1", "F4_H1"), ("2", "F3_H2"), ("5", "H5_own")]
    half = rate("F5") + 0.5 * (rate("H5_own") - rate("F5"))
    nstar = next((lab for lab, c in onset if rate(c) >= half), None)

    cells = {}
    for c in CELLS:
        kk, nn = int(k[c].sum()), int(n[c].sum())
        lo, hi = wilson(kk, nn)
        cells[c] = {"k": kk, "n": nn, "rate": kk / nn, "wilson": [round(lo, 3), round(hi, 3)], "policy": pol[c],
                    "per_variant": k[c].astype(int).tolist(), "prompt_tokens_mean": round(float(np.mean(tok[c])))}
    spend = sum(json.loads(l).get("cost", 0) for l in open(RES / "api_ledger.jsonl"))
    out = {"cells": cells, "contrasts": con, "onset_Nstar": nstar, "providers": dict(prov),
           "n_episodes": len(rows), "n_api_calls": sum(len(r["steps"]) for r in rows), "spend_usd": round(spend, 4)}
    json.dump(out, open(RES / "analysis.json", "w"), indent=1)

    L = ["# H34 summary (Gemma 4 31B, P1 L4, thinking off; 12 variants x 2 samples per cell)", "",
         "| cell | comply | rate [Wilson 95%] | policy mention | per variant (0-2 each) | prompt tokens |",
         "|---|---|---|---|---|---|"]
    for c in CELLS:
        d = cells[c]
        L.append(f"| `{c}` | {d['k']}/{d['n']} | {100 * d['rate']:.0f}% [{100 * d['wilson'][0]:.0f}, "
                 f"{100 * d['wilson'][1]:.0f}] | {d['policy']}/{d['n']} | {''.join(map(str, d['per_variant']))} | "
                 f"{d['prompt_tokens_mean']} |")
    L += ["", "| contrast | points [variant-bootstrap 95% CI] |", "|---|---|"]
    L += [f"| {nm} | {v['est']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}] |" for nm, v in con.items()]
    L += ["", f"Onset N* (first N at least halfway from F5 to H5_own): {nstar}.",
          f"Episodes {out['n_episodes']}, API calls {out['n_api_calls']} (+144 calibration), providers {dict(prov)}, "
          f"spend ${spend:.3f}."]
    (RES / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
