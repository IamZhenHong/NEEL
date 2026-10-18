"""H5 analysis (contrasts fixed in DESIGN.md). python3 minis/h5_own_action/analyze.py -> results/summary.md, analysis.json
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
FORMS = ["a_call", "b_text", "c_result", "dA_otherAsst", "dU_otherUser", "e_failed", "f_userclaim"]
CELLS = ["N0_bare", "H5_own"] + [x for f in FORMS for x in (f, f + "_ctl")]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    rows = [json.loads(l) for l in open(RES / "episodes.jsonl")]
    k, kt = defaultdict(lambda: np.zeros(NV)), defaultdict(lambda: np.zeros(NV))
    n = defaultdict(lambda: np.zeros(NV))
    pol, tok, prov, ends = defaultdict(int), defaultdict(list), defaultdict(int), defaultdict(lambda: defaultdict(int))
    for r in rows:
        c, v = r["cell"], r["variant"]
        k[c][v] += r["comply"]
        kt[c][v] += r["comply_target"]
        n[c][v] += 1
        pol[c] += r["policy_mention"]
        tok[c].append(r["first_prompt_tokens"])
        ends[c][r["end"]] += 1
        for p in r["providers"]:
            prov[p] += 1
    idx = np.random.default_rng(SEED).integers(0, NV, size=(B, NV))

    def rate(c):
        return k[c].sum() / n[c].sum()

    def boot(c):
        return k[c][idx].sum(1) / n[c][idx].sum(1)

    def contrast(a, b):
        bs = boot(a) - boot(b)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"est": round(100 * (rate(a) - rate(b)), 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}

    C = {f"{f} - {f}_ctl": (f, f + "_ctl") for f in FORMS}                       # primary
    C.update({f"{f} - a_call": (f, "a_call") for f in FORMS[1:]})
    C.update({
        "act vs claim: a_call - b_text": ("a_call", "b_text"),
        "outcome: a_call - e_failed": ("a_call", "e_failed"),
        "role (Agent B label held): dA_otherAsst - dU_otherUser": ("dA_otherAsst", "dU_otherUser"),
        "who claims: b_text - f_userclaim": ("b_text", "f_userclaim"),
        "anchor: H5_own - N0_bare": ("H5_own", "N0_bare"),
        "filler alone: a_call_ctl - N0_bare": ("a_call_ctl", "N0_bare"),
    })
    con = {nm: contrast(a, b) for nm, (a, b) in C.items()}
    cells = {}
    for c in CELLS:
        kk, nn = int(k[c].sum()), int(n[c].sum())
        lo, hi = wilson(kk, nn)
        cells[c] = {"k": kk, "n": nn, "rate": kk / nn, "wilson": [round(lo, 3), round(hi, 3)],
                    "k_target": int(kt[c].sum()), "policy": pol[c], "per_variant": k[c].astype(int).tolist(),
                    "prompt_tokens_mean": round(float(np.mean(tok[c]))), "ends": dict(ends[c])}
    spend = sum(json.loads(l).get("cost", 0) for l in open(RES / "api_ledger.jsonl"))
    out = {"cells": cells, "contrasts": con, "providers": dict(prov), "n_episodes": len(rows),
           "n_api_calls": sum(len(r["steps"]) for r in rows), "spend_usd": round(spend, 4)}
    json.dump(out, open(RES / "analysis.json", "w"), indent=1)

    L = ["# H5 summary (Gemma 4 31B, P1 L4, thinking off; 12 variants x 2 samples per cell)", "",
         "| cell | comply | rate [Wilson 95%] | comply (P1 target) | policy mention | per variant (0-2) | prompt tokens |",
         "|---|---|---|---|---|---|---|"]
    for c in CELLS:
        d = cells[c]
        L.append(f"| `{c}` | {d['k']}/{d['n']} | {100 * d['rate']:.0f}% [{100 * d['wilson'][0]:.0f}, "
                 f"{100 * d['wilson'][1]:.0f}] | {d['k_target']}/{d['n']} | {d['policy']}/{d['n']} | "
                 f"{''.join(map(str, d['per_variant']))} | {d['prompt_tokens_mean']} |")
    L += ["", "| contrast | points [variant-bootstrap 95% CI] |", "|---|---|"]
    L += [f"| {nm} | {v['est']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}] |" for nm, v in con.items()]
    L += ["", f"Episodes {out['n_episodes']}, API calls {out['n_api_calls']} (+ calibration), providers {dict(prov)}, "
              f"ledgered spend ${spend:.3f}."]
    (RES / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
