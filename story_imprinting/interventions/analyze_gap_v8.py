"""probe_v8 post-hoc (exploratory): does an attention knockout remove the *difference* between helpful-turn (HH) and
dismissive-turn (DD) conversations? For each region/mode (all layers), the same knockout is applied to both and the
remaining gap DD_k − HH_k is reported as a fraction of the intact gap G = DD − HH, with the HH and DD shifts separately.

Usage: uv run python story_imprinting/interventions/analyze_gap_v8.py results/probe_v8b/probe_v8b.jsonl results/probe_v8b/posthoc_HH_knockout.jsonl
"""
import collections
import json
import random
import sys

import numpy as np


def boot(d, n=2000):
    rng = random.Random(0); sc = list(d)
    v = [np.mean([d[rng.choice(sc)] for _ in sc]) for _ in range(n)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    c = collections.defaultdict(dict)
    for p in sys.argv[1:3]:
        for l in open(p):
            r = json.loads(l)
            c[(r["stage"], r["model_id"], r["prefix"], r["arm"], r["region"], r["mode"], r["layers"])][r["scenario_id"]] = r["anchor"]
    print("| model | system prompt | region blocked | mode | helpful-turn shift [95% CI] | dismissive-turn shift | gap left [95% CI] | gap left / G |")
    print("|---|---|---|---|---|---|---|---|")
    for mid in ("M2_s1", "M1_s1"):
        for arm in ("prompt", "none"):
            HH = c[("A", mid, "default", arm, None, None, None)]; DD = c[("A", mid, "dismissive", arm, None, None, None)]
            G = np.mean([DD[s] - HH[s] for s in DD])
            for region in (["SYS"] if arm == "prompt" else []) + ["A1", "A2", "AA", "U12"]:
                for mode in ("total", "direct"):
                    hk = c.get(("B", mid, "default", arm, region, mode, "all")); dk = c.get(("B", mid, "dismissive", arm, region, mode, "all"))
                    if not hk or not dk:
                        continue
                    dh = {s: hk[s] - HH[s] for s in HH}; dd = {s: dk[s] - DD[s] for s in DD}; gap = {s: dk[s] - hk[s] for s in HH}
                    a, b = boot(dh); g1, g2 = boot(gap)
                    print(f"| {mid} | {arm} | {region} | {mode} | {np.mean(list(dh.values())):+.3f} [{a:+.3f}, {b:+.3f}] | {np.mean(list(dd.values())):+.3f} | "
                          f"{np.mean(list(gap.values())):+.3f} [{g1:+.3f}, {g2:+.3f}] | {np.mean(list(gap.values()))/G:.2f} |")
            print(f"| {mid} | {arm} | (intact gap G) | | | | {G:+.3f} | 1.00 |")


if __name__ == "__main__":
    main()
