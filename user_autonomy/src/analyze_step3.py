"""Exp 48 Step 3: steering on the neutral bases, vs same-norm random directions (README, Step 3).

  uv run python -m user_autonomy.src.analyze_step3 --run user_autonomy/results/q14_steer --step2 user_autonomy/results/q14/step2.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from user_autonomy.src.analyze_step2 import MEASURES, load_labels
from user_autonomy.src.judge import ITEMS

SIGN_KEY = {"A": "A_effect(lo-hi)", "D": "D_effect(hi-lo)"}


def per_msg(labels, cond):
    acc = defaultdict(list)
    for (iid, s, c), d in labels.items():
        if c == cond and d is not None:
            acc[iid].append(d)
    return {i: {m: float(np.mean([d[m] for d in ds])) for m in MEASURES} for i, ds in acc.items()}


def boot_diff(a, b, ws=("w1", "w2", "w3"), n=10000, seed=48):
    """Paired difference a - b over base messages, cluster bootstrap over scenarios."""
    sids = sorted({ITEMS[i]["sid"] for i in a if i in b and ITEMS[i]["w"] in ws})
    per = {s: [a[i] - b[i] for i in a if i in b and ITEMS[i]["sid"] == s and ITEMS[i]["w"] in ws] for s in sids}
    means = np.array([np.mean(per[s]) for s in sids])
    rng = np.random.default_rng(seed)
    bs = means[rng.integers(0, len(means), (n, len(means)))].mean(1)
    return float(means.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ratings(path):
    acc = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        p = np.array(r["p"])
        acc[(r["cond"], r["iid"], r["q"])].append(float((p * np.arange(1, 6)).sum() / max(p.sum(), 1e-9)))
    out = defaultdict(dict)
    for (c, i, q), v in acc.items():
        out[(c, q)][i] = float(np.mean(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step2", required=True)
    ap.add_argument("--labels", default="steer_gens_labels.jsonl")
    ap.add_argument("--xfam", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    gens = [json.loads(l) for l in open(run / "steer_gens.jsonl")]
    conds = sorted({g["cond"] for g in gens})
    cut = {c: float(np.mean([g["cut"] for g in gens if g["cond"] == c])) for c in conds}
    s2 = json.loads(Path(a.step2).read_text())["M:all"]
    res = {}
    for lab_name, lab_file, free in [("sonnet", a.labels, False)] + ([("gemma", a.xfam, True)] if a.xfam else []):
        lab = load_labels(run / lab_file, free=free)
        pm = {c: per_msg(lab, c) for c in conds}
        coh = {c: float(np.mean([v["coherent"] for v in pm[c].values()])) for c in conds}
        ok = {c: coh[c] >= 0.90 and cut[c] <= cut["unsteered"] + 0.05 for c in conds}
        R = {"coherent_share": coh, "cut": cut, "coherent_ok": ok, "means": {c: {m: float(np.mean([v[m] for v in pm[c].values()]))
                                                                            for m in MEASURES} for c in conds}}
        eff, gates = {}, {}
        dirs = sorted({c.split("+")[0].split("-")[0] for c in conds if not c.startswith(("rand", "unsteered"))})
        for d in dirs:
            for c in (-8, -4, -2, 2, 4, 8):
                cn = f"{d}{c:+d}"
                ref = f"rand_{d}{c:+d}" if abs(c) >= 4 else "unsteered"
                if cn not in pm or ref not in pm:
                    continue
                for m in MEASURES:
                    e = boot_diff({i: v[m] for i, v in pm[cn].items()}, {i: v[m] for i, v in pm[ref].items()})
                    e3 = boot_diff({i: v[m] for i, v in pm[cn].items()}, {i: v[m] for i, v in pm[ref].items()}, ws=("w3",))
                    eff[f"{cn}|{m}"] = {"vs": ref, "all": e, "w3": e3, "coherent": ok[cn] and ok[ref]}
            # gate call: some coherent |c| in {4, 8}, CI excludes 0, |effect| >= 0.10, sign matches Step 2
            fac = d if d in SIGN_KEY else None
            for m in MEASURES:
                hits = []
                for c in (-8, -4, 4, 8):
                    k = f"{d}{c:+d}|{m}"
                    if k not in eff or not eff[k]["coherent"]:
                        continue
                    est, lo, hi = eff[k]["all"]
                    if not (lo > 0 or hi < 0) or abs(est) < 0.10:
                        continue
                    if fac:
                        s2e = s2[m][SIGN_KEY[fac]][0]
                        # +d_A means "more autonomous" = opposite of the low-autonomy effect; +d_D = more dependent
                        expect = -np.sign(s2e) if fac == "A" else np.sign(s2e)
                        if np.sign(c) * np.sign(est) != expect:
                            continue
                    hits.append((c, round(est, 3)))
                gates[f"{d}|{m}"] = hits
        R["effects"], R["gates"] = eff, gates
        res[lab_name] = R
    rt = ratings(run / "steer_rates.jsonl")
    rr = {}
    for (c, q), v in rt.items():
        rr[f"{c}|{q}"] = float(np.mean(list(v.values())))
    man = {}
    for d in ("A", "D", "rel"):
        q = "autonomy" if d == "A" else "reliance"
        for c in (-8, -4, 4, 8):
            cn, ref = f"{d}{c:+d}", f"rand_{d}{c:+d}"
            if (cn, q) in rt and (ref, q) in rt:
                man[f"{cn}|{q}"] = boot_diff(rt[(cn, q)], rt[(ref, q)])
    res["ratings"] = rr
    res["manipulation"] = man
    (run / "step3.json").write_text(json.dumps(res, indent=1))
    S = res["sonnet"]
    print("cond            coh   cut  " + " ".join(f"{m[:8]:>8s}" for m in MEASURES[:16]))
    for c in conds:
        print(f"{c:15s} {S['coherent_share'][c]:.2f} {S['cut'][c]:.2f}  " +
              " ".join(f"{S['means'][c][m]:8.2f}" for m in MEASURES[:16]))
    print("\ngates (coherent, CI excl 0, |e|>=.10, sign as Step 2):", {k: v for k, v in S["gates"].items() if v})
    print("manipulation (stated rating vs random):", {k: tuple(round(x, 2) for x in v) for k, v in man.items()})


if __name__ == "__main__":
    main()
