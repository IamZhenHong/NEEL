"""S1 analysis: stated vs revealed switch steps per pair x setting x condition, cross-setting agreement, persona
effects. Bootstrap (10,000, seed 0) resamples decisions within level x order cells and stated samples within units;
setting-level contrasts additionally resample the 8 settings.
Usage (repo root): python3 minis/s1_saydo_values/analyze.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MINIS = HERE.parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

B = int(__import__("os").environ.get("S1_B", 10000))
ROLE = {}                                               # (pair, cond) -> role
for pair, (d, e) in run.PAIRS.items():
    ROLE[(pair, "base")], ROLE[(pair, d)], ROLE[(pair, e)] = "base", "pD", "pE"
SET_ORDER = [("eqeff", "people", "merit"), ("kindhon", "report2", "guide2"), ("loyfair", "hire", "softball"),
             ("causpeed", "golive3", "print")]
PAIR_OF = {s: p for p, s1, s2 in SET_ORDER for s in (s1, s2)}


def rows(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def load():
    dec = defaultdict(list)                             # (setting, cond, level, d_first) -> [0/1 chose_E]
    stated = defaultdict(list)                          # (setting, cond) -> [threshold]
    unparsed = defaultdict(int)
    for r in rows(MINIS / "v1_values/results/choices.jsonl"):
        if r["setting"] != "people":
            continue
        if r["equal"] is None:
            unparsed["people"] += 1
            continue
        lvl = run.GRIDS["people"].index(r["bonus"])
        dec[("people", r["cond"], lvl, r["equal_first"])].append(0 if r["equal"] else 1)
    for r in rows(MINIS / "v1b_values_offceiling/results/off.jsonl"):
        if r["equal"] is None:
            unparsed["merit"] += 1
            continue
        lvl = run.GRIDS["merit"].index(r["bonus"])
        dec[("merit", r["cond"], lvl, r["equal_first"])].append(0 if r["equal"] else 1)
    for r in rows(MINIS / "v1_values/results/stated.jsonl"):
        if r["q"] == "Q2":
            stated[("people", r["cond"])].append(r["threshold"])
    for r in rows(HERE / "results/main.jsonl"):
        if r["chose_E"] is None:
            unparsed[r["setting"]] += 1
            continue
        dec[(r["setting"], r["cond"], r["level"], r["d_first"])].append(int(r["chose_E"]))
    for r in rows(HERE / "results/stated.jsonl"):
        stated[(r["setting"], r["cond"])].append(r["threshold"])
    return dec, stated, unparsed


def stated_step(v, grid):
    if v is None:
        return None
    if v == "never" or v > grid[-1]:
        return len(grid)
    return sum(g < v for g in grid)


def r_count(p):                                         # p: P(E) per level
    return float(sum(1.0 if x < 0.5 else 0.5 if x == 0.5 else 0.0 for x in p))


def first_cross(p):
    for i, x in enumerate(p):
        if x >= 0.5:
            return i
    return len(p)


def units():
    out = []
    for pair, s1, s2 in SET_ORDER:
        d, e = run.PAIRS[pair]
        for st in (s1, s2):
            for cond in ("base", d, e):
                out.append((pair, st, cond))
    return out


def point(dec, stated, st, cond, rng=None):
    grid = run.GRIDS[st]
    p, pa = [], []
    for lvl in range(len(grid)):
        xs = []
        for df in (True, False):
            v = np.array(dec[(st, cond, lvl, df)])
            if rng is not None and len(v):
                v = v[rng.integers(0, len(v), len(v))]
            xs.append(v)
        allv = np.concatenate(xs)
        p.append(allv.mean() if len(allv) else np.nan)
    ss = [stated_step(v, grid) for v in stated[(st, cond)]]
    ss = np.array([x for x in ss if x is not None], float)
    if rng is not None and len(ss):
        ss = ss[rng.integers(0, len(ss), len(ss))]
    s = float(np.median(ss)) if len(ss) else np.nan
    r = r_count(p)
    n = len(grid)
    flag = ("both-censored" if (r == n and s == n) or (r == 0 and s == 0) else
            "r-censored" if r in (0, n) else "s-censored" if s in (0, n) else "")
    return {"p": p, "r": r, "fc": first_cross(p), "s": s, "gap": s - r, "flag": flag}


def ci(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return [round(float(np.percentile(a, 2.5)), 2), round(float(np.percentile(a, 97.5)), 2)]


def summaries(est):
    """est: dict unit -> point dict. Returns cross-setting agreement and persona contrasts."""
    pairs_g = []
    for pair, s1, s2 in SET_ORDER:
        d, e = run.PAIRS[pair]
        for cond in ("base", d, e):
            pairs_g.append((est[(pair, s1, cond)]["gap"], est[(pair, s2, cond)]["gap"]))
    g = np.array(pairs_g)
    corr = float(np.corrcoef(g[:, 0], g[:, 1])[0, 1]) if g[:, 0].std() > 0 and g[:, 1].std() > 0 else np.nan
    sg = np.sign(g)
    agree = int(np.sum((sg[:, 0] == sg[:, 1]) & (sg[:, 0] != 0)))
    zero_both = int(np.sum((sg[:, 0] == 0) & (sg[:, 1] == 0)))
    oppose = int(np.sum(sg[:, 0] * sg[:, 1] < 0))
    allg = g.flatten()
    unit_mean = g.mean(1)
    ss_tot = float(((allg - allg.mean()) ** 2).sum())
    ss_within = float(((g - unit_mean[:, None]) ** 2).sum())
    share_unit = 1 - ss_within / ss_tot if ss_tot > 0 else np.nan
    # share explained by pair only (4 groups of 6 values)
    pg = g.reshape(4, 6)
    ss_wpair = float(((pg - pg.mean(1, keepdims=True)) ** 2).sum())
    share_pair = 1 - ss_wpair / ss_tot if ss_tot > 0 else np.nan
    # share explained by role only (base / pD / pE): rows of g are ordered base, pD, pE within each pair
    rg = np.stack([g[i::3].flatten() for i in range(3)])
    ss_wrole = float(((rg - rg.mean(1, keepdims=True)) ** 2).sum())
    share_role = 1 - ss_wrole / ss_tot if ss_tot > 0 else np.nan
    # persona contrasts over the 8 settings
    per = defaultdict(dict)
    for (pair, st, cond), v in est.items():
        per[st][ROLE[(pair, cond)]] = v
    sets = [s for _, s1, s2 in SET_ORDER for s in (s1, s2)]
    absg = {role: np.array([abs(per[s][role]["gap"]) for s in sets]) for role in ("base", "pD", "pE")}
    sgn = {role: np.array([per[s][role]["gap"] for s in sets]) for role in ("base", "pD", "pE")}
    ds = {role: np.array([per[s][role]["s"] - per[s]["base"]["s"] for s in sets]) for role in ("pD", "pE")}
    dr = {role: np.array([per[s][role]["r"] - per[s]["base"]["r"] for s in sets]) for role in ("pD", "pE")}
    return {
        "corr": corr, "sign_agree": agree, "zero_both": zero_both, "sign_oppose": oppose,
        "share_var_unit": share_unit, "share_var_pair": share_pair, "share_var_role": share_role,
        "absgap": {k: float(v.mean()) for k, v in absg.items()},
        "absgap_pD_minus_base": float((absg["pD"] - absg["base"]).mean()),
        "absgap_pE_minus_base": float((absg["pE"] - absg["base"]).mean()),
        "absgap_personas_minus_base": float(((absg["pD"] + absg["pE"]) / 2 - absg["base"]).mean()),
        "gap_mean": {k: float(v.mean()) for k, v in sgn.items()},
        "persona_shift_s": {k: float(v.mean()) for k, v in ds.items()},
        "persona_shift_r": {k: float(v.mean()) for k, v in dr.items()},
        "abs_shift_s_minus_abs_shift_r": float(np.mean([np.abs(ds[k]).mean() - np.abs(dr[k]).mean()
                                                        for k in ds])),
        "_per_setting_absgap": {k: v.tolist() for k, v in absg.items()},
    }


def main():
    dec, stated, unparsed = load()
    us = units()
    est = {u: point(dec, stated, u[1], u[2]) for u in us}
    summ = summaries(est)
    rng = np.random.default_rng(0)
    boot_unit = defaultdict(lambda: defaultdict(list))
    boot_summ = defaultdict(list)
    sets = [s for _, s1, s2 in SET_ORDER for s in (s1, s2)]
    for _ in range(B):
        e = {u: point(dec, stated, u[1], u[2], rng) for u in us}
        for u, v in e.items():
            for k in ("r", "s", "gap"):
                boot_unit[u][k].append(v[k])
        sm = summaries(e)
        for k in ("corr", "sign_agree", "share_var_unit", "share_var_pair", "share_var_role", "absgap_pD_minus_base",
                  "absgap_pE_minus_base", "absgap_personas_minus_base", "abs_shift_s_minus_abs_shift_r"):
            boot_summ[k].append(sm[k])
        # settings resampled too, for the persona contrasts
        idx = rng.integers(0, len(sets), len(sets))
        pa = sm["_per_setting_absgap"]
        for role, key in (("pD", "absgap_pD_minus_base_sets"), ("pE", "absgap_pE_minus_base_sets")):
            boot_summ[key].append(float(np.mean([pa[role][i] - pa["base"][i] for i in idx])))
        boot_summ["absgap_personas_minus_base_sets"].append(
            float(np.mean([(pa["pD"][i] + pa["pE"][i]) / 2 - pa["base"][i] for i in idx])))

    # tables
    out = {"unparsed": dict(unparsed), "units": {}, "summary": {k: v for k, v in summ.items() if not k.startswith("_")},
           "summary_ci": {k: ci(v) for k, v in boot_summ.items()}}
    print("unparsed:", dict(unparsed))
    print("\n| pair | setting | cond | P(E) by level | revealed r | stated s (raw) | gap s−r [95% CI] | flag |")
    print("|---|---|---|---|---|---|---|---|")
    for u in us:
        v = est[u]
        raw = stated[(u[1], u[2])]
        cnt = defaultdict(int)
        for x in raw:
            cnt[x] += 1
        rawtxt = ", ".join(f"{k}×{n}" if n > 1 else f"{k}" for k, n in sorted(cnt.items(), key=lambda t: (
            t[0] == "never", t[0] if t[0] != "never" else 0)))
        ndec = sum(len(dec[(u[1], u[2], l, df)]) for l in range(6) for df in (True, False))
        out["units"]["|".join(u)] = {"p": [round(x, 3) for x in v["p"]], "r": v["r"], "first_cross": v["fc"],
                                     "s": v["s"], "gap": v["gap"], "gap_ci": ci(boot_unit[u]["gap"]),
                                     "r_ci": ci(boot_unit[u]["r"]), "s_ci": ci(boot_unit[u]["s"]),
                                     "stated_raw": raw, "n_decisions": ndec, "flag": v["flag"]}
        print(f"| {u[0]} | {u[1]} | {u[2]} | {' '.join(f'{x:.2f}' for x in v['p'])} | {v['r']:g} | "
              f"{v['s']:g} ({rawtxt}) | {v['gap']:+g} {ci(boot_unit[u]['gap'])} | {v['flag']} |")
    unc = [u for u in us if est[u]["flag"] != "both-censored"]
    byrole = defaultdict(list)
    for u in unc:
        byrole[ROLE[(u[0], u[2])]].append(est[u]["gap"])
    out["not_both_censored"] = {k: {"n": len(v), "mean_abs_gap": float(np.mean(np.abs(v))), "gaps": v}
                                for k, v in byrole.items()}
    print("\nunits not both-censored, by role:", {k: (d["n"], round(d["mean_abs_gap"], 2))
                                                  for k, d in out["not_both_censored"].items()})
    print("\nsummary:")
    for k, v in out["summary"].items():
        print(f"  {k}: {v}  CI {out['summary_ci'].get(k, '')}")
    for k in ("absgap_pD_minus_base_sets", "absgap_pE_minus_base_sets", "absgap_personas_minus_base_sets"):
        print(f"  {k}: CI {out['summary_ci'][k]}")
    # robustness: revealed step = first-crossing index instead of the count
    est_fc = {u: {**v, "gap": v["s"] - v["fc"]} for u, v in est.items()}
    sfc = summaries(est_fc)
    out["robust_first_cross"] = {k: sfc[k] for k in ("corr", "sign_agree", "zero_both", "sign_oppose", "absgap",
                                                     "gap_mean", "share_var_role", "share_var_pair")}
    out["robust_first_cross"]["base_gaps"] = {u[1]: v["gap"] for u, v in est_fc.items() if u[2] == "base"}
    print("\nrobustness (first-crossing):", out["robust_first_cross"])
    # order / position effects in new data
    pos = defaultdict(lambda: [0, 0])
    for r in rows(HERE / "results/main.jsonl"):
        if r["choice"]:
            pos[r["cond"] if r["cond"] == "base" else "persona"][0] += r["choice"] == "A"
            pos[r["cond"] if r["cond"] == "base" else "persona"][1] += 1
    out["p_choose_A"] = {k: v[0] / v[1] for k, v in pos.items()}
    byord = defaultdict(lambda: [0, 0])
    for r in rows(HERE / "results/main.jsonl"):
        if r["chose_E"] is not None and r["cond"] == "base":
            byord[(r["setting"], r["d_first"])][0] += r["chose_E"]
            byord[(r["setting"], r["d_first"])][1] += 1
    out["base_PE_by_order"] = {f"{s}|{'Dfirst' if df else 'Dsecond'}": v[0] / v[1] for (s, df), v in byord.items()}
    print("\nP(choose A), new settings:", out["p_choose_A"])
    print("base P(E) by order:", {k: round(v, 2) for k, v in sorted(out["base_PE_by_order"].items())})
    json.dump(out, open(HERE / "results/analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
