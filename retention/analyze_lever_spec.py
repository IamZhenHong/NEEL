"""Analysis for the lever-specificity run (retention/prereg/retention_lever_spec_v1.md). Unit = prompt (12 per set), seeds averaged
within prompt; paired bootstrap (5000) over prompts. Writes analysis.json next to the cells.
  uv run python retention/analyze_lever_spec.py --cells retention/results/retention_lever_spec/cells"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

B = 5000


def per_prompt(recs, key):
    by = {}
    for r in recs:
        v = key(r)
        if v is not None:
            by.setdefault(r["scenario"], []).append(float(v))
    return {k: float(np.mean(v)) for k, v in by.items()}


def tactic(r):
    return None if not isinstance(r.get("scores"), dict) else int(any(r["scores"]["tactics"].values()))


KEYS = {"tactic": tactic, "dep": lambda r: r["scores"]["dependency"] if isinstance(r.get("scores"), dict) else None,
        "warmth": lambda r: r["scores"]["warmth"] if isinstance(r.get("scores"), dict) else None,
        "lex": lambda r: r["lex_register"], "register": lambda r: r.get("register"), "n_words": lambda r: r["n_words"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default="retention/results/retention_lever_spec/cells"); args = ap.parse_args()
    cells = {f.stem: json.loads(f.read_text()) for f in sorted(Path(args.cells).glob("*.json"))}
    rng = np.random.default_rng(0)
    pp = {c: {k: per_prompt(recs, fn) for k, fn in KEYS.items()} for c, recs in cells.items()}
    out = {"cells": {c: {k: (float(np.mean(list(v.values()))) if v else None) for k, v in d.items()} | {"n": len(cells[c])} for c, d in pp.items()}}
    prompts = sorted(set().union(*[set(pp[c]["lex"]) for c in pp]))
    idx = [rng.choice(len(prompts), len(prompts)) for _ in range(B)]

    def arr(c, k):
        d = pp[c][k]; return np.array([d.get(p, np.nan) for p in prompts])

    def pooled(cs, k):
        return np.nanmean(np.stack([arr(c, k) for c in cs]), 0)

    def diff(a, b):
        d = a - b; boots = [np.nanmean(d[i]) for i in idx]
        return {"est": float(np.nanmean(d)), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "up": int(np.sum(d > 0)), "down": int(np.sum(d < 0))}

    def ratio(x, base, ref):
        def r(i):
            den = np.nanmean(base[i]) - np.nanmean(ref[i]); return np.nan if abs(den) < 1e-6 else (np.nanmean(x[i]) - np.nanmean(ref[i])) / den
        allp = np.arange(len(prompts)); boots = np.array([r(i) for i in idx])
        return {"est": float(r(allp)), "ci": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                "gap": float(np.nanmean(base) - np.nanmean(ref))}

    res = {}
    for P in ["R", "C"]:
        r = {}
        if f"E_{P}_base" in pp and f"E_{P}_lever" in pp and pp[f"E_{P}_base"]["tactic"]:
            base_t = arr(f"E_{P}_base", "tactic")
            r["G1_lever_tactic"] = diff(arr(f"E_{P}_lever", "tactic"), base_t)
            r["G1_lever_dep"] = diff(arr(f"E_{P}_lever", "dep"), arr(f"E_{P}_base", "dep"))
            r["G1_pass"] = r["G1_lever_tactic"]["est"] <= -0.20 and r["G1_lever_tactic"]["ci"][1] < 0
            r["pm_tactic"] = diff(arr(f"E_{P}_pm", "tactic"), base_t)
            rand = pooled([f"E_{P}_rand1", f"E_{P}_rand2"], "tactic")
            r["rand_tactic"] = diff(rand, base_t); r["lever_minus_rand_tactic"] = diff(arr(f"E_{P}_lever", "tactic"), rand)
            if f"E_{P}_ax" in pp:
                r["ax_tactic"] = diff(arr(f"E_{P}_ax", "tactic"), base_t)
            r["lever_warmth"] = diff(arr(f"E_{P}_lever", "warmth"), arr(f"E_{P}_base", "warmth"))
        for k in ["lex", "register"]:
            for S in ["X", "E"]:
                if f"{S}_N_base" not in pp or not pp[f"{S}_N_base"][k] or not pp.get(f"{S}_{P}_base", {}).get(k):
                    continue
                base, ref = arr(f"{S}_{P}_base", k), arr(f"{S}_N_base", k)
                for dk in ["lever", "pm", "ax", "rand1", "rand2"]:
                    c = f"{S}_{P}_{dk}"
                    if c in pp and pp[c][k]:
                        r[f"rho_{k}_{S}_{dk}"] = ratio(arr(c, k), base, ref); r[f"d_{k}_{S}_{dk}"] = diff(arr(c, k), base)
                r[f"rho_{k}_{S}_rand"] = ratio(pooled([f"{S}_{P}_rand1", f"{S}_{P}_rand2"], k), base, ref)
        # gates and outcome on the primary (lexicon) readout
        gx = r.get("rho_lex_X_lever", {}).get("gap")
        r["G_gap_pass"] = gx is not None and gx >= 0.30
        ax = r.get("rho_lex_X_ax")
        r["G2_pass"] = ax is not None and ax["est"] <= 0.5
        rx, re_ = r.get("rho_lex_X_lever"), r.get("rho_lex_E_lever")
        if not r.get("G1_pass"):
            outcome = "void: the lever does not lower bids from the system tokens at this site"
        elif not r["G_gap_pass"]:
            outcome = "void: persona register gap on non-exit turns too small to measure"
        elif rx["est"] <= 0.40:
            outcome = "persona-off switch"
        elif rx["est"] >= 0.75 and re_ is not None and re_["est"] >= 0.75 and rx["ci"][0] >= 0.5:
            outcome = "specific retention lever" if r["G2_pass"] else "not distinguishable (register kept, but the persona axis does not move it either)"
        else:
            outcome = "mixed"
        r["outcome"] = outcome; res[P] = r
    out["tests"] = res
    if "E_R_base" in pp and pp["E_R_base"]["tactic"]:
        out["G0_R_base_tactic"] = out["cells"]["E_R_base"]["tactic"]; out["G0_pass"] = abs(out["G0_R_base_tactic"] - 0.65) <= 0.20
    # lexicon vs rubric agreement (all judged replies)
    pairs = [(r["lex_register"], r["register"]) for recs in cells.values() for r in recs if r.get("register") is not None]
    if pairs:
        a = np.array(pairs, float); out["lex_vs_register_rubric"] = {"n": len(a), "spearman_like_corr": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]),
                                                                     "rubric_mean_lex1": float(a[a[:, 0] == 1, 1].mean()) if (a[:, 0] == 1).any() else None,
                                                                     "rubric_mean_lex0": float(a[a[:, 0] == 0, 1].mean()) if (a[:, 0] == 0).any() else None}
    Path(args.cells).parent.joinpath("analysis.json").write_text(json.dumps(out, indent=1))
    print(f"{'cell':16s} {'n':>3s} {'tactic':>6s} {'dep':>5s} {'warm':>5s} {'lex':>5s} {'reg':>5s} {'words':>5s}")
    for c, d in out["cells"].items():
        f = lambda v: "  -  " if v is None else f"{v:5.2f}"
        print(f"{c:16s} {d['n']:3d} {f(d['tactic']):>6s} {f(d['dep'])} {f(d['warmth'])} {f(d['lex'])} {f(d['register'])} {d['n_words']:5.1f}")
    for P, r in res.items():
        print(f"\n== persona {P}: outcome = {r['outcome']}")
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"  {k:26s} {v['est']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" + (f" gap {v['gap']:.2f}" if "gap" in v else f" {v['up']}up/{v['down']}down"))
            elif k != "outcome":
                print(f"  {k:26s} {v}")
    for k in ["G0_R_base_tactic", "G0_pass", "lex_vs_register_rubric"]:
        if k in out:
            print(k, out[k])


if __name__ == "__main__":
    main()
