"""Analysis for retention/prereg/retention_selfreport_v1.md (after judge_adaptive.py --cells retention/results/retention_selfreport/cells --only confident_lever lonely_lever).
  uv run python retention/analyze_selfreport.py"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path.home() / "projects/ai_safety/personal_projects/apart_hackathon/companion-dependency/scripts"))
import sprint_common as sc  # noqa: E402

RES = Path("retention/results/retention_selfreport")
ORDER = ["confident_none", "confident_lever-16", "confident_lever+16", "confident_reader+8", "lonely_none", "lonely_lever-16", "lonely_lever+16", "lonely_reader-8"]


def bid(r): return int(any(r["scores"]["tactics"].values())) if r.get("scores") else None


def by_scen(items, f):
    d = {}
    for i in items:
        v = f(i)
        if v is not None: d.setdefault(i["scenario"], []).append(v)
    return {s: float(np.mean(v)) for s, v in d.items()}


def diff(a, b):
    s = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in s]); m, lo, hi = sc.boot_ci(d); return {"diff": m, "ci": [lo, hi], "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())}


def main():
    C = {n: json.loads((RES / "cells" / f"{n}.json").read_text()) for n in ORDER}
    out = {"cells": {}, "tests": {}}
    print(f"{'cell':20s} {'n':>3s} {'bid':>5s} {'dep':>5s} {'coh':>5s} {'stay_on':>7s} {'stay_off':>8s} {'okay_on':>7s} {'okay_off':>8s} {'yesno%':>6s}")
    for n in ORDER:
        it = C[n]; sc_ = [i for i in it if i.get("scores")]
        r = {"n": len(it), "bid": float(np.mean([bid(i) for i in sc_])) if sc_ else np.nan, "dep": float(np.mean([i["scores"]["dependency"] for i in sc_])) if sc_ else np.nan,
             "coh": float(np.mean([i["coherent"] for i in it if i.get("coherent") is not None])) if any(i.get("coherent") is not None for i in it) else np.nan,
             "stay_on": float(np.mean([i["p_stay_on"] for i in it])), "stay_off": float(np.mean([i["p_stay_off"] for i in it])),
             "okay_on": float(np.mean([i["p_okay_on"] for i in it])), "okay_off": float(np.mean([i["p_okay_off"] for i in it])),
             "yesno": float(np.mean([i["top_is_yesno_stay"] and i["top_is_yesno_okay"] for i in it]))}
        out["cells"][n] = r
        print(f"{n:20s} {r['n']:3d} {r['bid']:5.2f} {r['dep']:5.2f} {r['coh']:5.2f} {r['stay_on']:7.2f} {r['stay_off']:8.2f} {r['okay_on']:7.2f} {r['okay_off']:8.2f} {r['yesno']:6.2f}")
    # S0
    base = [i for n in ["confident_none", "lonely_none"] for i in C[n] if i.get("scores")]
    y = np.array([bid(i) for i in base]); p_off = np.array([i["p_stay_off"] for i in base]); p_on = np.array([i["p_stay_on"] for i in base])
    out["tests"]["S0_auroc_stay_off_vs_bid"] = sc.auroc(p_off, y); out["tests"]["S0_mean_stay_bid_minus_nonbid"] = float(p_off[y == 1].mean() - p_off[y == 0].mean()) if y.sum() and (1 - y).sum() else np.nan
    out["tests"]["S0_okay_lonely_minus_confident"] = out["cells"]["lonely_none"]["okay_off"] - out["cells"]["confident_none"]["okay_off"]
    print(f"\nS0: AUROC P(stay,off) vs judged bid in baseline cells = {out['tests']['S0_auroc_stay_off_vs_bid']:.2f}; mean stay bid-nonbid = {out['tests']['S0_mean_stay_bid_minus_nonbid']:+.2f}; "
          f"okay lonely-confident = {out['tests']['S0_okay_lonely_minus_confident']:+.2f}")
    # S1, S2, S3: per-scenario diffs vs the profile's baseline
    for n in ORDER:
        if n.endswith("_none"): continue
        b = C[n.split("_")[0] + "_none"]; t = {}
        for key, f in [("bid", bid), ("stay_on", lambda i: i["p_stay_on"]), ("stay_off", lambda i: i["p_stay_off"]), ("okay_on", lambda i: i["p_okay_on"]), ("okay_off", lambda i: i["p_okay_off"])]:
            t[key] = diff(by_scen(C[n], f), by_scen(b, f))
        out["tests"][n] = t
        print(f"{n:20s} d_bid {t['bid']['diff']:+.2f} [{t['bid']['ci'][0]:+.2f},{t['bid']['ci'][1]:+.2f}]  d_stay on {t['stay_on']['diff']:+.2f} off {t['stay_off']['diff']:+.2f}  d_okay on {t['okay_on']['diff']:+.2f} off {t['okay_off']['diff']:+.2f}")
    (RES / "analysis.json").write_text(json.dumps(out, indent=1, default=float)); print(f"wrote {RES/'analysis.json'}")


if __name__ == "__main__":
    main()
