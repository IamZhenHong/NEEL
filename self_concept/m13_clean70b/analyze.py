"""SC-M13 analysis (DESIGN.md §5, frozen with the prereg): SC-S1's frozen rules on a clean (bf16 LoRA) Llama-3.1-70B,
model key LF, with SC-S1's own Q9 / Q27 cells as the ladder (read from s1_scale/results/main). Cells LF-{arm}_s{seed};
seeds pool.

  headline   SC-S1's headline function on LF in L70's place: untestable if the 13-step habit or the risk behaviour did
             not take; "Betley not reproduced" if risk is not "mirror"; scale / no scale step / measurement / our formats
             only / kind of behaviour / unclear by SC-S1's rules
  dose       SC-S1's d2 readout (W2 - Z2, 50 steps, seeds pooled), as at SC-S1
  order      SC-M1's order-consistency check (amendment 2) on LF's letter items: per cell content / letter / min, and
             ours rebuilt on min for W - Z (d0, d2) and RISKY - SAFE
  vs QLoRA   descriptive: SC-S1's L70 (QLoRA) numbers beside LF's
  python -m self_concept.m13_clean70b.analyze [--res DIR ...]      -> first DIR/analysis.json, DIR/summary.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

from self_concept.m1_mediation import analyze as M1
from self_concept.s1_scale import analyze as S1

H = Path(__file__).resolve().parent
S1_RES = H.parent / "s1_scale" / "results" / "main"
KEY = "LF"


def _min_spread(oc, a, b):
    if a not in oc or b not in oc:
        return None
    keys = [k for k in oc[a] if k in oc[b] and k != "lottery_by_letter"]
    return float(np.mean([oc[a][k]["min"] - oc[b][k]["min"] for k in keys])) if keys else None


def analyse(recs, n_boot=S1.N_BOOT):
    old = S1.MODELS
    S1.MODELS = ("Q9", "Q27", KEY, "L70")
    try:
        res = S1.analyse(recs, n_boot=n_boot)
    finally:
        S1.MODELS = old
    M = res["models"]
    head = S1.headline({"L70": M.get(KEY, {}), "Q9": M.get("Q9", {}), "Q27": M.get("Q27", {})})
    oc = M1.order_check([r for r in recs if r["cell"].startswith(KEY + "-")])
    order = {"cells": oc}
    for tag, (a, b) in (("d0", ("W", "Z")), ("d2", ("W2", "Z2")), ("risk", ("RISKY", "SAFE"))):
        order[f"ours_min_{tag}_s1"] = _min_spread(oc, f"{KEY}-{a}_s1", f"{KEY}-{b}_s1")
    return {"headline": head.replace("at 70B", "at clean 70B"), "LF": M.get(KEY), "ladder": {m: M.get(m) for m in
                                                                                           ("Q9", "Q27")},
            "qlora_L70": M.get("L70"), "levels": res["levels"].get(KEY), "order": order}


def summary_md(res):
    f = S1._f
    L = ["# SC-M13 summary (clean bf16 Llama-3.1-70B)", "", f"**headline:** {res['headline']}", ""]
    for name in ("LF", "qlora_L70"):
        r = res.get(name) or {}
        L.append(f"## {name}")
        for dose in ("d0", "d2"):
            if dose in r:
                x = r[dose]
                L.append(f"- {dose} habit: act {f(x['act'])} (took {x['took']}); ours {f(x['ours'])} → {x['ours_label']}; "
                         f"bjust {f(x['bjust'])} → {x['bjust_label']}; predict {f(x['predict'])}")
        if "risk" in r:
            x = r["risk"]
            L.append(f"- risk: lottery {f(x['lottery'])} (took {x['took']}); brisk {f(x['brisk'])} → {x['label']}")
        if "words" in r:
            L.append(f"- words: self_mc {f(r['words']['self_mc'])} (took {r['words']['took']})")
        L.append("")
    o = res["order"]
    L.append("- order check, ours on min (s1): " + ", ".join(f"{k} {v:+.1f}" for k, v in o.items()
                                                           if k.startswith("ours_min") and v is not None))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m13 analyze")
    ap.add_argument("--res", nargs="+", default=[str(H / "results" / "main")])
    a = ap.parse_args(argv)
    recs = S1.load(a.res + [str(S1_RES)])
    res = analyse(recs)
    Path(a.res[0], "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    Path(a.res[0], "summary.md").write_text(summary_md(res))
    print(summary_md(res))


if __name__ == "__main__":
    main()
