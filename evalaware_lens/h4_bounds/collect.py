"""EL-H4: equivalence bounds on the programme's null results, read from each mini's frozen analysis files ($0).

For every null claim: the estimate, its 95% interval as the mini reported it, and the bound B = max(|lo|, |hi|):
"changed by at most B (95% interval)". Equivalent at margin M means the whole interval lies inside +-M; M is the mini's
own frozen null band (readouts +-0.2 SD in L2/L3, +-0.25 in L1; rates +-10 points; S3/C3/S2 their own bars).

    uv run python -m evalaware_lens.h4_bounds.collect      # -> h4_bounds/bounds.json, prints the tables
"""
import json
from pathlib import Path

import numpy as np

EL = Path(__file__).resolve().parents[1]
H = Path(__file__).resolve().parent


def bound(c):
    return None if c.get("lo") is None else round(max(abs(c["lo"]), abs(c["hi"])), 3)


def l2l3():
    """Fixed-reply readouts: per model, the largest bound over usable directions and the labelled cue contrasts."""
    out = {}
    for mini, f, cons in (("L2", "analysis_l2.json", ("safety_vs_neutral", "capability_vs_neutral",
                                                        "safety_vs_capability", "safety_vs_manip",
                                                        "capability_vs_manip")),
                          ("L3", "analysis_l3.json", ("observed", "scored", "observed_scored"))):
        d = json.load(open(EL / "l2l3_cuereads/results" / f))
        for m, v in d["models"].items():
            rows = {}
            for dname, dv in v["directions"].items():
                rows[dname] = {c: {**{k: dv[c]["fixed"][k] for k in ("est", "lo", "hi", "floor")},
                                   "bound": bound(dv[c]["fixed"])} for c in cons}
            real = [r["bound"] for dn, r2 in rows.items() if dn != "french" for r in r2.values()]
            out[f"{mini}|{m}"] = {"max_bound_sd": max(real), "median_bound_sd": round(float(np.median(real)), 3),
                                  "margin": 0.2, "all_within_margin": max(real) < 0.2,
                                  "french_max_bound": max(r["bound"] for r in rows["french"].values()), "rows": rows}
            if mini == "L2":
                reg = v["registered"]
                out[f"{mini}|{m}"]["p_eval"] = {k: reg[k]["p_eval"] for k in reg}
    return out


def l1():
    """Per model: end-of-reasoning natural read and the cue eval - manip summary read, per instrument."""
    d = json.load(open(EL / "l1_fingerprint/results/main/analysis.json"))
    out = {}
    for m, v in d["models"].items():
        rows = {}
        for name, iv in v["instruments"].items():
            rows[name] = {k: {**{x: iv[k].get(x) for x in ("est", "lo", "hi", "floor")}, "bound": bound(iv[k])}
                          for k in ("natural_summary", "cue_summary_eval-manip", "cue_prompt_end_eval-manip")
                          if isinstance(iv.get(k), dict)}
        for k in ("natural_summary", "cue_summary_eval-manip"):
            b = [r[k]["bound"] for n, r in rows.items() if k in r and n not in ("french",) and r[k]["bound"] is not None]
            out[f"L1|{m}|{k}"] = {"max_bound_sd": max(b), "median_bound_sd": round(float(np.median(b)), 3),
                                  "margin": 0.25}
        out[f"L1|{m}|rows"] = rows
    return out


def c3():
    d = json.load(open(EL / "s2_evalself/results/main/c3_analysis.json"))
    out = {}
    for pair, v in d["pairs"].items():
        act, rep = v["act"]["deploy"], v["report"]["deploy"] if isinstance(v["report"], dict) else None
        out[f"C3|{pair}"] = {"act_eval_vs_deploy": {**{k: act[k] for k in ("est", "lo", "hi")}, "bound": bound(act),
                                                    "trained_gap": act["gap_neutral"]},
                             "report_eval_vs_deploy": ({**{k: rep[k] for k in ("est", "lo", "hi")}, "bound": bound(rep)}
                                                       if rep and "lo" in rep else rep),
                             "probe": v.get("probe"), "verdict": v["verdict"]}
    return out


def w2():
    d = json.load(open(EL / "w2_setbacks/results/analysis.json"))
    return {"W2|contrasts": d["contrasts"], "W2|rates": d["rates"]}


def s3():
    b = json.load(open(EL / "s3_mask/results/stageB/analysis.json"))
    c = json.load(open(EL / "s3_mask/results/stageC_analysis/analysis.json"))
    return {"S3|B": b["contrasts"], "S3|C": c["contrasts"]}


def main():
    out = {**l2l3(), **l1(), **c3(), **w2(), **s3()}
    (H / "bounds.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if isinstance(v, dict) and "max_bound_sd" in v:
            print(f"{k}: max {v['max_bound_sd']} median {v['median_bound_sd']} margin {v['margin']} "
                  f"french {v.get('french_max_bound')}")
    for k, v in out.items():
        if k.startswith("C3|"):
            print(k, v["act_eval_vs_deploy"], v["report_eval_vs_deploy"], v["verdict"])


if __name__ == "__main__":
    main()
