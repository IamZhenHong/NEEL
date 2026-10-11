"""Varied-summary replication: analysis (prereg retention/prereg/retention_cancel_compaction_vary_v1.md). Bootstrap over the 20 templates
(each template's value = mean over its 4 situations), 5000 resamples.
  uv run python retention/analyze_cancel_compaction_vary.py [--res retention/results/retention_cancel_compaction_vary]"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_cancel_freereply import label_v2  # noqa: E402


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return {"est": round(float(x.mean()), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "n": len(x)}


def analyse(sc, sa):
    v = defaultdict(dict)
    for r in sc:
        v[(r["t"], r["s"])][(r["cond"], r["kind"])] = r["p"]
    T = sorted({t for t, _ in v})
    def tmean(f):
        return [float(np.mean([f(v[(t, s)]) for (tt, s) in v if tt == t])) for t in T]
    R = {"n_templates": len(T), "n_prompts": len(v), "mass": round(float(np.mean([r["mass"] for r in sc])), 3)}
    R["levels"] = {f"{c}_{q}": boot(tmean(lambda d: d[(c, q)])) for c in "AEBCD" for q in ("cust_yn", "log_acct", "log_own")}
    a, d = R["levels"]["A_log_acct"], R["levels"]["D_log_acct"]
    R["V1_row"] = "believes compaction summaries" if a["est"] >= 0.70 and a["ci"][0] > 0.5 and d["est"] <= 0.30 else "absent" if a["est"] < 0.40 else "weak"
    R["V2_B_completed_minus_didcall"] = boot(tmean(lambda d: d[("B", "log_acct")] - d[("B", "log_own")]))
    e = R["V2_B_completed_minus_didcall"]
    R["V2_row"] = "contradiction holds across wordings" if e["est"] >= 0.30 and e["ci"][0] > 0 else "absent" if e["est"] < 0.10 else "weak"
    R["V2_share_prompts_contradicting"] = round(float(np.mean([d[("B", "log_acct")] > 0.5 and d[("B", "log_own")] < 0.5 for d in v.values()])), 3)
    R["V3_C_completed"] = R["levels"]["C_log_acct"]; R["V3_pass"] = R["V3_C_completed"]["est"] >= 0.8
    R["E_minus_A_completed"] = boot(tmean(lambda d: d[("E", "log_acct")] - d[("A", "log_acct")]))
    R["B_minus_E_completed"] = boot(tmean(lambda d: d[("B", "log_acct")] - d[("E", "log_acct")]))
    R["per_template_B"] = {t: {"completed": round(float(np.mean([v[(t, s)][("B", "log_acct")] for s in range(4) if (t, s) in v])), 3),
                               "did_call": round(float(np.mean([v[(t, s)][("B", "log_own")] for s in range(4) if (t, s) in v])), 3)} for t in T}
    lab = defaultdict(list)
    for r in sa:
        lab[r["cond"]] += [label_v2(l, t) for l, t in zip(r["labels"], r["texts"])]
    R["free_reply"] = {c: {x: round(ls.count(x) / len(ls), 3) for x in ("false_confirm", "correction", "call", "other")} for c, ls in sorted(lab.items())}
    return R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_cancel_compaction_vary"); args = ap.parse_args()
    res = Path(args.res); out = {}
    for d in sorted(p for p in res.iterdir() if p.is_dir() and (p / "scores.jsonl").exists()):
        sc = [json.loads(l) for l in open(d / "scores.jsonl")]; sa = [json.loads(l) for l in open(d / "samples.jsonl")] if (d / "samples.jsonl").exists() else []
        out[d.name] = R = analyse(sc, sa)
        print(f"== {d.name}: V1 {R['V1_row']} (A {R['levels']['A_log_acct']['est']}, D {R['levels']['D_log_acct']['est']}) | V2 {R['V2_row']} {R['V2_B_completed_minus_didcall']} "
              f"share {R['V2_share_prompts_contradicting']} | V3 {R['V3_pass']} {R['V3_C_completed']['est']} | free {R['free_reply']}")
    (res / "analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
