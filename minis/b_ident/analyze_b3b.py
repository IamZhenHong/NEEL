"""B3b analysis (amendment 1). Usage: python3 minis/b_ident/analyze_b3b.py -> results/gpu_b3b/analysis.json + stdout."""
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parent / "results" / "gpu_b3b"
spec = json.load(open(R / "spec.json"))
rows = [json.loads(l) for l in open(R / "b3b.jsonl")]


def boot(d):
    rng = random.Random(0)
    bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(10000))
    return [round(bs[249], 3), round(bs[9749], 3)]


out = {"probe_acc": spec["probe_acc"], "specificity": {}}
for fr in ("none", "portray", "identify"):
    ss = [s for s in spec["spec"] if s["frame"] == fr]
    out["specificity"][fr] = {"signal": round(st.mean(s["signal"] for s in ss), 3), "ci": boot([s["signal"] for s in ss]),
                              "true_judged_true": f"{sum(s['s_true'] > 0 for s in ss)}/{len(ss)}",
                              "false_judged_true": f"{sum(s['s_false'] > 0 for s in ss)}/{len(ss)}"}
CONDS = ["none", "p1", "p2", "n2", "rand"]
for ctx, dirs in (("a_identify", ["loo"]), ("b_nopersona", ["loo", "own"])):
    for d in dirs:
        sel = [r for r in rows if r["ctx"] == ctx and r["dir"] == d]
        acts = [r for r in sel if r["kind"] == "act"]
        sig = [r for r in sel if r["kind"] == "signal"]
        ctl = [r for r in sel if r["kind"] == "control"]
        res = {}
        for c in CONDS:
            a = [r for r in acts if r["cond"] == c]
            res[c] = {"p_false": round(sum(r["label"] == "false" for r in a) / len(a), 3),
                      "labels": dict(Counter(r["label"] for r in a)),
                      "signal": round(st.mean(r["signal"] for r in sig if r["cond"] == c), 3),
                      "controls": f"{sum(r['correct'] for r in ctl if r['cond'] == c)}/{sum(1 for r in ctl if r['cond'] == c)}"}
            if c != "none":
                dp = [st.mean(r["label"] == "false" for r in acts if r["ci"] == ci and r["cond"] == c) -
                      st.mean(r["label"] == "false" for r in acts if r["ci"] == ci and r["cond"] == "none") for ci in range(8)]
                ds = [st.mean(r["signal"] for r in sig if r["ci"] == ci and r["cond"] == c) -
                      st.mean(r["signal"] for r in sig if r["ci"] == ci and r["cond"] == "none") for ci in range(8)]
                res[c]["d_p_false"] = {"diff": round(st.mean(dp), 3), "ci": boot(dp)}
                res[c]["d_signal"] = {"diff": round(st.mean(ds), 3), "ci": boot(ds)}
        out[f"{ctx}|{d}"] = res
json.dump(out, open(R / "analysis.json", "w"), indent=1)
print(f"probe acc {out['probe_acc']:.3f}")
for fr, v in out["specificity"].items():
    print(f"  specificity {fr:9s} signal {v['signal']:+.2f} {v['ci']}  true>0 {v['true_judged_true']}  false>0 {v['false_judged_true']}")
for k, res in out.items():
    if "|" in k:
        print(f"== {k}")
        for c, v in res.items():
            extra = f"  dP {v['d_p_false']['diff']:+.2f} {v['d_p_false']['ci']} dSig {v['d_signal']['diff']:+.2f} {v['d_signal']['ci']}" if c != "none" else ""
            print(f"  {c:5s} P(false)={v['p_false']:.2f} {v['labels']} signal {v['signal']:+.2f} controls {v['controls']}{extra}")
