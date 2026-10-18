"""A1 analysis: per-cell record accuracy (conversation-level mean over samples, bootstrap over conversations, 10,000
resamples, seed 0), paired contrasts over false-claim conversations, tool-call counts, and the gate summaries.
Usage (repo root): python3 minis/a1_source_channel/analyze.py  -> prints tables, writes results/analysis.json
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
CELLS = ["SR", "ST", "UR", "UT", "D96"]
CONTRASTS = {"format @ system (ST-SR)": (("ST", 1), ("SR", -1)), "format @ user (UT-UR)": (("UT", 1), ("UR", -1)),
             "source @ tool (ST-UT)": (("ST", 1), ("UT", -1)), "source @ report (SR-UR)": (("SR", 1), ("UR", -1)),
             "interaction (ST-SR)-(UT-UR)": (("ST", 1), ("SR", -1), ("UT", -1), ("UR", 1)),
             "run 96 gap (ST-D96)": (("ST", 1), ("D96", -1)), "wording (UR-D96)": (("UR", 1), ("D96", -1))}


def load(name):
    p = R / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def conv_means(rows, cell, truth, key="tt"):
    d = defaultdict(list)
    for r in rows:
        if r["cell"] == cell and r["truth_done"] == truth and key in r:
            d[r["id"]].append(r[key])
    return {k: float(np.mean(v)) for k, v in d.items()}


def boot(ids, f, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    ids = sorted(ids)
    est = f(ids)
    bs = [f([ids[i] for i in rng.integers(0, len(ids), len(ids))]) for _ in range(n)]
    return {"mean": round(est, 3), "lo": round(float(np.percentile(bs, 2.5)), 3),
            "hi": round(float(np.percentile(bs, 97.5)), 3), "n_conv": len(ids)}


def cell_table(rows, key="tt"):
    out = {}
    for truth in (False, True):
        for c in CELLS:
            m = conv_means(rows, c, truth, key)
            if m:
                out[f"{c}|{'true' if truth else 'false'}"] = boot(m, lambda ids, m=m: float(np.mean([m[i] for i in ids])))
    return out


def contrasts(rows):
    out = {}
    ms = {c: conv_means(rows, c, False) for c in CELLS}
    for name, terms in CONTRASTS.items():
        ids = set.intersection(*[set(ms[c]) for c, _ in terms])
        if ids:
            out[name] = boot(ids, lambda ii, t=terms: float(np.mean([sum(w * ms[c][i] for c, w in t) for i in ii])))
    return out


def calls(rows):
    out = {}
    for truth in (False, True):
        for c in CELLS:
            rr = [r for r in rows if r["cell"] == c and r["truth_done"] == truth]
            if rr:
                out[f"{c}|{'true' if truth else 'false'}"] = dict(Counter(
                    (r.get("decision") or r.get("answer")) + (" [+" + ",".join(r["calls"][1:]) + "]" if len(r["calls"]) > 1 else "")
                    for r in rr))
    return out


def gate(rows):
    return {c: f"{sum(r['tt'] for r in rows if r['cell'] == c)}/{sum(r['cell'] == c for r in rows)}"
            for c in ("ST", "D96")} | {"replies": dict(Counter((r["cell"], r.get("decision") or r.get("answer")) for r in rows).items()
                                                      ).__repr__()}


def main():
    res = {"gates": {}}
    for f in ("gate_gemma", "gate_qwen", "gate_noreason_gemma", "gate_noreason_qwen"):
        rows = load(f + ".jsonl")
        if rows:
            res["gates"][f] = gate(rows)
    for f in ("main_gemma", "main_qwen", "main_noreason_qwen"):
        rows = load(f + ".jsonl")
        if not rows:
            continue
        res[f] = {"n_rows": len(rows), "tt": cell_table(rows), "tt_verbal": cell_table(rows, "tt_verbal"),
                  "contrasts_false": contrasts(rows), "calls": calls(rows)}
    led = load("api_ledger.jsonl")
    res["spend_usd"] = round(sum(x["cost"] for x in led), 4)
    res["spend_by_label"] = {k: round(sum(x["cost"] for x in led if x["label"] == k), 4) for k in sorted({x["label"] for x in led})}
    json.dump(res, open(R / "analysis.json", "w"), indent=1)

    print("Gates (false claims, record-matching):")
    for k, v in res["gates"].items():
        print(f"  {k}: ST {v['ST']}  D96 {v['D96']}")
    for f in ("main_gemma", "main_qwen", "main_noreason_qwen"):
        if f not in res:
            continue
        print(f"\n{f}: record accuracy (tt), conversation-level mean [95% CI]")
        print("| cell | false claims | true claims | false: tt_verbal (report cells) |\n|---|---|---|---|")
        for c in CELLS:
            a, b = res[f]["tt"].get(f"{c}|false"), res[f]["tt"].get(f"{c}|true")
            v = res[f]["tt_verbal"].get(f"{c}|false")
            fmt = lambda x: f"{x['mean']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}]" if x else "-"
            print(f"| {c} | {fmt(a)} | {fmt(b)} | {fmt(v)} |")
        print("\nPaired contrasts, false claims:")
        for k, x in res[f]["contrasts_false"].items():
            print(f"  {k}: {x['mean']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}] (n={x['n_conv']})")
        print("\nReplies (first relevant call / answer):")
        for k, x in res[f]["calls"].items():
            print(f"  {k}: {x}")
    print(f"\nSpend: ${res['spend_usd']} {res['spend_by_label']}")


if __name__ == "__main__":
    main()
