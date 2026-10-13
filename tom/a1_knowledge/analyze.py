"""A1 analysis: curse-of-knowledge rate (missed: predicts the informed action), reverse error (saw: predicts the stale
action), check rate, by B type / presentation / cue / predictor; scenario-cluster bootstrap for the pre-registered
differences; agreement with the AI agent B's actual choices.

  uv run python -m tom.a1_knowledge.analyze [--preds preds.jsonl] [--truth truth.jsonl]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tom.a1_knowledge.render import LETTERS, order
from tom.a1_knowledge.scenarios import SCENARIOS

R = Path(__file__).parent / "results"
SID = {s["id"]: i for i, s in enumerate(SCENARIOS)}
RNG = np.random.default_rng(20261014)
B = 2000


def kind(si, letter):
    return order(si)[LETTERS.index(letter)]


def load_preds(f):
    rows = []
    for l in (R / f).read_text().splitlines():
        r = json.loads(l)
        if r["out"] is None:
            continue
        sid, info, btype, pres, cue, pred, s = r["key"].split("|")
        si = SID[sid]
        rows.append(dict(si=si, info=info, btype=btype, pres=pres, cue=cue, pred=pred,
                         kind=kind(si, r["out"]["choice"]), conf=r["out"].get("confidence")))
    return rows


def load_truth(f):
    t = collections.defaultdict(list)
    p = R / f
    if not p.exists():
        return t
    for l in p.read_text().splitlines():
        r = json.loads(l)
        if r["out"] is None:
            continue
        sid, info, model, s = r["key"].split("|")
        t[(SID[sid], info)].append(kind(SID[sid], r["out"]["choice"]))
    return t


def rate(rows, **f):
    sel = [r for r in rows if all(r[k] == v for k, v in f.items())]
    return sel


def curse(rows):
    m = [r for r in rows if r["info"] == "missed"]
    return np.mean([r["kind"] == "informed" for r in m]) if m else float("nan")


def boot_diff(rows, a, b, base):
    """difference curse(a) - curse(b) over scenarios, cluster bootstrap by scenario; a, b, base are filter dicts"""
    scen = sorted({r["si"] for r in rows})
    by = collections.defaultdict(list)
    for r in rows:
        by[r["si"]].append(r)

    def d(sample):
        ra = [r for si in sample for r in by[si] if all(r[k] == v for k, v in {**base, **a}.items())]
        rb = [r for si in sample for r in by[si] if all(r[k] == v for k, v in {**base, **b}.items())]
        return curse(ra) - curse(rb)
    est = d(scen)
    pts = [d(list(RNG.choice(scen, len(scen)))) for _ in range(B)]
    lo, hi = np.nanpercentile(pts, [2.5, 97.5])
    return est, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds.jsonl")
    ap.add_argument("--truth", default="truth.jsonl")
    ap.add_argument("--out", default="summary.md")
    a = ap.parse_args()
    rows = load_preds(a.preds)
    truth = load_truth(a.truth)
    preds = sorted({r["pred"] for r in rows})
    L = ["# A1: does a model track what another agent knows?", "", f"{len(rows)} predictions, "
         f"{len({r['si'] for r in rows})} scenarios, predictors {', '.join(preds)}", ""]
    # ground truth: AI B's own choices
    if truth:
        tm = collections.Counter(k for (si, info), ks in truth.items() if info == "missed" for k in ks)
        ts = collections.Counter(k for (si, info), ks in truth.items() if info == "saw" for k in ks)
        L += ["## The AI agent B's actual choices (gpt-5-mini + Haiku 5.5, 3 samples each)", "",
              f"- missed the update: {dict(tm)}", f"- saw the update: {dict(ts)}", ""]
    # main table
    L += ["## Predicted choice by condition (pooled predictors)", "",
          "| info | B | log | cue | n | stale | informed | check |", "|---|---|---|---|---|---|---|---|"]
    for info in ("missed", "saw"):
        for btype in ("ai", "human"):
            for pres in ("own", "narrated"):
                for cue in ("implicit", "explicit"):
                    sel = rate(rows, info=info, btype=btype, pres=pres, cue=cue)
                    if not sel:
                        continue
                    c = collections.Counter(r["kind"] for r in sel)
                    n = len(sel)
                    L.append(f"| {info} | {btype} | {pres} | {cue} | {n} | {c['stale'] / n:.2f} | "
                             f"{c['informed'] / n:.2f} | {c['check'] / n:.2f} |")
    L += ["", "## Curse-of-knowledge rate (missed: predicts the informed action) by predictor", "",
          "| predictor | all missed | implicit | explicit | AI B (impl.) | human B (impl.) | own (impl.) | narrated (impl.) | reverse error (saw, predicts stale) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in preds + ["ALL"]:
        rr = rows if p == "ALL" else [r for r in rows if r["pred"] == p]
        sw = [r for r in rr if r["info"] == "saw"]
        L.append(f"| {p} | {curse(rr):.2f} | {curse(rate(rr, cue='implicit')):.2f} | {curse(rate(rr, cue='explicit')):.2f} | "
                 f"{curse(rate(rr, cue='implicit', btype='ai')):.2f} | {curse(rate(rr, cue='implicit', btype='human')):.2f} | "
                 f"{curse(rate(rr, cue='implicit', pres='own')):.2f} | {curse(rate(rr, cue='implicit', pres='narrated')):.2f} | "
                 f"{np.mean([r['kind'] == 'stale' for r in sw]):.2f} |")
    # pre-registered differences
    L += ["", "## Pre-registered differences (scenario-cluster bootstrap 95% CI)", ""]
    tests = {
        "K1 implicit - explicit": (dict(cue="implicit"), dict(cue="explicit"), {}),
        "K2 AI B - human B (implicit)": (dict(btype="ai"), dict(btype="human"), dict(cue="implicit")),
        "K3 own - narrated (implicit)": (dict(pres="own"), dict(pres="narrated"), dict(cue="implicit")),
    }
    res = {}
    for name, (x, y, base) in tests.items():
        est, lo, hi = boot_diff(rows, x, y, base)
        res[name] = (est, lo, hi)
        per = []
        for p in preds:
            rp = [r for r in rows if r["pred"] == p]
            ra = rate(rp, **{**base, **x})
            rb = rate(rp, **{**base, **y})
            per.append(f"{p} {curse(ra) - curse(rb):+.2f}")
        L.append(f"- {name}: {est:+.3f} [{lo:+.3f}, {hi:+.3f}]; per predictor: {', '.join(per)}")
    sw = [r for r in rows if r["info"] == "saw"]
    L.append(f"- K4 reverse error in 'saw' (control, expect ≤ 0.10): {np.mean([r['kind'] == 'stale' for r in sw]):.3f}")
    # agreement with the AI agent B's modal choice, AI-type rows
    if truth:
        agree = []
        for r in rows:
            if r["btype"] != "ai":
                continue
            ks = truth.get((r["si"], r["info"]))
            if ks:
                agree.append(r["kind"] == collections.Counter(ks).most_common(1)[0][0])
        L.append(f"- Agreement with the AI agent B's modal actual choice (AI-B rows): {np.mean(agree):.3f} (n {len(agree)})")
    (R / a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
