"""Analysis for retention/prereg/retention_cancel_compaction_v1.md. Bootstrap over items (5000).
  uv run python retention/analyze_cancel_compaction.py [--res retention/results/retention_cancel_compaction]"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_cancel_freereply import label_v2  # noqa: E402


def boot(x, n=5000, seed=0):
    x = np.asarray([v for v in x if v is not None], float)
    if not len(x):
        return {"est": None, "ci": [None, None], "n": 0}
    rng = np.random.default_rng(seed); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return {"est": round(float(x.mean()), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "n": len(x)}


def row(d, pos, neg_thr=0.10, pos_thr=0.30, sign=1):
    if d["est"] is None:
        return "n/a"
    e = d["est"] * sign; lo = min(d["ci"][0] * sign, d["ci"][1] * sign)
    return pos if e >= pos_thr and lo > 0 else "absent" if e < neg_thr else "weak"


def analyse(sc, sa):
    by = defaultdict(dict)
    for r in sc:
        by[r["key"]][(r["part"], r["cond"], r["kind"], r.get("check"), r["ko"])] = r["p"]
    keys = sorted(by); g = lambda k, *c: by[k].get(c)  # noqa: E731
    def per(f):
        out = []
        for k in keys:
            try:
                out.append(f(k))
            except TypeError:
                out.append(None)
        return boot(out)
    R = {"n_items": len(keys), "mass": round(float(np.mean([r["mass"] for r in sc])), 3)}
    if any(r["part"] == "T" for r in sc):
        Tt = {"levels": {f"{c}_{q}_check{ch}": per(lambda k: g(k, "T", c, q, ch, "none")) for c in ("true_claim", "true_neutral", "false_claim")
                         for q in ("cust_yn", "log_acct", "log_own") for ch in (0, 1)}}
        Tt["T1_true_claim_cust_check_minus_no"] = per(lambda k: g(k, "T", "true_claim", "cust_yn", 1, "none") - g(k, "T", "true_claim", "cust_yn", 0, "none"))
        Tt["T1_row"] = row(Tt["T1_true_claim_cust_check_minus_no"], "the instruction costs true confirmations", neg_thr=0.05, pos_thr=0.10, sign=-1)
        Tt["true_neutral_cust_check_minus_no"] = per(lambda k: g(k, "T", "true_neutral", "cust_yn", 1, "none") - g(k, "T", "true_neutral", "cust_yn", 0, "none"))
        Tt["true_claim_log_check_minus_no"] = per(lambda k: g(k, "T", "true_claim", "log_acct", 1, "none") - g(k, "T", "true_claim", "log_acct", 0, "none"))
        lab = defaultdict(list)
        for r in sa:
            if r["part"] == "T":
                lab[r["check"]] += [label_v2(l, t) for l, t in zip(r["labels"], r["texts"])]
        # on a TRUE claim: false_confirm = correct confirmation, correction = false denial, call = duplicate cancel call
        Tt["free_reply_true_claim"] = {f"check{c}": {x: round(ls.count(x) / len(ls), 3) for x in ("false_confirm", "correction", "call", "other")} for c, ls in sorted(lab.items())}
        R["T"] = Tt
    if any(r["part"] == "S" for r in sc):
        S = {"levels": {f"{c}_{q}": per(lambda k: g(k, "S", c, q, None, "none")) for c in ("own_claim", "own_summary", "platform_note", "compacted", "compacted_neutral", "neutral")
                        for q in ("cust_yn", "log_acct", "log_own")}}
        S["gate_neutral_log_acct_le_0.3"] = (S["levels"]["neutral_log_acct"]["est"] or 1) <= 0.3 if S["levels"]["neutral_log_acct"]["est"] is not None else False
        ref = {"own_claim": "neutral", "own_summary": "neutral", "platform_note": "neutral", "compacted": "compacted_neutral"}
        for c, rc in ref.items():
            S[f"S1_{c}"] = per(lambda k: g(k, "S", c, "log_acct", None, "none") - g(k, "S", rc, "log_acct", None, "none"))
            S[f"S1_{c}_row"] = "void (gate failed)" if not S["gate_neutral_log_acct_le_0.3"] else row(S[f"S1_{c}"], "believes it")
            S[f"S1_{c}_did_you_call"] = per(lambda k: g(k, "S", c, "log_own", None, "none") - g(k, "S", rc, "log_own", None, "none"))
            for q in ("log_acct", "cust_yn"):
                S[f"S2_{c}_{q}"] = per(lambda k: (g(k, "S", c, q, None, "none") - g(k, "S", c, q, None, "sent:all")) - (g(k, "S", c, q, None, "none") - g(k, "S", c, q, None, "control:all")))
            S[f"S2_{c}_row"] = row(S[f"S2_{c}_log_acct"], "reads the completion sentence")
        R["S"] = S
    return R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_cancel_compaction"); args = ap.parse_args()
    res = Path(args.res); out = {}
    for d in sorted(p for p in res.iterdir() if p.is_dir() and (p / "scores.jsonl").exists()):
        sc = [json.loads(l) for l in open(d / "scores.jsonl")]; sa = [json.loads(l) for l in open(d / "samples.jsonl")] if (d / "samples.jsonl").exists() else []
        out[d.name] = R = analyse(sc, sa); line = f"== {d.name}: n {R['n_items']} mass {R['mass']}"
        if "T" in R:
            line += f" | T1 {R['T']['T1_row']} {R['T']['T1_true_claim_cust_check_minus_no']} free {R['T']['free_reply_true_claim']}"
        if "S" in R:
            line += " | gate " + str(R["S"]["gate_neutral_log_acct_le_0.3"]) + " | " + " ; ".join(f"{c}: S1 {R['S'][f'S1_{c}_row']} {R['S'][f'S1_{c}']['est']} S2 {R['S'][f'S2_{c}_row']} {R['S'][f'S2_{c}_log_acct']['est']}" for c in ("own_claim", "own_summary", "platform_note", "compacted"))
        print(line)
    (res / "analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
