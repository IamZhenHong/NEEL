"""Does the check-the-record fix survive compaction? Analysis (prereg retention/prereg/retention_cancel_compaction_fix_v1.md).
  uv run python retention/analyze_cancel_compaction_fix.py [--res retention/results/retention_cancel_compaction_fix]"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_cancel_freereply import label_v2  # noqa: E402
from analyze_cancel_compaction import boot, row  # noqa: E402


def analyse(sc, sa):
    by = defaultdict(dict)
    for r in sc:
        by[r["key"]][(r["cond"], r["check"], r["kind"])] = r["p"]
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
    R["levels"] = {f"{c}_check{ch}_{q}": per(lambda k: g(k, c, ch, q)) for c, chs in (("sum", (0, 1)), ("sum_lognone", (0, 1)), ("sum_logcall", (0, 1)), ("neutral_lognone", (0,)))
                   for ch in chs for q in ("cust_yn", "log_acct", "log_own")}
    R["X1_instruction_without_log"] = per(lambda k: g(k, "sum", 1, "log_acct") - g(k, "sum", 0, "log_acct"))
    R["X1_row"] = row(R["X1_instruction_without_log"], "the instruction works without a record", sign=-1)
    R["X2_log_with_instruction"] = per(lambda k: g(k, "sum_lognone", 1, "log_acct") - g(k, "sum", 1, "log_acct"))
    R["X2_row"] = row(R["X2_log_with_instruction"], "preserving the record restores the fix", sign=-1)
    R["X2b_log_without_instruction"] = per(lambda k: g(k, "sum_lognone", 0, "log_acct") - g(k, "sum", 0, "log_acct"))
    R["X2b_row"] = row(R["X2b_log_without_instruction"], "the record alone fixes it", sign=-1)
    R["X3_true_record_with_instruction"] = R["levels"]["sum_logcall_check1_log_acct"]
    R["X3_pass"] = (R["X3_true_record_with_instruction"]["est"] or 0) >= 0.8
    for q in ("log_own", "cust_yn"):
        R[f"X1_{q}"] = per(lambda k: g(k, "sum", 1, q) - g(k, "sum", 0, q))
        R[f"X2_{q}"] = per(lambda k: g(k, "sum_lognone", 1, q) - g(k, "sum", 1, q))
    lab = defaultdict(list)
    for r in sa:
        lab[(r["cond"], r["check"])] += [label_v2(l, t) for l, t in zip(r["labels"], r["texts"])]
    R["free_reply"] = {f"{c}_check{ch}": {x: round(ls.count(x) / len(ls), 3) for x in ("false_confirm", "correction", "call", "other")} for (c, ch), ls in sorted(lab.items())}
    return R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_cancel_compaction_fix"); args = ap.parse_args()
    res = Path(args.res); out = {}
    for d in sorted(p for p in res.iterdir() if p.is_dir() and (p / "scores.jsonl").exists()):
        sc = [json.loads(l) for l in open(d / "scores.jsonl")]; sa = [json.loads(l) for l in open(d / "samples.jsonl")] if (d / "samples.jsonl").exists() else []
        out[d.name] = R = analyse(sc, sa)
        print(f"== {d.name}: X1 {R['X1_row']} {R['X1_instruction_without_log']['est']} | X2 {R['X2_row']} {R['X2_log_with_instruction']['est']} | "
              f"X2b {R['X2b_row']} {R['X2b_log_without_instruction']['est']} | X3 {R['X3_pass']} {R['X3_true_record_with_instruction']['est']} | free {R['free_reply']}")
    (res / "analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
