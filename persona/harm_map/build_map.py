"""Harm vulnerability map v0: recompute every cell from the raw jsonl named in cells/*.json.

Descriptive (see DESIGN.md). Each entry maps a harm x model onto base / instr / instr_rule / rec / rec_rule with
python expressions over the raw rows; this script evaluates them, bootstraps task-clustered CIs and writes
results/map.json + results/map.md.
"""
import glob
import gzip
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ["base", "instr", "instr_rule", "rec", "rec_rule", "rec_opp"]
N_BOOT = 10_000
RNG = np.random.default_rng(0)


def load_rows(files):
    rows = []
    for f in files:
        with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def compile_expr(src):
    return eval("lambda r: " + src) if src else None


def cell_stats(rows, select, valid, harmful, cluster):
    """Rate (%) with a task-clustered bootstrap 95% CI; returns None if the cell is absent."""
    by_task = {}
    for r in rows:
        try:
            if not select(r) or not valid(r):
                continue
            by_task.setdefault(cluster(r), []).append(1.0 if harmful(r) else 0.0)
        except (KeyError, TypeError):
            continue
    if not by_task:
        return None
    keys = sorted(by_task, key=str)
    tasks = [by_task[k] for k in keys]
    hits = np.array([sum(t) for t in tasks])
    ns = np.array([len(t) for t in tasks])
    rate = 100 * hits.sum() / ns.sum()
    idx = RNG.integers(0, len(tasks), size=(N_BOOT, len(tasks)))
    boot = 100 * hits[idx].sum(1) / ns[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": int(ns.sum()), "tasks": len(tasks), "rate": round(rate, 1), "lo": round(lo, 1), "hi": round(hi, 1),
            "_hits": hits, "_ns": ns, "_keys": keys}


def diff_ci(a, b):
    """a - b with a task-clustered bootstrap: paired over tasks when both cells hold the same tasks, else
    each cell resampled on its own."""
    if a["_keys"] == b["_keys"]:
        idx = RNG.integers(0, len(a["_ns"]), size=(N_BOOT, len(a["_ns"])))
        d = (100 * a["_hits"][idx].sum(1) / a["_ns"][idx].sum(1)
             - 100 * b["_hits"][idx].sum(1) / b["_ns"][idx].sum(1))
    else:
        def boot(c):
            i = RNG.integers(0, len(c["_ns"]), size=(N_BOOT, len(c["_ns"])))
            return 100 * c["_hits"][i].sum(1) / c["_ns"][i].sum(1)
        d = boot(a) - boot(b)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return round(a["rate"] - b["rate"], 1), round(lo, 1), round(hi, 1)


def main():
    entries = []
    for path in sorted(glob.glob(os.path.join(HERE, "cells", "*.json"))):
        with open(path) as fh:
            for e in json.load(fh):
                e["_source"] = os.path.basename(path)
                entries.append(e)

    out = []
    for e in entries:
        if e.get("status") or not e.get("files"):
            out.append({k: e.get(k) for k in ("harm", "model", "harness", "status", "notes", "_source")})
            continue
        rows = load_rows(e["files"])
        valid = compile_expr(e["valid"]) or (lambda r: True)
        harmful = compile_expr(e["harmful"])
        cluster = compile_expr(e["cluster"])
        stats = {}
        for c in CELLS:
            sel = compile_expr((e.get("cells") or {}).get(c))
            stats[c] = cell_stats(rows, sel, valid, harmful, cluster) if sel else None
        rec = {k: e.get(k) for k in ("harm", "harm_label", "model", "harness", "rule_variant", "k_record",
                                     "record_form", "rule_kind", "same_run", "_source")}
        rec["cells"] = {c: ({k: v for k, v in s.items() if not k.startswith("_")} if s else None)
                        for c, s in stats.items()}
        if stats["instr"]:
            rec["prior_strength"] = round(100 - stats["instr"]["rate"], 1)
        if stats["rec"] and stats["base"]:
            rec["rec_minus_base"] = diff_ci(stats["rec"], stats["base"])
        if stats["rec_rule"] and stats["instr_rule"]:
            rec["daylight"] = diff_ci(stats["rec_rule"], stats["instr_rule"])
        check = e.get("check") or {}
        rec["check_mismatch"] = {c: (check[c], rec["cells"][c]["rate"]) for c in check
                                 if rec["cells"].get(c) and isinstance(check[c], list)
                                 and abs(check[c][1] - rec["cells"][c]["rate"]) > 0.5}
        out.append(rec)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "map.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{len(out)} entries -> results/map.json")


if __name__ == "__main__":
    main()
