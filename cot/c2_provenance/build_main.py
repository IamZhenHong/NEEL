"""Pre-pass -> the frozen item set: filter (Qwen >= 4/5 clean samples correct), pick one correct trace per author, cut and
corrupt it by rule, run the checks, compute the gate quantities, and write data/items_main.json(.gz) + results/gate.json
+ results/build_examples.txt (concluding step before/after, for reading).

  uv run python -m cot.c2_provenance.build_main [--prepass-dir cot/c2_provenance/results]
  uv run python -m cot.c2_provenance.build_main --fake --limit 3 --out-dir <dir>     # synthetic pre-pass (tests)

Trace choice per item and author: correct samples (closed thinking, final answer = gold) in sample order; the first
whose thinking has a concluding step (make_pair) is the first-choice trace, and the gate's parse rate is counted over
first-choice traces. If a first-choice pair fails a check, later correct samples are tried; an item with no passing
pair for both authors is dropped. The first 100 kept items per dataset, in candidate order, form the set.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from cot.c2_provenance.common import (DATA, HERE, RESULTS, check_pair, load_jsonl, make_pair, parse_final, same_answer,
                                      split_gemma, split_qwen)

N_PER_DS = 100
CONFIG = {"chunk_items": 8, "max_tokens_T": 3072, "max_tokens_U": 4096, "max_tokens_m5": 1536, "n_m5": 100}
SEED = 20261015


def fake_prepass(cands):
    """Synthetic raw generations built from the reference solutions (tests only)."""
    q, g = [], []
    for it in cands:
        if it["ds"] == "gsm8k":
            body = re.sub(r"<<[^>]*>>", "", it["solution"])
            th = f"Let me work through this step by step.\nThe question gives the quantities we need.\n{body}\nSo the answer is {it['gold']}."
            ans = f"Final answer: {it['gold']}"
        else:
            th = ("Let me think about each option carefully before choosing.\n"
                  + "\n".join(f"Option {L}: {c}." for L, c in zip("ABCD", it["choices"]))
                  + f"\nWeighing these, the answer is {it['gold']}.")
            ans = f"Final answer: {it['gold']}"
        for s in range(5):
            q.append({"id": it["id"], "ds": it["ds"], "sample": s, "text": f"{th}\n</think>\n\n{ans}", "finish": "stop",
                      "n_tok": 100})
        for s in range(3):
            g.append({"id": it["id"], "ds": it["ds"], "sample": s, "text": f"<|channel>thought\n{th}<channel|>{ans}<turn|>",
                      "finish": "stop", "n_tok": 80})
    return q, g


def score(rows, split, byid):
    out = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["id"], r["sample"])):
        if r["id"] not in byid:        # pre-pass rows for candidates outside the current candidate file
            continue
        it = byid[r["id"]]
        th, ans, closed = split(r["text"])
        fin = parse_final(it["ds"], ans) if closed else None
        out[r["id"]].append({"sample": r["sample"], "thinking": th, "closed": closed, "final": fin,
                             "correct": closed and same_answer(it["ds"], fin, it["gold"]), "n_tok": r["n_tok"],
                             "finish": r["finish"]})
    return out


def pick(it, samples):
    """-> (first_choice_info, chosen_pair or None, reasons Counter)"""
    reasons, first, chosen = Counter(), None, None
    for s in samples:
        if not s["correct"]:
            continue
        pair, why = make_pair(it, s["thinking"])
        if pair is None:
            reasons[why] += 1
            continue
        ck = check_pair(it, pair)
        if first is None:
            first = {"sample": s["sample"], "checks": ck}
        if ck["ok"]:
            chosen = {**pair, "sample": s["sample"], "checks": ck}
            break
        reasons["check:" + ",".join(k for k, v in ck.items() if not v and k != "ok")] += 1
    return first, chosen, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepass-dir", default=str(RESULTS))
    ap.add_argument("--out-dir", default=None, help="default: data/ and results/ of this folder")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated candidate ids (smoke)")
    a = ap.parse_args()
    cands = json.loads((DATA / "candidates.json").read_text())
    if a.ids:
        keep = set(a.ids.split(","))
        cands = [c for c in cands if c["id"] in keep]
    if a.limit:
        cands = [c for c in cands if c["ds"] == "gsm8k"][:a.limit] + [c for c in cands if c["ds"] == "mmlu"][:a.limit]
    byid = {c["id"]: c for c in cands}
    if a.fake:
        qrows, grows = fake_prepass(cands)
    else:
        pd = Path(a.prepass_dir)
        qrows, grows = load_jsonl(pd / "prepass_qwen.jsonl"), load_jsonl(pd / "prepass_gemma.jsonl")
    ddir = Path(a.out_dir) if a.out_dir else DATA
    rdir = Path(a.out_dir) if a.out_dir else RESULTS
    ddir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    qs, gs = score(qrows, split_qwen, byid), score(grows, split_gemma, byid)

    gate = {"by_ds": {}, "reasons": {}}
    kept, first_checks, ex = [], [], []
    for ds in ("gsm8k", "mmlu"):
        dc = [c for c in cands if c["ds"] == ds]
        n_pass, acc_kept, chosen_items = 0, [], []
        rs = Counter()
        for it in dc:
            q = qs.get(it["id"], [])
            nc = sum(s["correct"] for s in q)
            if len(q) < 5 or nc < 4:
                rs["qwen_below_4of5" if len(q) >= 5 else "qwen_missing"] += 1
                continue
            n_pass += 1
            acc_kept += [s["correct"] for s in q]
            fq, cq, rq = pick(it, q)
            fg, cg, rg = pick(it, gs.get(it["id"], []))
            for f, author in ((fq, "qwen"), (fg, "gemma")):
                if f:
                    first_checks.append({"id": it["id"], "ds": ds, "author": author, **f["checks"]})
            rs.update({f"qwen:{k}": v for k, v in rq.items()})
            rs.update({f"gemma:{k}": v for k, v in rg.items()})
            if cq is None or cg is None:
                rs["dropped_no_pair:" + ("qwen" if cq is None else "") + ("gemma" if cg is None else "")] += 1
                continue
            if len(chosen_items) < N_PER_DS:
                chosen_items.append({**it, "wrong": cq["wrong"],
                                     "traces": {"qwen": {"right": cq["right"], "corrupt": cq["corrupt"]},
                                                "gemma": {"right": cg["right"], "corrupt": cg["corrupt"]}},
                                     "trace_meta": {au: {"sample": c["sample"], "cut_chars": c["cut_chars"],
                                                         "full_chars": c["full_chars"]}
                                                    for au, c in (("qwen", cq), ("gemma", cg))}})
                for au, c in (("qwen", cq), ("gemma", cg)):
                    ex.append(f"== {it['id']} [{au}] gold={it['gold']} wrong={c['wrong']} cut {c['cut_chars']}/"
                              f"{c['full_chars']} chars\n  RIGHT  : {c['last_right'][-300:]}\n  CORRUPT: {c['last_corrupt'][-300:]}")
        q_all = [s["correct"] for it in dc for s in qs.get(it["id"], [])]
        g_all = [s["correct"] for it in dc for s in gs.get(it["id"], [])]
        gate["by_ds"][ds] = {"candidates": len(dc), "qwen_pass_4of5": n_pass,
                             "qwen_clean_acc_all": round(sum(q_all) / max(len(q_all), 1), 4),
                             "qwen_clean_acc_kept_4of5": round(sum(acc_kept) / max(len(acc_kept), 1), 4),
                             "gemma_clean_acc_all": round(sum(g_all) / max(len(g_all), 1), 4),
                             "items_with_both_pairs_in_order": None, "kept": len(chosen_items)}
        gate["reasons"][ds] = dict(rs)
        kept += chosen_items
    # clean accuracy on the final kept items (pooled over their 5 clean samples)
    acc = [s["correct"] for it in kept for s in qs[it["id"]]]
    gate["clean_acc_final_items"] = round(sum(acc) / max(len(acc), 1), 4)
    n_fc = len(first_checks)
    gate["first_choice_traces"] = n_fc
    gate["corrupt_parse_rate_first_choice"] = round(sum(c["corrupt_parses_wrong"] for c in first_checks) / max(n_fc, 1), 4)
    gate["first_choice_all_checks_rate"] = round(sum(c["ok"] for c in first_checks) / max(n_fc, 1), 4)
    gate["first_choice_check_fail_counts"] = {k: sum(not c[k] for c in first_checks) for k in first_checks[0]
                                              if k not in ("id", "ds", "author")} if first_checks else {}
    gate["gate_clean_acc_ge_85"] = gate["clean_acc_final_items"] >= 0.85
    gate["gate_parse_ge_95"] = gate["corrupt_parse_rate_first_choice"] >= 0.95
    gate["gate_pass"] = gate["gate_clean_acc_ge_85"] and gate["gate_parse_ge_95"]
    # trace length summaries
    for au in ("qwen", "gemma"):
        cc = sorted(it["trace_meta"][au]["cut_chars"] for it in kept)
        ff = sorted(it["trace_meta"][au]["full_chars"] for it in kept)
        if cc:
            gate[f"{au}_trace_chars"] = {"cut_median": cc[len(cc) // 2], "full_median": ff[len(ff) // 2],
                                         "cut_max": cc[-1]}
    tq = [s["n_tok"] for v in qs.values() for s in v]
    gate["qwen_prepass_tokens_mean"] = round(sum(tq) / max(len(tq), 1))
    gate["qwen_prepass_at_cap"] = sum(s["finish"] == "length" for v in qs.values() for s in v)

    # M5 cells: n_m5 items (half per dataset), one U-corrupted cell each, 25 per author x label cell
    rng = random.Random(SEED)
    cells = [("qwen", "self"), ("qwen", "other"), ("gemma", "self"), ("gemma", "other")]
    for ds in ("gsm8k", "mmlu"):
        ids = [it["id"] for it in kept if it["ds"] == ds]
        rng.shuffle(ids)
        for j, i in enumerate(ids[:CONFIG["n_m5"] // 2]):
            au, lab = cells[j % 4]
            next(it for it in kept if it["id"] == i)["m5"] = {"author": au, "label": lab}
    # chunk order: datasets interleaved
    g = [it for it in kept if it["ds"] == "gsm8k"]
    m = [it for it in kept if it["ds"] == "mmlu"]
    order = [x for pair in zip(g, m) for x in pair] + g[len(m):] + m[len(g):]
    d = {"config": CONFIG, "items": order}
    raw = json.dumps(d, indent=1).encode()
    (ddir / "items_main.json").write_bytes(raw)
    with gzip.GzipFile(ddir / "items_main.json.gz", "wb", mtime=0) as f:
        f.write(raw)
    gate["items_sha256"] = hashlib.sha256(raw).hexdigest()
    gate["n_items"] = len(order)
    (rdir / "gate.json").write_text(json.dumps(gate, indent=1))
    (rdir / "build_examples.txt").write_text("\n".join(ex))
    print(json.dumps({k: v for k, v in gate.items() if k != "reasons"}, indent=1))
    print("reasons:", json.dumps(gate["reasons"], indent=1))


if __name__ == "__main__":
    main()
