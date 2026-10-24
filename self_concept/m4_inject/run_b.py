"""SC-M4 amendment 1 run (DESIGN.md §9): the rerun with the mean-difference direction as primary, on two seeds.

Same machinery as run.py (directions(), the block-16 injection over each prompt's user span, teacher-forced reads), on
a larger report set and fewer directions:
  report  V1c's habit question (self_mc, 16 prompts: 8 stems x 2 orders) and SC-N6b's habit level inside each of the 16
          desks' own system prompt (2 paraphrases x 2 orders x 16 desks = 64), plus N6b's trait and identity letters
          (4 + 4; secondary)
  just    the 16 justice tasks at the canonical decision token (the direction gate)
Conditions: none; dom at k in DOSES_B; dom_perp at 2; probe at 1, 2; 8 random directions at 2 (scaled like dom).
No free text (pod 1's judges agreed at kappa 0.37).

  python -m self_concept.m4_inject.run_b --cell NW_s2 --model DIR --out DIR [--tiny]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from persona_affect.v1_value_hook.fmt import prompt_ids
from self_concept.common import lens as L
from self_concept.m4_inject import items as I
from self_concept.m4_inject import run as R1
from self_concept.n6b_dose import items as N6B

DOSES_B = (1.0, 2.0, 3.0)
T0 = time.time()


def report_prompts():
    out = [{"pid": p["pid"], "family": "self_mc", "cluster": f"s{p['stem']}", "order": p["order"],
            "messages": p["messages"], "amap": p["amap"]} for p in I.self_prompts()]
    seen = set()
    for it in N6B.levels():
        desk = it["i"] % 16
        key = (it["cond"], it["tid"], desk)
        if key in seen:
            continue
        seen.add(key)
        p, o = it["tid"][1], int(it["tid"][-1])
        fam = {"habit": "habit_level", "trait": "trait", "ident": "ident"}[it["cond"]]
        cluster = f"p{p}d{desk}" if fam == "habit_level" else f"p{p}"
        out.append({"pid": f"{fam}|{it['tid']}|d{desk}" if fam == "habit_level" else f"{fam}|{it['tid']}",
                    "family": fam, "cluster": cluster, "order": o, "messages": it["messages"], "amap": it["amap"]})
    # trait / ident do not depend on the desk: keep one copy of each prompt
    uniq, out2 = set(), []
    for p in out:
        if p["pid"] in uniq:
            continue
        uniq.add(p["pid"])
        out2.append(p)
    return out2


def conds(dirs):
    out = [("none", 0.0)] + [("dom", k) for k in DOSES_B] + [("dom_perp", 2.0), ("probe", 1.0), ("probe", 2.0)]
    out += [(f"rand{j}", 2.0) for j in range(8)]
    return [(n, k) for n, k in out if n == "none" or n in dirs]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m4 run_b")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, layers, norm, head, cap = L.load_model("qwen9", tiny=a.tiny, path=None if a.tiny else a.model)
    R = L.Reader(tok, model, layers, norm, head, cap)
    layer = R1.LAYER if not a.tiny else 1
    R1.LAYER = layer
    dirs, dinfo = R1.directions(R, tok, layer, a.tiny)
    dom_gap = dirs["dom"][1]
    for j in range(8):                                      # random directions scaled like the primary (dom)
        dirs[f"rand{j}"] = (dirs[f"rand{j}"][0], dom_gap)
    letters = {x: tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B")}
    cs = conds(dirs)
    rows = []
    for p in report_prompts():
        ids = prompt_ids(tok, p["messages"], None)
        span = I.user_span(tok, ids)
        uni = next(x for x, r in p["amap"].items() if r == "unilateral")
        for n, k in cs:
            add = None if n == "none" else R1.adds(span, k * dirs[n][1] * dirs[n][0])
            _, z = R.forward(ids, [], pos=-1, add=add, logits=True)
            lp = torch.log_softmax(z, -1)
            ab = torch.log_softmax(torch.stack([lp[letters["A"]], lp[letters["B"]]]), 0)
            rows.append({"block": "self", "family": p["family"], "cluster": p["cluster"], "pid": p["pid"],
                         "order": p["order"], "dir": n, "k": k, "p_uni": float(ab[0 if uni == "A" else 1].exp()),
                         "top1_letter": int(z.argmax()) in letters.values()})
    print(f"[m4b-run {time.time() - T0:5.0f}s] {a.cell} report done {len(rows)}", flush=True)
    for p in I.just_prompts():
        p_ids = prompt_ids(tok, p["messages"], p["tools"])
        dec = I.decision(tok, p_ids, p["reply_prefix"])
        if dec is None:
            continue
        ids, first = dec
        span = I.user_span(tok, p_ids)
        for n, k in cs:
            add = None if n == "none" else R1.adds(span, k * dirs[n][1] * dirs[n][0])
            _, z = R.forward(ids, [], pos=-1, add=add, logits=True)
            lp = torch.log_softmax(z, -1)
            rows.append({"block": "just", "pid": p["pid"], "dir": n, "k": k,
                         "lo": float(lp[first["penalise"]] - lp[first["refer"]])})
    (out / f"readsb_{a.cell}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / f"infob_{a.cell}.json").write_text(json.dumps({"cell": a.cell, "layer": layer, **dinfo, "n_rows": len(rows),
                                                          "conds": cs}, indent=1))
    print(f"[m4b-run {time.time() - T0:5.0f}s] done {a.cell} {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
