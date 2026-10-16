"""Inputs for the second pod (label-based ablations, README "e71 / e73 ablation"), built locally once the labels and
categories exist: neurons/data/abl_inputs.json.

e71: up to 24 kept episodes per condition (token ids from gen/gen_e71); authority set = e71 differential top-20
     neurons categorised `authority`; limit set = neurons categorised `limit` among the e71 differential top-20 and the
     e71 frequent lists. 20 random layer-matched sets per non-empty set (excluding the e71 differential top-20).
e73: the 20 items under theatrical; register set / safety set = e73 differential top-20 neurons categorised
     `register` / `safety`; 5 random layer-matched sets each.
Mean-ablation values: the corpus means (results/main/corpus_stats_<kind>.pt).
  uv run python -m neurons.build_abl"""
import gzip
import json
import os
import random

import torch

from neurons.common import HERE, load_jsonl, parse_nid

MAIN = HERE / "results" / os.environ.get("NEURONS_RUN", "main")
D_FF = 17408


def cats():
    out = {}
    for l in open(MAIN / "categories.jsonl"):
        r = json.loads(l)
        out[(r["kind"], r["nid"])] = r["category"]
    return out


def random_sets(base, excl, n, seed):
    rng = random.Random(seed)
    sets = []
    for _ in range(n):
        draw = []
        for x in base:
            l, _ = parse_nid(x)
            while True:
                c = f"L{l}N{rng.randrange(D_FF)}"
                if c not in excl and c not in draw:
                    break
            draw.append(c)
        sets.append(draw)
    return sets


def with_vals(nids, mean, zero):
    return [[*parse_nid(x), 0.0 if zero else float(mean[parse_nid(x)])] for x in nids]


def main():
    C = cats()
    out = {}
    # e71
    s71 = json.loads((MAIN / "e71_summary.json").read_text())
    diff71 = [x["nid"] for x in s71["diff_top20"]]
    freq71 = [k for lst in s71["frequent"].values() for k, _ in lst]
    auth = [x for x in diff71 if C.get(("qwen36", x)) == "authority"]
    limit = sorted({x for x in diff71 + freq71 if C.get(("qwen36", x)) == "limit"})
    m36 = torch.load(MAIN / "corpus_stats_qwen36.pt")["mean"]
    gen = {}
    with gzip.open(MAIN / "gen" / "gen_e71.jsonl.gz", "rt") as f:
        for l in f:
            e = json.loads(l)
            gen[e["key"]] = e
    rows = load_jsonl(MAIN / "e71_rows.jsonl")
    ep = []
    for cond in ("sup", "peer"):
        rs = sorted([r for r in rows if r["cond"] == cond], key=lambda r: r["tid"])[:24]
        for r in rs:
            g = gen[r["tid"]]
            ep.append({"tid": r["tid"], "cond": cond, "ids": g["prompt_ids"], "t": len(g["prompt_ids"]) - 1})
    # E1 (exploratory, D1 run): supervisor prompts refused under greedy decoding, attributed with the same L so the
    # differential set can be read with the requester fixed (sup-paid vs sup-refused) or the decision fixed
    # (sup-refused vs peer-refused)
    xr = []
    for k, g in sorted(gen.items()):
        if g["job"]["cond"] == "sup" and "<function=issue_refund>" not in g["text"] and g["out_ids"][0] != 248058:
            xr.append({"tid": k, "cond": "sup_refused", "ids": g["prompt_ids"], "t": len(g["prompt_ids"]) - 1})
    import numpy as np
    cand = np.load(MAIN / "e71_cand.npz")
    conds71 = {"none": []}
    for name, st in (("auth", auth), ("limit", limit)):
        if not st:
            continue
        conds71[f"{name}_zero"] = with_vals(st, m36, True)
        conds71[f"{name}_mean"] = with_vals(st, m36, False)
        for k, rs in enumerate(random_sets(st, set(diff71), 20, 7100 + len(name))):
            conds71[f"{name}_rand{k}"] = with_vals(rs, m36, True)
    out["e71"] = {"episodes": ep, "conds": conds71, "auth": auth, "limit": limit, "sup_refused": xr,
                  "cand_flat": [int(x) for x in cand["flat"]],
                  "pay_tok": s71["pay_tok"], "ref_tok": s71["ref_tok"]}
    # e73
    p73 = MAIN / "e73_summary.json"
    if p73.exists():
        s73 = json.loads(p73.read_text())
        diff73 = [x["nid"] for x in s73["diff_top20"]]
        reg = [x for x in diff73 if C.get(("qwen14", x)) == "register"]
        safe = [x for x in diff73 if C.get(("qwen14", x)) == "safety"]
        m14 = torch.load(MAIN / "corpus_stats_qwen14.pt")["mean"]
        items = load_jsonl(HERE / "data" / "e73.jsonl")
        conds73 = {"none": []}
        for name, st in (("register", reg), ("safety", safe)):
            if not st:
                continue
            conds73[f"{name}_zero"] = with_vals(st, m14, True)
            conds73[f"{name}_mean"] = with_vals(st, m14, False)
            for k, rs in enumerate(random_sets(st, set(diff73), 5, 7300 + len(name))):
                conds73[f"{name}_rand{k}"] = with_vals(rs, m14, True)
        out["e73"] = {"items": [{"item": it["item"], "prompt": it["prompt"], "ids": it["ids"]["theatrical"]}
                                for it in items], "conds": conds73, "register": reg, "safety": safe,
                      "openers": json.loads((HERE / "data" / "e73_meta.json").read_text())["openers"]}
    (HERE / "data" / "abl_inputs.json").write_text(json.dumps(out))
    print(f"e71: {len(ep)} episodes; authority set {auth}; limit set {limit}; {len(conds71)} conditions")
    if "e73" in out:
        print(f"e73: register set {out['e73']['register']}; safety set {out['e73']['safety']}; "
              f"{len(out['e73']['conds'])} conditions")


if __name__ == "__main__":
    main()
