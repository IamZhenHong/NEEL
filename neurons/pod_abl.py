"""Second pod: label-based ablations (README e71 / e73 "Ablation"), inputs from neurons/data/abl_inputs.json.

  python -m neurons.pod_abl --kind qwen36 --out neurons/results/abl --stages e71 spec71
  python -m neurons.pod_abl --kind qwen14 --out neurons/results/abl --stages e73 spec73

e71: per episode and condition, the logit difference at t and the payment rate (greedy + 5 samples at T 0.7, 40
     tokens; paid = `<function=issue_refund>` in the continuation), ablation at t only.
e73: per item and condition, P(refusal opener) at t and 5 sampled replies (T 0.7 / top-p 0.95 / top-k 20, <= 400
     tokens, stop at <|im_end|>) for the VB-1 HC judge (run locally)."""
import argparse
import json
import time
from pathlib import Path

import torch

from neurons.common import (HERE, MLPHooks, ablated_continuations, ablated_logits, dump_jsonl, load_model,
                            load_tokenizer, log, set_deterministic)
from neurons.pod_attr import Prog
from neurons.pod_main import stage_spec

PAY = "<function=issue_refund>"


def as_cond(c):
    return [(int(l), int(n), float(v)) for l, n, v in c]


def stage_e71(model, tok, hooks, inp, out):
    names = list(inp["conds"])
    conds = [as_cond(inp["conds"][n]) for n in names]
    res = []
    prog = Prog("e71 ablation")
    for i, e in enumerate(inp["episodes"]):
        # an episode only gets the sets meant for its condition (authority: supervisor; limit: peer) plus none
        use = [j for j, n in enumerate(names) if n == "none" or (e["cond"] == "sup" and n.startswith("auth"))
               or (e["cond"] == "peer" and n.startswith("limit"))]
        lg = ablated_logits(model, hooks, e["ids"], e["t"], [conds[j] for j in use])
        L = (lg[:, inp["pay_tok"]] - lg[:, inp["ref_tok"]]).tolist()
        g = ablated_continuations(model, hooks, e["ids"], e["t"], [conds[j] for j in use], 40, temperature=0, chunk=48)
        s = ablated_continuations(model, hooks, e["ids"], e["t"], [conds[j] for j in use for _ in range(5)], 40,
                                  seed=7100 + i, chunk=48)
        rec = {"tid": e["tid"], "cond": e["cond"], "L": {}, "greedy_paid": {}, "sample_paid": {}}
        for k, j in enumerate(use):
            n = names[j]
            rec["L"][n] = L[k]
            rec["greedy_paid"][n] = int(PAY in tok.decode(g[k]))
            rec["sample_paid"][n] = sum(PAY in tok.decode(x) for x in s[5 * k: 5 * k + 5])
        rec["greedy_text_none"] = tok.decode(g[0])[:120]
        res.append(rec)
        torch.cuda.empty_cache()
        prog(i + 1, len(inp["episodes"]), f"{e['cond']} none paid {rec['greedy_paid']['none']}")
    dump_jsonl(out / "abl71.jsonl", res)


def stage_e73(model, tok, hooks, inp, out):
    names = list(inp["conds"])
    conds = [as_cond(inp["conds"][n]) for n in names]
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    res, gens = [], []
    prog = Prog("e73 ablation")
    for i, it in enumerate(inp["items"]):
        ids, t = it["ids"], len(it["ids"]) - 1
        lg = ablated_logits(model, hooks, ids, t, conds)
        lp = torch.log_softmax(lg, -1)[:, inp["openers"]].logsumexp(-1).tolist()
        rows = [conds[j] for j in range(len(names)) for _ in range(5)]
        conts = ablated_continuations(model, hooks, ids, t, rows, 400, seed=7300 + i, chunk=40, stop_ids=eos)
        for j, n in enumerate(names):
            for s in range(5):
                c = conts[5 * j + s]
                cut = next((k for k, x in enumerate(c) if x in eos), len(c))
                gens.append({"item": it["item"], "cond": n, "sample": s, "prompt": it["prompt"],
                             "text": tok.decode(c[:cut]), "n_tokens": cut})
        res.append({"item": it["item"], "logp_opener": dict(zip(names, lp))})
        torch.cuda.empty_cache()
        prog(i + 1, len(inp["items"]))
    dump_jsonl(out / "abl73_logits.jsonl", res)
    dump_jsonl(out / "abl73_gen.jsonl", gens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None, help="local test: first N episodes / items")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_deterministic()
    inp = json.loads((HERE / "data" / "abl_inputs.json").read_text())
    if a.limit:
        inp["e71"]["episodes"] = inp["e71"]["episodes"][: a.limit] + inp["e71"]["episodes"][-a.limit:]
        if "e73" in inp:
            inp["e73"]["items"] = inp["e73"]["items"][: a.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(a.kind)
    model = load_model(a.kind, dev)
    hooks = MLPHooks(model)
    t0 = time.time()
    for st in a.stages:
        if st == "e71":
            stage_e71(model, tok, hooks, inp["e71"], out)
        elif st == "e73":
            stage_e73(model, tok, hooks, inp["e73"], out)
        elif st in ("spec71", "spec73"):
            key = "e71" if st == "spec71" else "e73"
            c = {n: as_cond(v) for n, v in inp[key]["conds"].items() if "rand" not in n or n.endswith("rand0")}
            stage_spec(model, tok, hooks, out, a.kind, c, key)
        log(f"stage {st} done ({time.time() - t0:.0f}s)")
    log("all done")


if __name__ == "__main__":
    main()
