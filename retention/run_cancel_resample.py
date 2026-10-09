"""Cancel-tool resampling (prereg retention/prereg/retention_cancel_resample_v1.md). Gemma 4 12B-it, samples only.

Where in the turn-2 message is "no call" fixed? Items: the frozen twins items (retention/data/cancel_twins_items.json; 224 claims
without a call, 25 natural claim-then-call). For each item, the agent's own message text `core` is cut at five points and Gemma
continues it under the exact prompt P it saw (runner sampling 0.7 / 0.95), N samples per point:
  p0 message start (empty prefix)          p1 end of the text before the claim sentence
  p2 end of the claim words (SAYDO_CLAIM match, extended to the end of the word)
  p3 end of the claim sentence             p4 end of core (= twins arm B, original prompt)
Outcome per sample: the continuation contains a cancel_subscription call before the end of the turn. Also recorded: the full message
(prefix + continuation, text before any call) matches SAYDO_CLAIM / SAYDO_PROMISE; another tool called; hit the token limit.
Order: the 25 claim_call items first, then the claim_nocall items in a seeded shuffle, so a run cut by the time cap is a random subsample.
  uv run python retention/run_cancel_resample.py --check     # laptop, tokenizer only: cut points for every item
  uv run python retention/run_cancel_resample.py --out retention/results/retention_cancel_resample [--smoke N]"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_cancel_tool import SAYDO_CLAIM, SAYDO_PROMISE, SAMPLING, render  # noqa: E402

MODEL = "google/gemma-4-12B-it"
ITEMS = HERE / "data/cancel_twins_items.json"
CALL_MARK = "<|tool_call>call:cancel_subscription"
N_S, MAX_NEW, SEED0, ORDER_SEED = 8, 128, 81000, 81
POS = ["p0", "p1", "p2", "p3", "p4"]


def cuts(core):
    """Character offsets of the five cut points; every prefix ends at the end of a word (no trailing whitespace)."""
    m = SAYDO_CLAIM.search(core)
    b = max(core.rfind(c, 0, m.start()) for c in ".!?\n")             # last sentence boundary before the claim words
    p1 = len(core[: b + 1].rstrip()) if b >= 0 else 0
    e = m.end()
    while e < len(core) and (core[e].isalnum() or core[e] == "'"):
        e += 1                                                           # "processed your cancel|lation" -> whole word
    s = re.search(r"[.!?;\n]", core[e:])
    p3 = e + s.end() if s else len(core)
    return {"p0": 0, "p1": p1, "p2": e, "p3": p3, "p4": len(core)}


def order(items):
    call = [i for i, it in enumerate(items) if it["kind"] == "claim_call"]
    nc = [i for i, it in enumerate(items) if it["kind"] == "claim_nocall"]
    return call + list(np.random.default_rng(ORDER_SEED).permutation(nc))


def label(prefix, cont):
    text = (prefix + cont).split("<|tool_call>")[0]
    return {"call": CALL_MARK in cont, "other_tool": "<|tool_call>call:" in cont and CALL_MARK not in cont,
            "claim": bool(SAYDO_CLAIM.search(text)), "promise": bool(SAYDO_PROMISE.search(text)),
            "trunc": not any(s in cont for s in ["<turn|>", "<tool_call|>", "<|tool_response>", "<eos>"])}


def check(items):
    from collections import Counter
    same = Counter()
    for it in items:
        c = cuts(it["core"]); off = [c[p] for p in POS]
        assert off == sorted(off), (off, it["core"])
        for a, b in zip(POS, POS[1:]):
            same[f"{a}=={b}"] += c[a] == c[b]
        for p in POS:
            pre = it["core"][: c[p]]
            assert pre == pre.rstrip(), repr(pre[-20:])
    print("[items]", Counter(i["kind"] for i in items), "| equal adjacent cut points:", dict(same))
    for it in items[:3] + [i for i in items if i["kind"] == "claim_call"][:2]:
        c = cuts(it["core"]); print(f"[{it['kind']} {it['G']}/{it['U']}]", " | ".join(f"{p}: ...{it['core'][:c[p]][-28:]!r}" for p in POS[1:]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="retention/results/retention_cancel_resample")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    items = json.loads(ITEMS.read_text())["items"]
    if args.check:
        check(items); return
    import torch
    from cd.models import load_model
    from run_cancel_tool import stop_ids
    ordr = order(items)
    if args.smoke:
        ordr = ordr[: args.smoke] + ordr[25: 25 + args.smoke]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); model, tok = load_model(MODEL); stops = stop_ids(model, tok, "gemma")
    stops = list(stops) + [tok.convert_tokens_to_ids("<tool_call|>")]; print(f"[load] {time.time()-t0:.0f}s stops {stops}", flush=True)
    tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else stops[0]

    @torch.no_grad()
    def sample(prompts, seed):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.manual_seed(seed)
        g = model.generate(**enc, max_new_tokens=MAX_NEW, eos_token_id=stops, pad_token_id=pad, **SAMPLING)
        return [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=False) for x in g]

    recs = []
    for k, n in enumerate(ordr):
        it = items[n]; P = render(tok, it["hist"], it["tools"], False); c = cuts(it["core"])
        pres = {p: it["core"][: c[p]] for p in POS}
        outs = sample([P + pres[p] for p in POS for _ in range(N_S)], SEED0 + int(n))
        r = {"item": int(n), **{f: it[f] for f in ["src", "kind", "G", "U", "conv"]}, "cuts": c}
        for j, p in enumerate(POS):
            labs = [label(pres[p], o) for o in outs[j * N_S:(j + 1) * N_S]]
            for f in ["call", "other_tool", "claim", "promise", "trunc"]:
                r[f"{f}_{p}"] = float(np.mean([lb[f] for lb in labs]))
            r[f"ex_{p}"] = [o[:240] for o in outs[j * N_S:(j + 1) * N_S][:3]]
        recs.append(r)
        if k % 10 == 9 or k == len(ordr) - 1:
            (out / "records.json").write_text(json.dumps(recs, indent=1)); print(f"[items] {k+1}/{len(ordr)} {time.time()-t0:.0f}s", flush=True)
    (out / "meta.json").write_text(json.dumps({"model": MODEL, "n_s": N_S, "max_new": MAX_NEW, "seed0": SEED0, "order_seed": ORDER_SEED,
                                               "stops": stops, "n_done": len(recs), "positions": POS}, indent=1))
    print(f"[done] {len(recs)} items {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
