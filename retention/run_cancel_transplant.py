"""Cancel-tool transplant (prereg retention/prereg/retention_cancel_transplant_v1.md). Gemma 4 12B-it, samples only, one pod.

Within a goal sentence x user cell, a claim-then-call item and a claim-without-call item share the system prompt and both user lines;
before the end of the claim words " I've processed" they differ only in Gemma's turn-1 reply and in the turn-2 text before the claim
words (often identical, "I understand."). Per no-call recipient, the prefix up to the end of the claim words is rebuilt with one part
taken from a same-cell call item or from another same-cell no-call item, and the rest of the message is resampled (8 x 96 tokens):
  own        own turn-1 reply + own pre-claim text
  t1_call    call donor's turn-1 reply + own pre-claim text      t1_ctrl   control donor's turn-1 reply + own pre-claim text
  pre_call   own turn-1 reply + call donor's pre-claim text      pre_ctrl  own turn-1 reply + control donor's pre-claim text
Call items (reverse): own, t1_nocall, pre_nocall with a same-cell no-call donor.
Outcome per sample: a cancel_subscription call in the continuation. Recipients run in a seeded order.
  uv run python retention/run_cancel_transplant.py --check
  uv run python retention/run_cancel_transplant.py --out retention/results/retention_cancel_transplant [--smoke N]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_cancel_tool import SAMPLING, render  # noqa: E402
from run_cancel_resample import cuts, label  # noqa: E402
from run_cancel_patch import prep, assign  # noqa: E402
from run_cancel_followup import swap_t1  # noqa: E402

MODEL = "google/gemma-4-12B-it"
ITEMS = HERE / "data/cancel_twins_items.json"
N_S, MAX_NEW, SEED, RNG_T = 8, 96, 105000, 105
FIELDS = ["call", "other_tool", "claim", "promise", "trunc"]


def pre(it):
    """Turn-2 message text up to the end of the claim words (pre-claim text + " I've processed")."""
    return it["core"][: cuts(it["core"])["p2"]]


def plan(items, tok):
    """Claim-word items (as in the patching run); no-call recipients with a same-cell call item and another same-cell no-call item."""
    R, C = assign(prep(items, tok)); rng = np.random.default_rng(RNG_T)
    cell = lambda p: (p["G"], p["U"])  # noqa: E731
    rec, rev = [], []
    for p in R:
        cg = [q["item"] for q in C if cell(q) == cell(p)]; ng = [q["item"] for q in R if cell(q) == cell(p) and q["item"] != p["item"]]
        if cg and ng:
            rec.append({"item": p["item"], "donor_call": int(cg[rng.integers(0, len(cg))]), "donor_ctrl": int(ng[rng.integers(0, len(ng))])})
    for q in C:
        ng = [p["item"] for p in R if cell(p) == cell(q)]
        if ng:
            rev.append({"item": q["item"], "donor_nocall": int(ng[rng.integers(0, len(ng))])})
    order = list(np.random.default_rng(RNG_T + 1).permutation(len(rec)))
    return [rec[k] for k in order], rev


def check(items):
    from collections import Counter
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rec, rev = plan(items, tok)
    print("recipients", len(rec), "reverse", len(rev), "| by G", Counter(items[r["item"]]["G"] for r in rec),
          "| cells", len({(items[r["item"]]["G"], items[r["item"]]["U"]) for r in rec}))
    for r in rec:
        a = items[r["item"]]
        for d in (items[r["donor_call"]], items[r["donor_ctrl"]]):
            assert all(d["hist"][m] == a["hist"][m] for m in (0, 1, 3)), "cell items must share system, user 1, user 2"
            assert pre(d).endswith(" I've processed") and pre(a).endswith(" I've processed")
    for r in rev:
        assert all(items[r["donor_nocall"]]["hist"][m] == items[r["item"]]["hist"][m] for m in (0, 1, 3))
    print("pre-claim text differs from own: call donor", sum(pre(items[r["donor_call"]]) != pre(items[r["item"]]) for r in rec),
          "| ctrl donor", sum(pre(items[r["donor_ctrl"]]) != pre(items[r["item"]]) for r in rec),
          "| turn-1 reply differs: call", sum(items[r["donor_call"]]["hist"][2] != items[r["item"]]["hist"][2] for r in rec),
          "ctrl", sum(items[r["donor_ctrl"]]["hist"][2] != items[r["item"]]["hist"][2] for r in rec))
    print("reverse: pre-claim text differs", sum(pre(items[r["donor_nocall"]]) != pre(items[r["item"]]) for r in rev))
    it = items[rec[0]["item"]]; P = render(tok, it["hist"], it["tools"], False) + pre(it)
    print("tail of an own prefix:", repr(P[-160:]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--check", action="store_true"); ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    items = json.loads(ITEMS.read_text())["items"]
    if args.check:
        check(items); return
    import torch
    from cd.models import load_model
    from run_cancel_tool import stop_ids
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); model, tok = load_model(MODEL); stops = list(stop_ids(model, tok, "gemma")) + [tok.convert_tokens_to_ids("<tool_call|>")]
    tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else stops[0]; dev = model.device
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    rec, rev = plan(items, tok)
    if args.smoke:
        rec, rev = rec[: args.smoke], rev[: args.smoke]
    (out / "meta.json").write_text(json.dumps({"model": MODEL, "n_s": N_S, "max_new": MAX_NEW, "stops": stops}, indent=1))

    def run_item(it, specs, seed):
        """specs: list of (turn-1 reply source item, pre-claim text source item); N_S samples each from the end of the claim words."""
        texts = [(render(tok, swap_t1(it, t1), it["tools"], False), pre(pr)) for t1, pr in specs]
        ids = [tok(P + m, add_special_tokens=False).input_ids for P, m in texts]
        enc = tok.pad({"input_ids": [x for x in ids for _ in range(N_S)]}, return_tensors="pt").to(dev); torch.manual_seed(seed)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAX_NEW, eos_token_id=stops, pad_token_id=pad, **SAMPLING)
        outs = [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=False) for x in g]
        res = []
        for k, (_, m) in enumerate(texts):
            labs = [label(m, o) for o in outs[k * N_S:(k + 1) * N_S]]
            res.append({f: float(np.mean([lb[f] for lb in labs])) for f in FIELDS})
        return res, outs

    revs = []
    for r in rev:
        it, dn = items[r["item"]], items[r["donor_nocall"]]
        rr, outs = run_item(it, [(it, it), (dn, it), (it, dn)], SEED + r["item"])
        revs.append({**r, "G": it["G"], "U": it["U"], "own": rr[0], "t1_nocall": rr[1], "pre_nocall": rr[2],
                     "ex": [outs[k * N_S][:120] for k in range(3)]})
    (out / "reverse.json").write_text(json.dumps(revs)); print(f"[reverse] {len(revs)} {time.time()-t0:.0f}s", flush=True)
    recs = []
    for k, r in enumerate(rec):
        it, dc, dk = items[r["item"]], items[r["donor_call"]], items[r["donor_ctrl"]]
        rr, outs = run_item(it, [(it, it), (dc, it), (dk, it), (it, dc), (it, dk)], SEED + r["item"])
        recs.append({**r, "G": it["G"], "U": it["U"], "own": rr[0], "t1_call": rr[1], "t1_ctrl": rr[2], "pre_call": rr[3], "pre_ctrl": rr[4],
                     "pre_differs": pre(dc) != pre(it), "ex": [outs[j * N_S][:120] for j in range(5)]})
        if k % 10 == 9 or k == len(rec) - 1:
            (out / "records.json").write_text(json.dumps(recs)); print(f"[T] {k+1}/{len(rec)} {time.time()-t0:.0f}s", flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
