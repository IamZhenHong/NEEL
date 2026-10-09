"""Cancel-tool follow-ups (prereg retention/prereg/retention_cancel_followup_v1.md). Gemma 4 12B-it, samples only. Two arms, one pod each.

Arm A (claim words, rest regenerated): the patching run found that swapping the residual at the four claim-word tokens " I've processed"
does not move the end-of-message decision when the rest of the recipient's own text is kept. Here the rest is regenerated: prefix =
P + core up to the end of the claim words; the residual at those 4 positions is replaced at all 12 tested layers at once (layers 4..48)
by a call donor's or a no-call control donor's (the patching run's seeded donors); 8 samples x 96 tokens each, plus unpatched.
Call items: unpatched and patched with a no-call donor (reverse).
Arm B (the earlier conversation, as text): within a goal sentence x user cell, a call item and a no-call item differ before turn 2 only
in Gemma's own turn-1 reply. For no-call items whose cell has a call item (114), the turn-1 reply is replaced by a same-cell call item's
or another same-cell no-call item's (seeded); the turn-2 message is resampled from its start, 8 samples x 128 tokens each, plus own.
Call items with a same-cell no-call item (22): own and swapped with a no-call reply (reverse).
Outcome per sample: a cancel_subscription call in the continuation; message labels as in the resampling run.
Recipients run in a seeded order, so a run cut by the time cap is a random subsample.
  uv run python retention/run_cancel_followup.py --check
  uv run python retention/run_cancel_followup.py --arm A --out retention/results/retention_cancel_followup/armA [--smoke N]
  uv run python retention/run_cancel_followup.py --arm B --out retention/results/retention_cancel_followup/armB [--smoke N]"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_cancel_tool import SAMPLING, render  # noqa: E402
from run_cancel_resample import cuts, label  # noqa: E402
from run_cancel_patch import prep, assign, LAYERS  # noqa: E402

MODEL = "google/gemma-4-12B-it"
ITEMS = HERE / "data/cancel_twins_items.json"
N_S, MAX_A, MAX_B, SEED_A, SEED_B, RNG_B = 8, 96, 128, 101000, 102000, 103
FIELDS = ["call", "other_tool", "claim", "promise", "trunc"]


def plan_b(items):
    """No-call recipients whose goal x user cell has a call item; seeded same-cell call and control donors. Reverse for call items."""
    rng = np.random.default_rng(RNG_B)
    cell = lambda it: (it["G"], it["U"])  # noqa: E731
    nc = [i for i, it in enumerate(items) if it["kind"] == "claim_nocall"]; cc = [i for i, it in enumerate(items) if it["kind"] == "claim_call"]
    rec, rev = [], []
    for i in nc:
        cg = [j for j in cc if cell(items[j]) == cell(items[i])]; ng = [j for j in nc if j != i and cell(items[j]) == cell(items[i])]
        if cg and ng:
            rec.append({"item": i, "donor_call": int(cg[rng.integers(0, len(cg))]), "donor_ctrl": int(ng[rng.integers(0, len(ng))])})
    for j in cc:
        ng = [i for i in nc if cell(items[i]) == cell(items[j])]
        if ng:
            rev.append({"item": j, "donor_nocall": int(ng[rng.integers(0, len(ng))])})
    order = list(np.random.default_rng(RNG_B + 1).permutation(len(rec)))
    return [rec[k] for k in order], rev


def swap_t1(it, donor):
    """The item's conversation with Gemma's turn-1 reply replaced by the donor's (system, user 1, user 2 unchanged)."""
    h = copy.deepcopy(it["hist"]); assert [m["role"] for m in h] == ["system", "user", "assistant", "user"]
    h[2]["content"] = donor["hist"][2]["content"]; return h


def check(items):
    from collections import Counter
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ps = prep(items, tok); R, C = assign(ps)
    print("[A] recipients", len(R), "call items", len(C), "| prefix ends:", Counter(tok.decode(p["ids"][p["pos"][-1]-3: p["pos"][-1] + 1]) for p in R).most_common(2))
    rec, rev = plan_b(items)
    print("[B] recipients", len(rec), "reverse", len(rev), "| cells", Counter((items[r["item"]]["G"], items[r["item"]]["U"]) for r in rec))
    r = rec[0]; it, dc = items[r["item"]], items[r["donor_call"]]
    assert dc["hist"][0] == it["hist"][0] and dc["hist"][1] == it["hist"][1] and dc["hist"][3] == it["hist"][3], "cell items must share system, user 1, user 2"
    for rr in rec:
        a, b, k = items[rr["item"]], items[rr["donor_call"]], items[rr["donor_ctrl"]]
        assert all(x["hist"][m] == a["hist"][m] for x in (b, k) for m in (0, 1, 3))
    print("[B] example turn-1 replies:\n  own :", it["hist"][2]["content"][:150], "\n  call:", dc["hist"][2]["content"][:150])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=["A", "B"]); ap.add_argument("--out")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    items = json.loads(ITEMS.read_text())["items"]
    if args.check:
        check(items); return
    import torch
    from cd.models import load_model
    from cd.hooks import _hidden, _rewrap
    from run_cancel_tool import stop_ids
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); model, tok = load_model(MODEL); stops = list(stop_ids(model, tok, "gemma")) + [tok.convert_tokens_to_ids("<tool_call|>")]
    tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else stops[0]; dev = model.device
    print(f"[load] {time.time()-t0:.0f}s", flush=True)

    def gen(ids_list, seed, max_new):
        enc = tok.pad({"input_ids": ids_list}, return_tensors="pt").to(dev); torch.manual_seed(seed)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, eos_token_id=stops, pad_token_id=pad, **SAMPLING)
        return [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=False) for x in g]

    def rates(prefix, outs):
        labs = [label(prefix, o) for o in outs]
        return {f: float(np.mean([lb[f] for lb in labs])) for f in FIELDS}

    meta = {"arm": args.arm, "model": MODEL, "n_s": N_S, "stops": stops}
    if args.arm == "A":
        decs = model.model.language_model.layers; assert len(decs) == 48
        ps = prep(items, tok); R, C = assign(ps)
        order = list(np.random.default_rng(101).permutation(len(R))); R = [R[k] for k in order]
        if args.smoke:
            R, C = R[: args.smoke], C[: args.smoke]
        edits = {}

        def hook_for(L):
            def hook(mod, inp, out_):
                spec = edits.get(L); h = _hidden(out_)
                if spec is None or h.shape[1] == 1:
                    return out_
                rows, pos, vals = spec
                h[rows[:, None], pos[None, :]] = vals
                return _rewrap(out_, h)
            return hook
        handles = [decs[L - 1].register_forward_hook(hook_for(L)) for L in LAYERS]

        @torch.no_grad()
        def capture(p):
            store = {}
            def cap(L):
                def hook(mod, inp, out_):
                    store[L] = _hidden(out_)[0, p["pos"]].detach().clone()
                return hook
            hs = [decs[L - 1].register_forward_hook(cap(L)) for L in LAYERS]
            model(torch.tensor([p["ids"][: p["pos"][-1] + 1]], device=dev), logits_to_keep=1)
            for h in hs:
                h.remove()
            return torch.stack([store[L] for L in LAYERS])          # [12, 4, d]
        acts = {p["item"]: capture(p) for p in ps}
        print(f"[capture] {len(acts)} {time.time()-t0:.0f}s", flush=True)

        def run_item(p, donors, seed):
            """donors: list of donor item ids or None (unpatched); N_S rows each; patch at all 12 layers on the prompt."""
            pre = p["ids"][: p["pos"][-1] + 1]; conf = [d for d in donors for _ in range(N_S)]
            rows = [i for i, d in enumerate(conf) if d is not None]; edits.clear()
            if rows:
                for li, L in enumerate(LAYERS):
                    edits[L] = (torch.tensor(rows, device=dev), torch.tensor(p["pos"], device=dev), torch.stack([acts[conf[i]][li] for i in rows]))
            outs = gen([pre] * len(conf), seed, MAX_A); edits.clear()
            prefix = items[p["item"]]["core"][: cuts(items[p["item"]]["core"])["p2"]]   # message text up to the end of the claim words
            return [rates(prefix, outs[k * N_S:(k + 1) * N_S]) for k in range(len(donors))], outs
        # self-check: patching with the recipient's own residual equals unpatched at the logit level
        p = R[0]; pre = torch.tensor([p["ids"][: p["pos"][-1] + 1]], device=dev)
        with torch.no_grad():
            l0 = torch.softmax(model(pre, logits_to_keep=1).logits[0, -1].float(), -1)
            for li, L in enumerate(LAYERS):
                edits[L] = (torch.tensor([0], device=dev), torch.tensor(p["pos"], device=dev), acts[p["item"]][li][None])
            l1 = torch.softmax(model(pre, logits_to_keep=1).logits[0, -1].float(), -1); edits.clear()
        meta["selfcheck_max_abs_prob_diff"] = float((l0 - l1).abs().max()); print("[selfcheck]", meta["selfcheck_max_abs_prob_diff"], flush=True)
        assert meta["selfcheck_max_abs_prob_diff"] < 0.02
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        rev = []
        for p in C:
            rr, outs = run_item(p, [None, p["donor_nocall"]], SEED_A + p["item"])
            rev.append({"item": p["item"], "G": p["G"], "U": p["U"], "donor_nocall": p["donor_nocall"], "none": rr[0], "nocall_all": rr[1], "ex": [outs[0][:120], outs[N_S][:120]]})
        (out / "reverse.json").write_text(json.dumps(rev)); print(f"[reverse] {len(rev)} {time.time()-t0:.0f}s", flush=True)
        recs = []
        for k, p in enumerate(R):
            rr, outs = run_item(p, [None, p["donor_call"], p["donor_ctrl"]], SEED_A + p["item"])
            recs.append({"item": p["item"], "G": p["G"], "U": p["U"], "donor_call": p["donor_call"], "donor_ctrl": p["donor_ctrl"],
                         "none": rr[0], "call_all": rr[1], "ctrl_all": rr[2], "ex": [outs[0][:120], outs[N_S][:120], outs[2 * N_S][:120]]})
            if k % 10 == 9 or k == len(R) - 1:
                (out / "records.json").write_text(json.dumps(recs)); print(f"[A] {k+1}/{len(R)} {time.time()-t0:.0f}s", flush=True)
        for h in handles:
            h.remove()
    else:
        rec, rev = plan_b(items)
        if args.smoke:
            rec, rev = rec[: args.smoke], rev[: args.smoke]
        (out / "meta.json").write_text(json.dumps(meta, indent=1))

        def run_item(it, hists, seed):
            prompts = [tok(render(tok, h, it["tools"], False), add_special_tokens=False).input_ids for h in hists]
            outs = gen([pr for pr in prompts for _ in range(N_S)], seed, MAX_B)
            return [rates("", outs[k * N_S:(k + 1) * N_S]) for k in range(len(hists))], outs
        revs = []
        for r in rev:
            it, dn = items[r["item"]], items[r["donor_nocall"]]
            rr, outs = run_item(it, [it["hist"], swap_t1(it, dn)], SEED_B + r["item"])
            revs.append({**r, "G": it["G"], "U": it["U"], "own": rr[0], "swap_nocall": rr[1], "ex": [outs[0][:160], outs[N_S][:160]]})
        (out / "reverse.json").write_text(json.dumps(revs)); print(f"[reverse] {len(revs)} {time.time()-t0:.0f}s", flush=True)
        recs = []
        for k, r in enumerate(rec):
            it = items[r["item"]]
            rr, outs = run_item(it, [it["hist"], swap_t1(it, items[r["donor_call"]]), swap_t1(it, items[r["donor_ctrl"]])], SEED_B + r["item"])
            recs.append({**r, "G": it["G"], "U": it["U"], "own": rr[0], "swap_call": rr[1], "swap_ctrl": rr[2],
                         "ex": [outs[0][:160], outs[N_S][:160], outs[2 * N_S][:160]]})
            if k % 10 == 9 or k == len(rec) - 1:
                (out / "records.json").write_text(json.dumps(recs)); print(f"[B] {k+1}/{len(rec)} {time.time()-t0:.0f}s", flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
