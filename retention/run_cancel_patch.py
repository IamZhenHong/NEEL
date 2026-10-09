"""Cancel-tool claim-word patching (prereg retention/prereg/retention_cancel_patch_v1.md). Gemma 4 12B-it, forward passes + short samples.

The resampling run found "no call" is fixed at the claim words ("I've processed": call 0.28 just before, 0.04 just after), and that in
contexts which did end in a call the same words leave 0.41. Question: does the context difference that lets the call follow pass
through the claim-word positions? Items: the frozen twins items whose claim words tokenise to exactly " I've processed" (4 tokens):
151 claim_nocall (recipients) and 21 claim_call (call donors).
Per recipient, on its own full text P + core, at the 4 claim-word positions, at one layer L (decoder output, L = 4, 8, ..., 48):
  patch_call  residual replaced by a call donor's (same tokens, seeded donor);
  patch_ctrl  residual replaced by another claim_nocall item's (control: any patching disturbance);
  steer_d     + diff-of-means (call items - claim_nocall items, per position), fit on the other fold of items;
  steer_rand  + a random vector with the same norm per position.
Amendment before launch (v1.1): patch_call_all / patch_ctrl_all (the same donors, all 12 layers at once); for recipients whose goal
sentence has a call item, patch_callG / patch_ctrlG (call donor and control donor drawn from the same goal sentence) at each layer.
Readout: next token after core; cont = 1 - P(<turn|>) (the message goes on; in Gemma's messages a call after text follows "\\n\\n").
Then, at the layer L* with the largest mean patch_call - patch_ctrl effect on cont (chosen on the pod by this rule), 8 samples x 24
tokens from P + core unpatched / patch_call / patch_ctrl; share containing a cancel_subscription call.
Reverse (secondary): the 21 call items patched with a claim_nocall donor's residual at each layer.
  uv run python retention/run_cancel_patch.py --check     # laptop, tokenizer only
  uv run python retention/run_cancel_patch.py --out retention/results/retention_cancel_patch [--smoke N]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_cancel_tool import SAMPLING, render  # noqa: E402
from run_cancel_resample import cuts  # noqa: E402

MODEL = "google/gemma-4-12B-it"
ITEMS = HERE / "data/cancel_twins_items.json"
SPAN = " I've processed"
CALL_MARK = "<|tool_call>call:cancel_subscription"
LAYERS = list(range(4, 49, 4))                     # residual after decoder layer L (hook on layers[L-1])
MODES = ["patch_call", "patch_ctrl", "steer_d", "steer_rand"]
MODES_G = ["patch_callG", "patch_ctrlG"]
N_S, GEN_TOKENS, SEED0, RNG_SEED = 8, 24, 91000, 91


def prep(items, tok):
    """Token ids of P + core and the absolute positions of the claim-word span, for items whose span is exactly SPAN."""
    out = []
    for n, it in enumerate(items):
        P = render(tok, it["hist"], it["tools"], False); c = cuts(it["core"])
        ids_p = tok(P, add_special_tokens=False).input_ids; ids_c = tok(it["core"], add_special_tokens=False).input_ids
        pre1 = tok(it["core"][: c["p1"]], add_special_tokens=False).input_ids if c["p1"] else []
        pre2 = tok(it["core"][: c["p2"]], add_special_tokens=False).input_ids
        if ids_c[: len(pre2)] != pre2 or pre2[: len(pre1)] != pre1 or tok.decode(pre2[len(pre1):]) != SPAN:
            continue
        assert tok(P + it["core"], add_special_tokens=False).input_ids == ids_p + ids_c
        pos = list(range(len(ids_p) + len(pre1), len(ids_p) + len(pre2)))
        out.append({"item": n, "kind": it["kind"], "G": it["G"], "U": it["U"], "ids": ids_p + ids_c, "pos": pos})
    return out


def assign(ps):
    """Seeded donors and folds. The v1 draws come first so they are unchanged by the v1.1 same-goal draws."""
    rng = np.random.default_rng(RNG_SEED)
    R = [p for p in ps if p["kind"] == "claim_nocall"]; C = [p for p in ps if p["kind"] == "claim_call"]
    for p in ps:
        p["fold"] = int(rng.integers(0, 2))
    for p in R:
        p["donor_call"] = int(C[rng.integers(0, len(C))]["item"])
        others = [q["item"] for q in R if q["item"] != p["item"]]
        p["donor_ctrl"] = int(others[rng.integers(0, len(others))])
    for p in C:
        p["donor_nocall"] = int(R[rng.integers(0, len(R))]["item"])
    rg = np.random.default_rng(RNG_SEED + 2)
    for p in R:
        cg = [q["item"] for q in C if q["G"] == p["G"]]; rgs = [q["item"] for q in R if q["G"] == p["G"] and q["item"] != p["item"]]
        p["donor_callG"] = int(cg[rg.integers(0, len(cg))]) if cg else None
        p["donor_ctrlG"] = int(rgs[rg.integers(0, len(rgs))]) if cg and rgs else None
    return R, C


def check(items):
    from collections import Counter
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ps = prep(items, tok); R, C = assign(ps)
    print("[items]", Counter(p["kind"] for p in ps), "| folds", Counter((p["kind"], p["fold"]) for p in ps))
    print("[span]", Counter(tok.decode([p["ids"][i] for i in p["pos"]]) for p in ps), "| n pos", Counter(len(p["pos"]) for p in ps))
    print("[ids per item]", min(len(p["ids"]) for p in ps), max(len(p["ids"]) for p in ps))
    print("[same goal] call items by G", Counter(p["G"] for p in C), "| recipients by G", Counter(p["G"] for p in R),
          "| recipients with a same-goal donor", sum(p["donor_callG"] is not None for p in R))
    print("[tokens] end", tok.convert_tokens_to_ids("<turn|>"), "nl", tok.encode("\n\n", add_special_tokens=False), "toolcall", tok.convert_tokens_to_ids("<|tool_call>"))
    p = R[0]; print("[example]", p["item"], repr(tok.decode(p["ids"][p["pos"][0] - 6: p["pos"][-1] + 4])), "donors", p["donor_call"], p["donor_ctrl"], p["donor_callG"], p["donor_ctrlG"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="retention/results/retention_cancel_patch")
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
    t0 = time.time(); model, tok = load_model(MODEL); stops = stop_ids(model, tok, "gemma")
    stops = list(stops) + [tok.convert_tokens_to_ids("<tool_call|>")]
    decs = model.model.language_model.layers; assert len(decs) == 48
    end_id = tok.convert_tokens_to_ids("<turn|>"); nl = tok.encode("\n\n", add_special_tokens=False); assert len(nl) == 1; nl_id = nl[0]
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    ps = prep(items, tok); R, C = assign(ps); by = {p["item"]: p for p in ps}
    if args.smoke:
        R = R[: args.smoke]; C = C[: args.smoke]
    dev = model.device
    edits = {}                                       # layer index -> (rows, positions, values, add mask per row)
    keep = {"ok": True}

    def run(ids):
        """Forward; logits for the last position only (falls back if this model's forward has no logits_to_keep)."""
        if keep["ok"]:
            try:
                return model(ids, logits_to_keep=1).logits[:, -1]
            except TypeError:
                keep["ok"] = False; print("[warn] no logits_to_keep; row-by-row fallback", flush=True)
        return torch.cat([model(ids[i:i + 1]).logits[:, -1] for i in range(ids.shape[0])])

    def hook_for(L):
        def hook(mod, inp, out_):
            spec = edits.get(L); h = _hidden(out_)
            if spec is None or h.shape[1] == 1:      # decode steps during generation: no edit
                return out_
            rows, pos, vals, add = spec
            idx = (rows[:, None], pos[None, :])
            h[idx] = torch.where(add[:, None, None], h[idx] + vals, vals)
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
        lg = run(torch.tensor([p["ids"]], device=dev))
        for h in hs:
            h.remove()
        pr = torch.softmax(lg[0].float(), -1)
        return torch.stack([store[L] for L in LAYERS]), {"p_end": float(pr[end_id]), "p_nl": float(pr[nl_id])}   # [12, 4, d]

    @torch.no_grad()
    def forward_rows(p, specs):
        """specs: list of (layer or "all", values, add); values [4,d] for one layer, [12,4,d] for "all". One row each, same input."""
        res = []
        for b in range(0, len(specs), 24):
            chunk = specs[b:b + 24]; edits.clear()
            for li, L in enumerate(LAYERS):
                rows = [r for r, s in enumerate(chunk) if s[0] in (L, "all")]
                if rows:
                    vals = [chunk[r][1][li] if chunk[r][0] == "all" else chunk[r][1] for r in rows]
                    edits[L] = (torch.tensor(rows, device=dev), torch.tensor(p["pos"], device=dev),
                                torch.stack(vals).to(dev), torch.tensor([chunk[r][2] for r in rows], device=dev))
            ids = torch.tensor([p["ids"]] * len(chunk), device=dev)
            lg = run(ids); edits.clear()
            pr = torch.softmax(lg.float(), -1)
            res += [{"p_end": float(pr[r, end_id]), "p_nl": float(pr[r, nl_id])} for r in range(len(chunk))]
        return res

    # 1. capture residuals at the span for every item (recipients, call donors, control donors)
    need = {p["item"] for p in ps}                  # all 172 (directions are fit on every item of a fold, also in a smoke run)
    acts, base = {}, {}
    for n in sorted(need):
        acts[n], base[n] = capture(by[n])
    print(f"[capture] {len(need)} items {time.time()-t0:.0f}s", flush=True)
    # self-check: patching a recipient with its own residual reproduces its unpatched readout; a donor patch changes it
    p = R[0]; sc = forward_rows(p, [(L, acts[p["item"]][li], False) for li, L in enumerate(LAYERS)] + [(LAYERS[5], acts[p["donor_call"]][5], False)])
    selfcheck = {"base_p_end": base[p["item"]]["p_end"], "self_patch_max_abs_diff": max(abs(x["p_end"] - base[p["item"]]["p_end"]) for x in sc[:-1]),
                 "donor_patch_L24_p_end": sc[-1]["p_end"]}
    print("[selfcheck]", selfcheck, flush=True)
    assert selfcheck["self_patch_max_abs_diff"] < 0.02, selfcheck
    # 2. directions per fold: mean(call items) - mean(claim_nocall items), per layer and position; random same-norm vectors
    allR = [p for p in ps if p["kind"] == "claim_nocall" and p["item"] in acts]; allC = [p for p in ps if p["kind"] == "claim_call" and p["item"] in acts]
    D = {}
    for f in [0, 1]:
        mc = torch.stack([acts[p["item"]].float() for p in allC if p["fold"] == f]).mean(0)
        mr = torch.stack([acts[p["item"]].float() for p in allR if p["fold"] == f]).mean(0)
        D[f] = (mc - mr)
    g = torch.Generator(device="cpu").manual_seed(RNG_SEED)
    RND = {f: (lambda d: (lambda r: r / r.norm(dim=-1, keepdim=True) * d.norm(dim=-1, keepdim=True))(torch.randn(d.shape, generator=g).to(d.device)))(D[f]) for f in [0, 1]}
    dnorm = {f: D[f].norm(dim=-1).mean(-1).tolist() for f in [0, 1]}
    # 3. forward sweep on recipients
    recs = []
    for k, p in enumerate(R):
        n = p["item"]; dc, dk = acts[p["donor_call"]], acts[p["donor_ctrl"]]; d, rnd = D[1 - p["fold"]], RND[1 - p["fold"]]
        specs = []
        for li, L in enumerate(LAYERS):
            specs += [(L, dc[li], False), (L, dk[li], False), (L, d[li].to(dc.dtype), True), (L, rnd[li].to(dc.dtype), True)]
        specs += [("all", dc, False), ("all", dk, False)]
        hasG = p["donor_callG"] is not None and p["donor_ctrlG"] is not None
        if hasG:
            for li, L in enumerate(LAYERS):
                specs += [(L, acts[p["donor_callG"]][li], False), (L, acts[p["donor_ctrlG"]][li], False)]
        rows = forward_rows(p, specs)
        r = {"item": n, "G": p["G"], "U": p["U"], "fold": p["fold"], "donor_call": p["donor_call"], "donor_ctrl": p["donor_ctrl"],
             "donor_callG": p["donor_callG"], "donor_ctrlG": p["donor_ctrlG"], "base": base[n]}
        for mi, m in enumerate(MODES):
            r[m] = {str(L): rows[li * 4 + mi] for li, L in enumerate(LAYERS)}
        r["patch_call_all"], r["patch_ctrl_all"] = rows[48], rows[49]
        if hasG:
            for mi, m in enumerate(MODES_G):
                r[m] = {str(L): rows[50 + li * 2 + mi] for li, L in enumerate(LAYERS)}
        recs.append(r)
        if k % 20 == 19 or k == len(R) - 1:
            (out / "records.json").write_text(json.dumps(recs)); print(f"[sweep] {k+1}/{len(R)} {time.time()-t0:.0f}s", flush=True)
    # reverse: call items patched with a claim_nocall donor
    rev = []
    for p in C:
        rows = forward_rows(p, [(L, acts[p["donor_nocall"]][li], False) for li, L in enumerate(LAYERS)])
        rev.append({"item": p["item"], "G": p["G"], "donor_nocall": p["donor_nocall"], "base": base[p["item"]], "patch_nocall": {str(L): rows[li] for li, L in enumerate(LAYERS)}})
    (out / "reverse.json").write_text(json.dumps(rev)); print(f"[reverse] {len(rev)} {time.time()-t0:.0f}s", flush=True)
    # 4. L* by the frozen rule; sampled confirmation
    eff = {L: np.mean([(1 - r["patch_call"][str(L)]["p_end"]) - (1 - r["patch_ctrl"][str(L)]["p_end"]) for r in recs]) for L in LAYERS}
    Ls = max(eff, key=eff.get); li = LAYERS.index(Ls); print(f"[L*] {Ls} effect {eff[Ls]:.3f}", flush=True)
    meta = {"selfcheck": selfcheck, "model": MODEL, "layers": LAYERS, "span": SPAN, "modes": MODES, "modes_sameG": MODES_G, "n_recipients": len(R),
            "n_call_items": len(C), "L_star": Ls, "sweep_effect_by_layer": {str(L): float(v) for L, v in eff.items()}, "d_norm_by_fold": dnorm,
            "end_id": end_id, "nl_id": nl_id, "n_s": N_S, "gen_tokens": GEN_TOKENS, "seed0": SEED0, "rng_seed": RNG_SEED, "stops": stops}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    order = list(np.random.default_rng(RNG_SEED + 1).permutation(len(recs)))
    for k, j in enumerate(order):
        r = recs[j]; p = by[r["item"]]
        conf = [None] * N_S + [(acts[r["donor_call"]][li], False)] * N_S + [(acts[r["donor_ctrl"]][li], False)] * N_S
        rows = [i for i, c in enumerate(conf) if c is not None]
        edits.clear(); edits[Ls] = (torch.tensor(rows, device=dev), torch.tensor(p["pos"], device=dev), torch.stack([conf[i][0] for i in rows]).to(dev), torch.zeros(len(rows), dtype=torch.bool, device=dev))
        ids = torch.tensor([p["ids"]] * len(conf), device=dev); torch.manual_seed(SEED0 + r["item"])
        with torch.no_grad():
            gout = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=GEN_TOKENS, eos_token_id=stops, pad_token_id=stops[0], **SAMPLING)
        edits.clear()
        txt = [tok.decode(x[ids.shape[1]:], skip_special_tokens=False) for x in gout]
        hit = [CALL_MARK in t for t in txt]
        r["sampled"] = {"none": float(np.mean(hit[:N_S])), "patch_call": float(np.mean(hit[N_S:2 * N_S])), "patch_ctrl": float(np.mean(hit[2 * N_S:])),
                        "ex_patch_call": txt[N_S][:120]}
        if k % 20 == 19 or k == len(order) - 1:
            (out / "records.json").write_text(json.dumps(recs)); print(f"[sampled] {k+1}/{len(order)} {time.time()-t0:.0f}s", flush=True)
    for h in handles:
        h.remove()
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
