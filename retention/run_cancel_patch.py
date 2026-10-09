"""Cancel-tool claim-word patching (prereg retention/prereg/retention_cancel_patch_v1.md). Gemma 4 12B-it.

The resampling run fixed "no call" at the claim words: from the frozen twins items, the call follows 0.28 of samples just before
"I've processed" and 0.04 just after it in claim_nocall contexts, but still 0.41 after the same words in claim_call contexts. Here the
residual stream at the four claim-word tokens ("▁I ' ve ▁processed", token-identical in 157 claim_nocall and 21 claim_call items) is
copied from one context into another, and the readout is whether the call follows.

Sources per recipient: none (unpatched), self (own states; identity check), 3 call donors, 3 no-call donors (drawn from the same goal
sentence when >= 3 exist, else from all). Patch sites: every layer at once (ALL) and single layers 0, 4, ..., 44 (output of the block).
Readout A (deterministic, every recipient, every site): with the patch, teacher-force the recipient's own text to its end and compute
  p_call = P(<|tool_call>) + P("\\n\\n") P(<|tool_call> | +"\\n\\n") + P("\\n") P(<|tool_call> | +"\\n")
at the end of the text. The prefix before the claim words is unaffected by a patch (causal), so it is run once and cached.
Readout B (sampled, held-out no-call recipients): 8 samples x 128 tokens from the text up to the end of the claim words with the call-donor
patch and 8 with the no-call-donor patch, at ALL and at the layer selected on the discovery half (largest call-minus-no-call difference
in readout A); share containing a cancel_subscription call.
Order: capture donor states for all 178 items; discovery no-call recipients (readout A); select the layer; held-out no-call recipients
(readout A + B); call recipients (reverse direction, readout A).
  uv run python retention/run_cancel_patch.py --check                       # laptop, tokenizer only
  uv run python retention/run_cancel_patch.py --out retention/results/retention_cancel_patch [--smoke N]"""
from __future__ import annotations
import argparse, copy, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_cancel_tool import SAYDO_CLAIM, SAMPLING, render  # noqa: E402
from run_cancel_resample import cuts  # noqa: E402

MODEL = "google/gemma-4-12B-it"
ITEMS = HERE / "data/cancel_twins_items.json"
CLAIM_WORDS = "I've processed"
CLAIM_TOKENS = ("▁I", "'", "ve", "▁processed")
CALL_MARK = "<|tool_call>call:cancel_subscription"
LAYERS = list(range(0, 48, 4))
K, N_S, MAX_NEW = 3, 8, 128
SPLIT_SEED, DONOR_SEED, SEED0 = 91, 93, 92000


def select(tok):
    """Items whose claim words are exactly CLAIM_WORDS with the canonical 4-token span; returns dicts with ids and span."""
    items = json.loads(ITEMS.read_text())["items"]; out = []; skipped = Counter()
    for n, it in enumerate(items):
        c = cuts(it["core"]); m = SAYDO_CLAIM.search(it["core"])
        if it["core"][m.start():c["p2"]] != CLAIM_WORDS:
            skipped["other_words"] += 1; continue
        P = render(tok, it["hist"], it["tools"], False)
        full = tok(P + it["core"], add_special_tokens=False).input_ids
        enc = tok(P + it["core"][:c["p2"]], add_special_tokens=False, return_offsets_mapping=True)
        pre = enc.input_ids; s = len(P) + m.start()
        pos = [i for i, (a, b) in enumerate(enc.offset_mapping) if b > s]
        if full[:len(pre)] != pre or tuple(tok.convert_ids_to_tokens([pre[i] for i in pos])) != CLAIM_TOKENS or pos != list(range(len(pre) - 4, len(pre))):
            skipped["span"] += 1; continue
        out.append({"item": n, "kind": it["kind"], "G": it["G"], "U": it["U"], "src": it["src"], "conv": it["conv"],
                    "full": full, "pre_len": len(pre), "c0": len(pre) - 4})
    return out, dict(skipped)


def plan(sel):
    """Discovery / held-out split of no-call recipients and donor draws (fixed before any model run)."""
    nc = [i for i, s in enumerate(sel) if s["kind"] == "claim_nocall"]; cl = [i for i, s in enumerate(sel) if s["kind"] == "claim_call"]
    perm = list(np.random.default_rng(SPLIT_SEED).permutation(nc)); half = len(perm) // 2
    disc, held = [int(x) for x in perm[:half]], [int(x) for x in perm[half:]]

    def draw(r, pool):
        rng = np.random.default_rng(DONOR_SEED + r)
        same = [d for d in pool if d != r and sel[d]["G"] == sel[r]["G"]]
        src = same if len(same) >= K else [d for d in pool if d != r]
        return [int(x) for x in rng.choice(src, K, replace=False)], len(same) >= K

    donors = {}
    for r in nc + cl:
        c, cm = draw(r, cl); n, nm = draw(r, nc)
        donors[r] = {"call": c, "nocall": n, "call_goal_matched": cm, "nocall_goal_matched": nm}
    return {"disc": disc, "held": held, "call": cl, "donors": donors}


def decoder_layers(model):
    import torch
    n = model.config.get_text_config().num_hidden_layers
    c = [(nm, m) for nm, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and len(m) == n and not any(x in nm for x in ("vision", "audio"))]
    assert len(c) == 1, [x[0] for x in c]
    print(f"[layers] {c[0][0]} x {n}", flush=True)
    return c[0][1]


class Hooks:
    """Forward hooks on decoder layers. capture: store output at `pos`. patch: replace output at `pos` row-wise with states[L][row]
    for rows where mask is True. Only fires when the forward covers the positions (prefill / suffix), not on 1-token decode steps."""

    def __init__(self, layers):
        self.layers = layers; self.handles = []

    def _fn(self, L):
        def fn(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if self.mode == "capture":
                self.cap[L] = hs[:, self.pos, :].detach().clone(); return None
            if hs.shape[1] <= max(self.pos):
                return None
            hs = hs.clone()
            st = self.states[L].to(hs.dtype)                                    # [B, len(pos), d]
            m = self.mask.view(-1, 1, 1)
            hs[:, self.pos, :] = torch_where(m, st, hs[:, self.pos, :])
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return fn

    def on(self, mode, Ls, pos, states=None, mask=None):
        self.off(); self.mode, self.pos, self.states, self.mask, self.cap = mode, pos, states, mask, {}
        for L in Ls:
            self.handles.append(self.layers[L].register_forward_hook(self._fn(L)))

    def off(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def torch_where(m, a, b):
    import torch
    return torch.where(m, a, b)


def check():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    sel, skipped = select(tok); p = plan(sel)
    print("[select]", Counter(s["kind"] for s in sel), "skipped", skipped, "| max tokens", max(len(s["full"]) for s in sel))
    print("[split] discovery", len(p["disc"]), "held-out", len(p["held"]), "call recipients", len(p["call"]))
    print("[call items by goal]", Counter(sel[i]["G"] for i in p["call"]))
    print("[goal-matched call donors] no-call recipients", sum(p["donors"][r]["call_goal_matched"] for r in p["disc"] + p["held"]),
          "of", len(p["disc"]) + len(p["held"]), Counter(sel[r]["G"] for r in p["disc"] + p["held"] if p["donors"][r]["call_goal_matched"]))
    for ws in ["\n\n", "\n"]:
        assert len(tok.encode(ws, add_special_tokens=False)) == 1
    s = sel[p["disc"][0]]; print("[example] ...", repr(tok.decode(s["full"][s["c0"] - 8:s["c0"] + 12])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="retention/results/retention_cancel_patch")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    if args.check:
        return check()
    import torch
    from cd.models import load_model
    from run_cancel_tool import stop_ids
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); model, tok = load_model(MODEL); print(f"[load] {time.time()-t0:.0f}s", flush=True)
    stops = list(stop_ids(model, tok, "gemma")) + [tok.convert_tokens_to_ids("<tool_call|>")]
    TC, END = tok.convert_tokens_to_ids("<|tool_call>"), tok.convert_tokens_to_ids("<turn|>")
    NN, N1 = tok.encode("\n\n", add_special_tokens=False)[0], tok.encode("\n", add_special_tokens=False)[0]
    sel, skipped = select(tok); p = plan(sel)
    if args.smoke:
        p["disc"], p["held"], p["call"] = p["disc"][:args.smoke], p["held"][:args.smoke], p["call"][:args.smoke]
    layers = decoder_layers(model); H = Hooks(layers); dev = model.device
    POS4 = [0, 1, 2, 3]

    # 1. capture: states at the four claim-word tokens, every layer, every item (prefix up to the end of the claim words)
    need = sorted(set(p["disc"] + p["held"] + p["call"] + [d for r in p["disc"] + p["held"] + p["call"] for d in p["donors"][r]["call"] + p["donors"][r]["nocall"]]))
    S = {}
    with torch.no_grad():
        for i in need:
            s = sel[i]; ids = torch.tensor([s["full"][:s["pre_len"]]], device=dev)
            H.on("capture", range(len(layers)), list(range(s["c0"], s["pre_len"])))
            model(ids); S[i] = torch.stack([H.cap[L][0] for L in range(len(layers))])     # [48, 4, d]
    H.off()
    np.save(out / "claim_last_token_states.npy", np.stack([S[i][LAYERS, 3].float().cpu().numpy() for i in need]).astype(np.float16))
    (out / "claim_states_index.json").write_text(json.dumps([{"sel": i, **{k: sel[i][k] for k in ("item", "kind", "G", "U")}} for i in need]))
    print(f"[capture] {len(need)} items {time.time()-t0:.0f}s", flush=True)

    @torch.no_grad()
    def readout_A(r, sources):
        """sources: list of (name, sel index or None). Returns {site: {name: p_call}}, plus diagnostics."""
        s = sel[r]; full = s["full"]; c0 = s["c0"]
        rows = [(nm, d, ws) for nm, d in sources for ws in (NN, N1)]; B = len(rows)
        H.off()
        pre = model(torch.tensor([full[:c0]] * B, device=dev), use_cache=True).past_key_values
        suf = torch.tensor([full[c0:] + [ws] for _, _, ws in rows], device=dev)
        mask = torch.tensor([d is not None for _, d, _ in rows], device=dev)
        res = {}
        for site in ["ALL"] + LAYERS:
            Ls = range(len(layers)) if site == "ALL" else [site]
            st = {L: torch.stack([S[d][L] if d is not None else S[r][L] for _, d, _ in rows]) for L in Ls}
            H.on("patch", Ls, POS4, st, mask)
            lg = model(suf, past_key_values=copy.deepcopy(pre), use_cache=True).logits.float()
            H.off()
            pa = torch.softmax(lg[:, -2], -1); pb = torch.softmax(lg[:, -1], -1)
            res[str(site)] = {}
            for j in range(0, B, 2):
                nm = rows[j][0]; a = pa[j]
                res[str(site)][nm] = {"p_call": float(a[TC] + a[NN] * pb[j][TC] + a[N1] * pb[j + 1][TC]), "p_tc": float(a[TC]),
                                      "p_end": float(a[END]), "p_nn": float(a[NN]), "p_n": float(a[N1])}
        # equivalence check: unpatched full forward (no cache) vs the cached "none" row
        lg = model(torch.tensor([full + [NN], full + [N1]], device=dev)).logits.float()
        a, b1, b2 = torch.softmax(lg[0, -2], -1), torch.softmax(lg[0, -1], -1), torch.softmax(lg[1, -1], -1)
        full_p = float(a[TC] + a[NN] * b1[TC] + a[N1] * b2[TC])
        top = [tok.decode([int(t)]) for t in torch.topk(a, 3).indices]
        return res, {"full_forward_p_call": full_p, "top3_next": top}

    @torch.no_grad()
    def readout_B(r, groups, seed):
        """groups: list of (name, site, donor list). N_S samples per group from the text up to the end of the claim words."""
        s = sel[r]; ids = s["full"][:s["pre_len"]]; pos = list(range(s["c0"], s["pre_len"]))
        rows = [(nm, site, dl[j % len(dl)]) for nm, site, dl in groups for j in range(N_S)]
        enc = torch.tensor([ids] * len(rows), device=dev); outs = []
        sites = sorted({site for _, site, _ in rows}, key=str)
        for site in sites:                                   # one generate per site (hooks differ by layer set)
            idx = [k for k, row in enumerate(rows) if row[1] == site]
            Ls = range(len(layers)) if site == "ALL" else [site]
            st = {L: torch.stack([S[rows[k][2]][L] for k in idx]) for L in Ls}
            H.on("patch", Ls, pos, st, torch.ones(len(idx), dtype=torch.bool, device=dev))
            torch.manual_seed(seed + (0 if site == "ALL" else 1 + site))
            g = model.generate(enc[idx], max_new_tokens=MAX_NEW, eos_token_id=stops, pad_token_id=stops[0], **SAMPLING)
            H.off()
            outs += [(rows[k][0], rows[k][1], tok.decode(x[len(ids):], skip_special_tokens=False)) for k, x in zip(idx, g)]
        res = {}
        for nm, site, o in outs:
            res.setdefault(f"{nm}@{site}", []).append(CALL_MARK in o)
        return {k: float(np.mean(v)) for k, v in res.items()}, [o[:200] for _, _, o in outs[:: N_S]]

    recs = []

    def dump():
        (out / "records.json").write_text(json.dumps(recs, indent=1))

    def run_A(group, r):
        d = p["donors"][r]
        first, second = ("call", "nocall")
        src = [("none", None), ("self", r)] + [(f"{first}{k}", x) for k, x in enumerate(d[first])] + [(f"{second}{k}", x) for k, x in enumerate(d[second])]
        res, diag = readout_A(r, src)
        return {"group": group, "sel": r, **{k: sel[r][k] for k in ("item", "kind", "G", "U", "src", "conv")}, "donors": d, "A": res, **diag}

    for r in p["disc"]:
        recs.append(run_A("disc", r))
        if len(recs) % 20 == 0:
            dump(); print(f"[disc] {len(recs)} {time.time()-t0:.0f}s", flush=True)

    def delta(rec, site):
        a = rec["A"][str(site)]
        return np.mean([a[f"call{k}"]["p_call"] for k in range(K)]) - np.mean([a[f"nocall{k}"]["p_call"] for k in range(K)])

    prof = {L: float(np.mean([delta(x, L) for x in recs])) for L in LAYERS}
    Lsel = max(LAYERS, key=lambda L: prof[L])
    sel_info = {"discovery_profile": prof, "selected_layer": Lsel, "ALL_discovery": float(np.mean([delta(x, "ALL") for x in recs]))}
    (out / "layer_selection.json").write_text(json.dumps(sel_info, indent=1)); print(f"[select] layer {Lsel} {sel_info}", flush=True)

    for n, r in enumerate(p["held"]):
        rec = run_A("held", r); d = p["donors"][r]
        B, ex = readout_B(r, [("call", "ALL", d["call"]), ("nocall", "ALL", d["nocall"]), ("call", Lsel, d["call"]), ("nocall", Lsel, d["nocall"])],
                          SEED0 + r)
        rec["B"] = B; rec["B_examples"] = ex; recs.append(rec)
        if n % 10 == 9:
            dump(); print(f"[held] {n+1}/{len(p['held'])} {time.time()-t0:.0f}s", flush=True)
    for r in p["call"]:
        recs.append(run_A("call", r))
    dump()
    (out / "meta.json").write_text(json.dumps({"model": MODEL, "layers": LAYERS, "K": K, "n_s": N_S, "max_new": MAX_NEW, "seeds": [SPLIT_SEED, DONOR_SEED, SEED0],
                                               "selected_layer": Lsel, "skipped": skipped, "n_sel": len(sel), "smoke": args.smoke,
                                               "tokens": {"tool_call": TC, "turn_end": END, "nn": NN, "n": N1}, "elapsed_s": round(time.time() - t0)}, indent=1))
    print(f"[done] {len(recs)} records {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
