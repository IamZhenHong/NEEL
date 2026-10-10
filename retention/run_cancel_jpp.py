"""J++ lens on Gemma 4 31B's audience split (prereg retention/prereg/retention_cancel_jpp_v1.md). No sampling, no judge.

In the reporting mini, after the cancel tool returned an explicit failure and the agent's next message had nevertheless told the
subscriber "I've processed your cancellation", Gemma 4 31B answered the subscriber "Yes, cancelled" (P(Yes) 0.98-1.00) and the internal
log "No" (0.00). Here, at the answer position of those prompts (and of matched success / no-failure / no-call prompts), every layer's
residual is read with the released J++ lens for Gemma 4 31B (koayon/jpp-lenses, gemma-4-31b/lens.pt): logits = softcap(lm_head(
final_norm(h @ J_l.T))), ranked over tokens containing a letter or digit (the paper's readout filtering). Then attention knockouts (as in
retention/run_cancel_ownclaim.py, with the stock sdpa mask builder) stop the answer and everything after a span from reading it: the
tool call + failure record, the agent's earlier "done" message, or a length-matched control span from its turn-1 reply.
  uv run python retention/run_cancel_jpp.py --render-only --out <dir>
  uv run python retention/run_cancel_jpp.py --out retention/results/retention_cancel_jpp [--limit 40] [--every 2]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, yaml
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_cancel_tool as T  # noqa: E402
from run_cancel_report import rows as report_rows, ITEMS as REPORT_ITEMS, STIM  # noqa: E402
from run_cancel_audit import YES, NO, single_ids  # noqa: E402
import run_cancel_ownclaim as OC  # noqa: E402

MODEL = "google/gemma-4-31B-it"
LENS = ("koayon/jpp-lenses", "gemma-4-31b/lens.pt")
CONCEPTS = ["error", "failed", "fail", "active", "not", "still", "unable", "cancelled", "canceled", "processed", "done", "completed"]


def build_rows(st, items):
    """Report-mini prompts for the 31B: V=0; G0 and G2; success / failure x C0 / C1 and nocall; subscriber and log yes/no questions."""
    R = [r for r in report_rows(st, items) if r["V"] == 0 and r["kind"] in ("cust_yn", "log_yn")]
    out = [{**r, "ko": "none"} for r in R]
    for r in R:
        if r["O"] == "failure" and r["C"] == 1 and r["G"] == "G2":
            for ko in ("record", "message", "control"):
                out.append({**r, "ko": ko})
    return out


def spans(tok, prompt, r):
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False); off = enc["offset_mapping"]
    def toks(a, b):
        return [j for j, (s, e) in enumerate(off) if e > a and s < b]
    a = prompt.rfind("<|tool_call>call:cancel_subscription"); b = prompt.find("<tool_response|>", a) + len("<tool_response|>")
    c = prompt.find("<turn|>", b); assert 0 < a < b < c
    msg = toks(b, c); rec = toks(a, b)
    t1 = r["msgs"][2]["content"].strip(); ts = prompt.find(t1); assert 0 < ts < a
    ctl = toks(ts, ts + len(t1))[-len(msg):]
    return {"record": rec, "message": msg, "control": ctl, "ids": enc["input_ids"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--limit", type=int, default=40); ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--render-only", action="store_true"); args = ap.parse_args()
    st = yaml.safe_load(open(HERE / STIM)); items = json.load(open(REPORT_ITEMS))[: args.limit]
    R = build_rows(st, items)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    P = [T.render(tok, r["msgs"], st["tools"], False) for r in R]
    SP = {i: spans(tok, P[i], r) for i, r in enumerate(R) if r["O"] == "failure" and r["C"] == 1}
    yes_ids, no_ids = single_ids(tok, YES), single_ids(tok, NO)
    cids = {w: sorted({tok.encode(v, add_special_tokens=False)[0] for v in (w, " " + w, w.capitalize(), " " + w.capitalize())}) for w in CONCEPTS}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    print(f"[jpp] {len(R)} rows {dict(Counter((r['O'], r['C'], r['kind'], r['ko']) for r in R))}; spans: message {np.mean([len(s['message']) for s in SP.values()]):.0f} "
          f"record {np.mean([len(s['record']) for s in SP.values()]):.0f} control {np.mean([len(s['control']) for s in SP.values()]):.0f} tokens", flush=True)
    if args.render_only:
        return
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    from cd.models import load_model
    from cd.hooks import _hidden
    AttentionInterface.register("ko_sdpa", OC.ko_attention); AttentionMaskInterface.register("ko_sdpa", sdpa_mask)
    t0 = time.time(); model, tok = load_model(MODEL); dev = model.device
    lm = model.model.language_model; decs = lm.layers; nL = len(decs); norm = lm.norm; head = model.lm_head
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    ck = torch.load(hf_hub_download(*LENS), map_location="cpu", weights_only=True, mmap=True)
    layers = [l for l in sorted(ck["parameters"]["jacobians"]) if l % args.every == 0 or l == nL - 1]
    J = {l: ck["parameters"]["jacobians"][l].to(dev, torch.bfloat16) for l in layers}; del ck
    print(f"[load] {time.time()-t0:.0f}s layers {nL} lens layers {len(layers)} softcap {cap}", flush=True)
    vocab_ok = torch.tensor([any(ch.isalnum() for ch in tok.decode([i])) for i in range(head.weight.shape[0])], device=dev)
    yi, ni = list(yes_ids.values()), list(no_ids.values())
    cap_h = {}
    def mk(l):
        def h(mod, inp, o):
            cap_h[l] = _hidden(o)[0, -1].detach()
        return h
    hooks = [decs[l].register_forward_hook(mk(l)) for l in layers]

    @torch.no_grad()
    def readout(ids, spec=None):
        OC.KO["spec"] = spec; cap_h.clear()
        lg = model(torch.tensor([ids], device=dev), logits_to_keep=1).logits[0, -1].float(); OC.KO["spec"] = None
        pr = torch.softmax(lg, -1); py, pn = float(pr[yi].sum()), float(pr[ni].sum())
        res = {"p": py / (py + pn) if py + pn > 0 else None, "mass": py + pn, "lens": {}}
        for l in layers:
            x = (cap_h[l].to(torch.bfloat16) @ J[l].T)
            z = head(norm(x[None]))[0].float()
            if cap:
                z = cap * torch.tanh(z / cap)
            zf = z.masked_fill(~vocab_ok, float("-inf")); order = zf.argsort(descending=True); rank = torch.empty_like(order); rank[order] = torch.arange(len(order), device=dev)
            ly, ln = float(z[yi].max()), float(z[ni].max())
            res["lens"][l] = {"yes_minus_no": ly - ln, "rank_yes": int(rank[yi].min()), "rank_no": int(rank[ni].min()),
                              "concept_rank": {w: int(rank[c].min()) for w, c in cids.items()},
                              "top10": [tok.decode([int(t)]) for t in order[:10]]}
        return res

    # self-checks: custom attention reproduces stock sdpa; an empty knockout changes nothing; the lens at the last layer agrees with the output
    ids0 = tok(P[0], add_special_tokens=False).input_ids; stock = readout(ids0)["p"]
    model.set_attn_implementation("ko_sdpa"); base = readout(ids0); n0 = len(ids0); z = torch.zeros(n0, dtype=torch.bool, device=dev)
    empty = readout(ids0, {"layers": set(range(nL)), "q": z, "k": z})["p"]
    chk = {"stock": stock, "custom": base["p"], "empty_ko": empty, "last_layer_lens_yes_minus_no": base["lens"][layers[-1]]["yes_minus_no"]}
    print("[selfcheck]", chk, flush=True); assert abs(base["p"] - empty) < 1e-3 and abs(base["p"] - stock) < 0.05
    meta = lambda r: {k: v for k, v in r.items() if k not in ("msgs", "base", "tcs", "tms")}  # noqa: E731
    with open(out / "readouts.jsonl", "w") as fo:
        for c, (i, r) in enumerate(zip(range(len(R)), R)):
            ids = SP[i]["ids"] if i in SP else tok(P[i], add_special_tokens=False).input_ids; spec = None
            if r["ko"] != "none":
                n = len(ids); k = torch.zeros(n, dtype=torch.bool, device=dev); k[SP[i][r["ko"]]] = True
                q = torch.zeros(n, dtype=torch.bool, device=dev); q[max(SP[i][r["ko"]]) + 1:] = True
                spec = {"layers": set(range(nL)), "q": q, "k": k}
            fo.write(json.dumps({"i": i, **meta(r), **readout(ids, spec)}) + "\n")
            if c % 100 == 99:
                fo.flush(); print(f"[read] {c+1}/{len(R)} {time.time()-t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    (out / "meta.json").write_text(json.dumps({"model": MODEL, "lens": LENS, "lens_fitted_on": "google/gemma-4-31B (per the lens config)", "layers": layers,
                                               "n_rows": len(R), "selfcheck": chk, "s": round(time.time() - t0)}, indent=1))
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
