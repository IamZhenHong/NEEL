"""J++ lens where Gemma 4 12B ends its turn instead of calling cancel_subscription (prereg retention/prereg/retention_jpp12_v1.md).

Question: at the point where the call would be emitted, is "call the tool / cancel" in the workspace at early-mid layers and then gone
by the output (the J++ paper's retrieve-then-suppress), or never there?
Items: the frozen twins items (retention/data/cancel_twins_items.json): 224 claim_nocall (completion claim, turn ended, no call) and 25
claim_call (the same kind of claim, then a cancel_subscription call). Input = P + core: P = the prompt the agent saw for that message,
core = its message text up to the end of the claim/closing line (call and stop tokens stripped). Positions:
  E  last token of core: the model's next token is <turn|> (end the turn) or "\\n\\n" (Gemma writes "\\n\\n<|tool_call>call:cancel_...
     " after text), so this is where the call is emitted or the turn ends;
  S  last prompt token (<channel|>): the first token of the turn-2 message is chosen here;
  C  last claim-word token (" I've processed" items only, run_cancel_patch.prep).
Readout (the jpp_lens recipe): at block output l in LAYERS, logits = softcap(lm_head(final_norm(h @ J_l^T))), ranked over the vocabulary
minus the tokens with no letter or digit (Readout Filtering, the fit's exclusion list; special tokens such as <turn|> and <|tool_call>
stay in). Per position and layer: min rank of each pre-registered concept group, logits of <turn|>, "\\n\\n", <|tool_call> (unfiltered),
the unfiltered top-1, and for the J++ lens the top 10 and (at E) the rank of every ACT token. Same group ranks for the pooled lens (R-lens of
the same fit) and the logit lens. The model's own next-token probabilities at each position.
Causal (after the readout; frozen rules): ACT atoms = the N_ATOMS ACT tokens with the lowest mean J++ rank at E over the EM layers in the
claim_call items; null atoms = NULL_WORDS. At every EM layer, remove from the residual the span of the atoms' unit lens vectors
(W_U[t] @ J_l, jpp_lens readout_vectors; = jpp's Clamp with targets 0) over the message positions (msg) or at E only; read cont =
1 - P(<turn|>) at E. Matched-norm random control (msg): subtract a fixed random unit direction (orthogonal to the ACT span) scaled per
position by the norm of the ACT component the ACT clamp removes. claim_nocall items also: clamp-in at E (ACT coordinates set to the
claim_call items' mean, and the same for the null atoms), and a machinery check (remove <turn|>'s lens vector at layers 40, 44, 46 at E).
  python retention/run_jpp12.py --check                                        # laptop, tokenizer only
  python retention/run_jpp12.py --jpp /root/jpp_lens --work /root/jpp12 --out retention/results/retention_jpp12 [--limit N] [--tiny]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_cancel_tool import render  # noqa: E402
from run_cancel_patch import prep as claim_prep  # noqa: E402
import jpp_fit_gemma12 as F  # noqa: E402

MODEL, LAYERS = F.MODEL, F.LAYERS
ITEMS = HERE / "data/cancel_twins_items.json"
EM, LATE, SANITY_LAYERS, TOPK, N_ATOMS = [12, 16, 20, 24, 28, 32], [44, 46], [40, 44, 46], 10, 4
GROUPS = {"cancel": ["cancel", "cancelled", "canceled", "cancellation", "cancelling", "canceling"],
          "tool": ["call", "calling", "tool", "tools"],                                   # + the <|tool_call> token
          "subscription": ["subscription", "subscriptions"],
          "completion": ["processed", "process", "processing", "done", "completed", "complete", "finished", "confirmed"]}
NULL_WORDS = [" table", " window", " pencil", " carpet"]
POS, RAND_SEED = ("S", "C", "E"), 1212


def concept_ids(tok):
    def single(words):
        ids = set()
        for w in words:
            for f in {w, " " + w, w.capitalize(), " " + w.capitalize()}:
                e = tok.encode(f, add_special_tokens=False)
                if len(e) == 1:
                    ids.add(e[0])
        return sorted(ids)
    g = {k: single(v) for k, v in GROUPS.items()}
    sp = {"toolcall": tok.convert_tokens_to_ids("<|tool_call>"), "end": tok.convert_tokens_to_ids("<turn|>")}
    nl = tok.encode("\n\n", add_special_tokens=False); assert len(nl) == 1; sp["nl"] = nl[0]
    g["tool"] = sorted(set(g["tool"]) | {sp["toolcall"]}); g["toolcall"] = [sp["toolcall"]]; g["end"] = [sp["end"]]
    g["act"] = sorted(set(g["cancel"]) | set(g["tool"]))
    null = []
    for w in NULL_WORDS:
        e = tok.encode(w, add_special_tokens=False); assert len(e) == 1, w; null.append(e[0])
    return g, null, sp


def build(items, tok):
    cp = {p["item"]: p["pos"][-1] for p in claim_prep(items, tok)}
    out = []
    for n, it in enumerate(items):
        P = render(tok, it["hist"], it["tools"], False)
        ip = tok(P, add_special_tokens=False).input_ids; ic = tok(it["core"], add_special_tokens=False).input_ids
        ids = ip + ic; assert tok(P + it["core"], add_special_tokens=False).input_ids == ids
        out.append({"item": n, "kind": it["kind"], "G": it["G"], "U": it["U"], "ids": ids, "msg0": len(ip),
                    "pos": {"S": len(ip) - 1, "C": cp.get(n), "E": len(ids) - 1}})
    return out


def check(items):
    from collections import Counter
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    R = build(items, tok); g, null, sp = concept_ids(tok)
    print("[items]", Counter(r["kind"] for r in R), "| with C", Counter(r["kind"] for r in R if r["pos"]["C"] is not None))
    print("[len]", min(len(r["ids"]) for r in R), max(len(r["ids"]) for r in R))
    for p in POS:
        print(f"[{p}]", Counter(tok.decode([r["ids"][r["pos"][p]]]) for r in R if r["pos"][p] is not None).most_common(6))
    print("[next after S]", Counter(tok.decode([r["ids"][r["pos"]["S"] + 1]]) for r in R))
    for k, v in g.items():
        print(f"[group {k}] {len(v)}", [tok.decode([i]) for i in v])
    print("[null]", [tok.decode([i]) for i in null], "| special", sp)
    r = R[0]; print("[example E]", repr(tok.decode(r["ids"][r["pos"]["E"] - 12: r["pos"]["E"] + 1])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jpp"); ap.add_argument("--work"); ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tiny", action="store_true"); ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    items = json.loads(ITEMS.read_text())["items"]
    if args.check:
        check(items); return
    F.jpp_paths(args.jpp)
    import torch
    from workspace_lens.config import LensConfig
    from workspace_lens.lenses.base_lens import BaseLens
    from workspace_lens.lenses.logit_lens import LogitLens
    from workspace_lens.utils import readout_vectors
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); work = Path(args.work)
    t0 = time.time(); model = F.load(args.tiny); tok = model.tokenizer; dev = model.input_device
    hf = model._hf_model; decs = model.layers; nL = model.n_layers; last = nL - 1
    lenses = {"jpp": BaseLens.load(str(work / "lens.pt")), "pooled": BaseLens.load(str(work / "pooled.pt")),
              "logit": LogitLens(source_layers=LAYERS, config=LensConfig(hf_model_name=MODEL, checkpoint_name="logit", lens_type="logit", d_model=model.d_model))}
    for ln in lenses.values():
        ln.move_parameters_to_device(dev)
    assert lenses["jpp"].source_layers == LAYERS, lenses["jpp"].source_layers
    excl = torch.zeros(hf.lm_head.weight.shape[0], dtype=torch.bool, device=dev); excl[json.loads((work / "excluded_ids.json").read_text())] = True
    g, null, sp = concept_ids(tok); assert not excl[[i for v in g.values() for i in v] + null].any(), "a concept id is excluded"
    gid = {k: torch.tensor(v, device=dev) for k, v in g.items()}
    R = build(items, tok)
    if args.limit:
        R = [r for r in R if r["kind"] == "claim_call"][: args.limit] + [r for r in R if r["kind"] == "claim_nocall"][: args.limit]
    print(f"[load] {time.time() - t0:.0f}s, {len(R)} items, excluded ids {int(excl.sum())}", flush=True)

    def ranks_of(zf, ids):                        # min over ids of 1 + number of tokens with a strictly higher (filtered) logit; zf [P, V]
        return (zf[:, None, :] > zf[:, ids][:, :, None]).sum(-1).min(-1).values + 1

    cap = {}
    def mk(l):
        def h(mod, inp, o):
            x = o[0] if isinstance(o, tuple) else o
            cap[l] = x[0, cap["_pos"]].detach()
        return h
    hooks = [decs[l].register_forward_hook(mk(l)) for l in sorted(set(LAYERS) | {last})]
    hE = {}                                      # item -> {layer: residual at E} (EM layers; for the clamp-in targets)
    selfcheck = {}
    t1 = time.time()
    with open(out / "readouts.jsonl", "w") as fo, torch.no_grad():
        for c, r in enumerate(R):
            names = [p for p in POS if r["pos"][p] is not None]; idx = [r["pos"][p] for p in names]
            cap.clear(); cap["_pos"] = torch.tensor(idx, device=dev)
            lg = hf(input_ids=torch.tensor([r["ids"]], device=dev), logits_to_keep=1).logits[0, -1].float()
            own = model.unembed(cap[last]).float()                                    # [P, V]: the model's own logits at S, C, E
            if c == 0:
                selfcheck["unembed_vs_model_logits_maxabs"] = float((own[names.index("E")] - lg).abs().max())
            pr = torch.softmax(own, -1)
            rec = {"item": r["item"], "kind": r["kind"], "G": r["G"], "U": r["U"], "pos": r["pos"], "model": {}, "lens": {k: {} for k in lenses}}
            for j, p in enumerate(names):
                top = pr[j].topk(5)
                rec["model"][p] = {"p_end": float(pr[j, sp["end"]]), "p_nl": float(pr[j, sp["nl"]]), "p_toolcall": float(pr[j, sp["toolcall"]]),
                                   "logit_end": float(own[j, sp["end"]]), "logit_nl": float(own[j, sp["nl"]]), "top1": int(own[j].argmax()),
                                   "top5": [[tok.decode([int(i)]), round(float(v), 4)] for v, i in zip(top.values, top.indices)]}
            hE[r["item"]] = {l: cap[l][names.index("E")].clone() for l in EM}
            for ln_name, ln in lenses.items():
                per = {p: {} for p in names}
                for l in LAYERS:
                    z = model.unembed(ln.transport(cap[l].float(), l)).float()               # [P, V], unfiltered
                    zf = z.masked_fill(excl, float("-inf"))
                    gr = {k: ranks_of(zf, v).tolist() for k, v in gid.items()}
                    t10 = zf.topk(TOPK).indices if ln_name == "jpp" else None
                    for j, p in enumerate(names):
                        d = {"rank": {k: gr[k][j] for k in gr}, "top1_unf": int(z[j].argmax()),
                             "logit_end": float(z[j, sp["end"]]), "logit_nl": float(z[j, sp["nl"]]), "logit_toolcall": float(z[j, sp["toolcall"]])}
                        if ln_name == "jpp":
                            d["top10"] = [tok.decode([int(i)]) for i in t10[j]]
                            if p == "E":
                                d["act_id_ranks"] = ((zf[j][None, :] > zf[j, gid["act"]][:, None]).sum(-1) + 1).tolist()
                        per[p][str(l)] = d
                rec["lens"][ln_name] = per
            fo.write(json.dumps(rec) + "\n")
            if c % 25 == 24 or c == len(R) - 1:
                fo.flush(); print(f"[read] {c + 1}/{len(R)} {time.time() - t1:.0f}s", flush=True)
    for h in hooks:
        h.remove()

    # ---- causal (frozen rules) ----
    recs = [json.loads(x) for x in open(out / "readouts.jsonl")]
    call_recs = [x for x in recs if x["kind"] == "claim_call"]
    mean_rank = {tid: float(np.mean([x["lens"]["jpp"]["E"][str(l)]["act_id_ranks"][k] for x in call_recs for l in EM])) for k, tid in enumerate(g["act"])}
    atoms = sorted(mean_rank, key=lambda t: (mean_rank[t], t))[:N_ATOMS]
    print("[atoms]", [(tok.decode([t]), round(mean_rank[t], 1)) for t in atoms], "| null", [tok.decode([t]) for t in null], flush=True)
    jl = lenses["jpp"]

    def basis(tids, layers):
        Q = {}
        for l in layers:
            V = torch.stack([(lambda v: v / v.norm())(readout_vectors(jl, model, int(t), layers=[l])[l].float().to(dev)) for t in tids], 1)
            Q[l] = torch.linalg.qr(V)[0]                                                  # orthonormal basis of span(atoms) [d, K]
        return Q
    QA, QN, QT = basis(atoms, EM), basis(null, EM), basis([sp["end"]], SANITY_LAYERS)
    call_items = [x["item"] for x in call_recs]
    mA = {l: torch.stack([hE[i][l].float() @ QA[l] for i in call_items]).mean(0) for l in EM}
    mN = {l: torch.stack([hE[i][l].float() @ QN[l] for i in call_items]).mean(0) for l in EM}
    gen = torch.Generator(device="cpu").manual_seed(RAND_SEED); RD = {}
    for l in EM:                                  # one fixed random unit direction per layer, orthogonal to the ACT span
        r_ = torch.randn(model.d_model, generator=gen).to(dev); r_ = r_ - QA[l] @ (QA[l].T @ r_); RD[l] = r_ / r_.norm()
    RAND = ("rand", RD)
    EDIT, NORMS = {}, {}

    def mk_edit(l):
        def h(mod, inp, o):
            specs = EDIT.get(l)
            if not specs:
                return None
            x = o[0] if isinstance(o, tuple) else o
            for row, pos, Q, m in specs:
                v = x[row, pos].float(); c_ = v @ Q
                if m is None:                       # clamp out: remove the span
                    new = v - c_ @ Q.T
                elif isinstance(m, str):            # identity (self-check)
                    new = v
                elif isinstance(m, tuple):          # matched-norm random: minus a fixed random unit direction x |span component|
                    new = v - (c_ @ Q.T).norm(dim=-1, keepdim=True) * m[1][None, :]
                else:                               # clamp in: span coordinates set to the target
                    new = v + (m[None, :] - c_) @ Q.T
                NORMS.setdefault(row, []).append(float(((new - v).norm(dim=-1) / v.norm(dim=-1)).mean()))
                x[row, pos] = new.to(x.dtype)
            return None
        return h
    ehooks = [decs[l].register_forward_hook(mk_edit(l)) for l in sorted(set(EM) | set(SANITY_LAYERS))]

    def run_rows(r, conds):
        """conds: list of (name, [(layers, positions, Q dict, target dict / None / 'identity')]); one batched forward, readout at E."""
        EDIT.clear(); NORMS.clear()
        for row, (_, specs) in enumerate(conds):
            for layers, pos, Q, m in specs:
                for l in layers:
                    ml = m if (m is None or isinstance(m, str)) else ((m[0], m[1][l]) if isinstance(m, tuple) else m[l])
                    EDIT.setdefault(l, []).append((row, torch.tensor(pos, device=dev), Q[l], ml))
        ids = torch.tensor([r["ids"]] * len(conds), device=dev)
        with torch.no_grad():
            p = torch.softmax(hf(input_ids=ids, logits_to_keep=1).logits[:, -1].float(), -1)
        EDIT.clear()
        return {name: {"p_end": float(p[row, sp["end"]]), "p_nl": float(p[row, sp["nl"]]), "p_toolcall": float(p[row, sp["toolcall"]]),
                       "cont": 1 - float(p[row, sp["end"]]), "edit_rel_norm": float(np.mean(NORMS[row])) if NORMS.get(row) else 0.0}
                for row, (name, _) in enumerate(conds)}

    t2 = time.time(); base_read = {x["item"]: x["model"]["E"]["p_end"] for x in recs}
    with open(out / "causal.jsonl", "w") as fo:
        for c, r in enumerate(R):
            msg, E = list(range(r["msg0"], len(r["ids"]))), [r["pos"]["E"]]
            conds = [("base", []), ("outA_msg", [(EM, msg, QA, None)]), ("outN_msg", [(EM, msg, QN, None)]),
                     ("randA_msg", [(EM, msg, QA, RAND)]), ("outA_E", [(EM, E, QA, None)]), ("outN_E", [(EM, E, QN, None)])]
            if r["kind"] == "claim_nocall":
                conds += [("inA_E", [(EM, E, QA, mA)]), ("inN_E", [(EM, E, QN, mN)]), ("sanity_end", [(SANITY_LAYERS, E, QT, None)])]
            if c == 0:
                conds.append(("identity_msg", [(EM, msg, QA, "identity")]))
            res = run_rows(r, conds)
            if c == 0:
                selfcheck["identity_edit_minus_base_p_end"] = res["identity_msg"]["p_end"] - res["base"]["p_end"]   # gate in the analysis
                print("[selfcheck]", selfcheck, flush=True)
                del res["identity_msg"]
            fo.write(json.dumps({"item": r["item"], "kind": r["kind"], "G": r["G"], "base_readout_p_end": base_read[r["item"]], "rows": res}) + "\n")
            if c % 25 == 24 or c == len(R) - 1:
                fo.flush(); print(f"[causal] {c + 1}/{len(R)} {time.time() - t2:.0f}s", flush=True)
    for h in ehooks:
        h.remove()
    cosA = {str(l): (lambda V: float((V.T @ V - torch.eye(V.shape[1], device=dev)).abs().max()))(
        torch.stack([(lambda v: v / v.norm())(readout_vectors(jl, model, int(t), layers=[l])[l].float()) for t in atoms], 1)) for l in EM}
    meta = {"model": MODEL, "n_items": len(R), "limit": args.limit, "tiny": args.tiny, "layers": LAYERS, "EM": EM, "LATE": LATE,
            "sanity_layers": SANITY_LAYERS, "topk": TOPK, "groups": {k: [[i, tok.decode([i])] for i in v] for k, v in g.items()},
            "null": [[i, tok.decode([i])] for i in null], "special": sp, "atoms": [[t, tok.decode([t]), mean_rank[t]] for t in atoms],
            "act_mean_rank_call_E_EM": {tok.decode([t]): v for t, v in mean_rank.items()}, "atoms_max_offdiag_cos": cosA,
            "clampin_target_norm": {str(l): [float(mA[l].norm()), float(mN[l].norm())] for l in EM}, "selfcheck": selfcheck,
            "excluded_ids": int(excl.sum()), "s": round(time.time() - t0)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {time.time() - t0:.0f}s {selfcheck}", flush=True)


if __name__ == "__main__":
    main()
