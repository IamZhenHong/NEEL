"""SC-M4 pod run (DESIGN.md §4): steering reads on one merged model (HF, bf16).

Directions (data/dirs.npz, frozen): u_probe (N3's habit probe), u_dom (difference of means), their projections off the
answer words (u_*_perp: the span of the logit-lens and, at block 16, the J-lens read vectors of N3's UNI and REF word
tokens removed, renormalised), and 8 random unit directions. Dose k adds k x gap x u to block 16's output at every
position of the prompt's user span (items.user_span), with gap = the habit gap of the direction's parent (prep.py;
random directions use the probe's gap). k in DOSES; k = 0 is the plain model.
  self  16 prompts: softmax over the letters A / B at the last prompt token; the share on the 'unilateral' letter;
        whether the top-1 token is a letter (coherence)
  just  16 tasks: log p(penalise) - log p(refer) at D (first option tokens), the reply prefix canonical
  free  u_probe at FREE_DOSES: sampled replies (V1's decoding, max 160 tokens, seeded), the injection on the prompt
        span during prefill only
Writes reads_{cell}.jsonl (one row per prompt x direction x dose) and free_{cell}.jsonl.

  python -m self_concept.m4_inject.run --cell NW_s1 --model DIR --out DIR [--tiny]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from persona_affect.common.util import seed
from persona_affect.v1_value_hook.fmt import prompt_ids
from self_concept.common import lens as L
from self_concept.m4_inject import items as I

DATA = Path(__file__).resolve().parent / "data"
LAYER = 16
DOSES = (0.5, 1.0, 2.0)
FREE_DOSES = (0.0, 1.0)
WORDS = ["myself", "directly", "penalty", "apply", "fine", "sanction", "punish", "personally", "impose", "deal",
         "pass", "compliance", "team", "escalate", "report", "authority", "supervisor", "charge", "decide", "hand"]
JLENS = ("camilablank/workspace-lenses", "qwen3.5-9b/j-lens/lens.pt", L.CB[1])
DECODE = {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_new_tokens": 160}
T0 = time.time()


def log(*a):
    print(f"[m4-run {time.time() - T0:6.0f}s]", *a, flush=True)


def directions(R, tok, layer, tiny=False):
    z = np.load(DATA / "dirs.npz")
    meta = json.loads((DATA / "dirs.json").read_text())
    d = {"probe": z["u_probe"], "dom": z["u_dom"]}
    vecs = [R.lens_vectors(layer, L.word_ids(tok, WORDS), "logit")]
    if not tiny:
        from huggingface_hub import hf_hub_download
        ck = torch.load(hf_hub_download(*JLENS[:2], revision=JLENS[2]), map_location="cpu", weights_only=True)
        if layer in {int(k) for k in ck["J"]}:
            R.add_lens("cb", {"J": {layer: ck["J"][layer] if layer in ck["J"] else ck["J"][str(layer)]}})
            vecs.append(R.lens_vectors(layer, L.word_ids(tok, WORDS), "cb"))
    q, _ = torch.linalg.qr(torch.cat(vecs).T.float())                                    # [d, k] orthonormal
    q = q.cpu().numpy()
    out = {}
    for n, u in d.items():
        u = u.astype(np.float64)[: q.shape[0]]
        perp = u - q @ (q.T @ u)
        out[n] = (u / np.linalg.norm(u), meta["gap"][n])
        out[f"{n}_perp"] = (perp / np.linalg.norm(perp), meta["gap"][n])
    for j, u in enumerate(z["rand"]):
        out[f"rand{j}"] = (u.astype(np.float64)[: q.shape[0]] / np.linalg.norm(u[: q.shape[0]]), meta["gap"]["probe"])
    info = {"perp_rank": int(q.shape[1]), "cos_perp": {n: float(out[n][0] @ out[f"{n}_perp"][0]) for n in d}}
    return out, info


def adds(span, vec):
    v = torch.tensor(vec, dtype=torch.float32)
    return [(LAYER, p, v) for p in span]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m4 run")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, layers, norm, head, cap = L.load_model("qwen9", tiny=a.tiny, path=None if a.tiny else a.model)
    R = L.Reader(tok, model, layers, norm, head, cap)
    layer = LAYER if not a.tiny else 1
    globals()["LAYER"] = layer
    dirs, dinfo = directions(R, tok, layer, a.tiny)
    letters = {x: tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B")}
    rows = []
    conds = [("none", 0.0)] + [(n, k) for n in dirs for k in DOSES]
    for p in I.self_prompts():
        ids = prompt_ids(tok, p["messages"], None)
        span = I.user_span(tok, ids)
        uni = next(x for x, r in p["amap"].items() if r == "unilateral")
        for n, k in conds:
            add = None if n == "none" else adds(span, k * dirs[n][1] * dirs[n][0])
            _, z = R.forward(ids, [], pos=-1, add=add, logits=True)
            lp = torch.log_softmax(z, -1)
            ab = torch.log_softmax(torch.stack([lp[letters["A"]], lp[letters["B"]]]), 0)
            rows.append({"block": "self", "pid": p["pid"], "stem": p["stem"], "order": p["order"], "dir": n, "k": k,
                         "p_uni": float(ab[0 if uni == "A" else 1].exp()), "top1_letter": int(z.argmax()) in letters.values()})
    log(a.cell, "self done", len(rows))
    for p in I.just_prompts():
        p_ids = prompt_ids(tok, p["messages"], p["tools"])
        dec = I.decision(tok, p_ids, p["reply_prefix"])
        if dec is None:
            continue
        ids, first = dec
        span = I.user_span(tok, p_ids)
        for n, k in conds:
            add = None if n == "none" else adds(span, k * dirs[n][1] * dirs[n][0])
            _, z = R.forward(ids, [], pos=-1, add=add, logits=True)
            lp = torch.log_softmax(z, -1)
            rows.append({"block": "just", "pid": p["pid"], "dir": n, "k": k,
                         "lo": float(lp[first["penalise"]] - lp[first["refer"]])})
    log(a.cell, "just done", len(rows))
    (out / f"reads_{a.cell}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    free = []
    u, gap = dirs["probe"]
    for p in I.free_prompts():
        ids = prompt_ids(tok, p["messages"], None)
        span = I.user_span(tok, ids)
        for k in FREE_DOSES:
            vec = torch.tensor(k * gap * u, dtype=torch.float32)

            def hook(mod, inp, o, vec=vec, span=span):
                t = o if torch.is_tensor(o) else o[0]
                if t.shape[1] > 1 and k:
                    t[0, span] += vec.to(t.device, t.dtype)
                return o
            h = R.blocks[LAYER].register_forward_hook(hook)
            try:
                torch.manual_seed(seed(f"SC-M4|free|{a.cell}|{p['pid']}|{p['i']}|{k}"))
                with torch.no_grad():
                    g = model.generate(torch.tensor([ids], device=R.dev), do_sample=True, pad_token_id=tok.eos_token_id,
                                       **({**DECODE, "max_new_tokens": 8} if a.tiny else DECODE))
            finally:
                h.remove()
            free.append({"block": "free", "pid": p["pid"], "i": p["i"], "k": k, "cell": a.cell,
                         "key": f"SC-M4|free|{p['pid']}|{p['i']}|{k}",
                         "text": tok.decode(g[0, len(ids):], skip_special_tokens=True)})
    (out / f"free_{a.cell}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in free))
    (out / f"info_{a.cell}.json").write_text(json.dumps({"cell": a.cell, "layer": LAYER, "doses": DOSES, **dinfo,
                                                         "n_rows": len(rows), "n_free": len(free)}, indent=1))
    log("done", a.cell, len(rows), "rows,", len(free), "free")


if __name__ == "__main__":
    main()
