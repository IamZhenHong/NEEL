"""SC-M22 reads (DESIGN.md §3), on one merged model (HF, bf16): the fitted directions at the report and at the act.

Directions, all unit length, each added at block 16 over a prompt's user span at norm k * R_NORM (k in DOSES):
  fit        fit.py's causal habit direction (fit on NZ to push it toward penalising)
  fit_perp   fit with the span of the logit-lens and J-lens read vectors of N3's UNI / REF words removed (SC-M4's
             answer-words control), renormalised
  shuf0..3   fit.py's shuffled-label directions (optimised the same way, label shuffled)
  rand0..7   seeded Gaussian directions
Reads:
  report  SC-M4 amendment 1's set (run_b.report_prompts: V1c's self_mc, N6b's in-desk habit, trait, ident); the A / B
          softmax share on 'unilateral' at the last prompt token; top-1 a letter
  just    the 16 justice tasks at the decision token (8 training, 8 held out), lo = log p(penalise) - log p(refer)
Writes reads_{cell}.jsonl and info_{cell}.json.

  python -m self_concept.m22_fitdir.read --cell NW_s1 --model DIR --dirs DIR/dirs_fit.npz --out DIR [--tiny]
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
from self_concept.m22_fitdir import fit as F
from self_concept.m4_inject import items as I4
from self_concept.m4_inject import run as R1
from self_concept.m4_inject import run_b as RB

DOSES = (1.0, 2.0)
N_RAND = 8
T0 = time.time()


def perp(R, tok, layer, u, tiny):
    vecs = [R.lens_vectors(layer, L.word_ids(tok, R1.WORDS), "logit")]
    if not tiny:
        from huggingface_hub import hf_hub_download
        ck = torch.load(hf_hub_download(*R1.JLENS[:2], revision=R1.JLENS[2]), map_location="cpu", weights_only=True)
        J = {int(k): v for k, v in ck["J"].items()}
        if layer in J:
            R.add_lens("cb", {"J": {layer: J[layer]}})
            vecs.append(R.lens_vectors(layer, L.word_ids(tok, R1.WORDS), "cb"))
    q, _ = torch.linalg.qr(torch.cat(vecs).T.float())
    q = q.cpu().numpy()
    p = u - q @ (q.T @ u)
    return p / np.linalg.norm(p), int(q.shape[1])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m22 read")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, layers, norm, head, cap = L.load_model("qwen9", tiny=a.tiny, path=None if a.tiny else a.model)
    R = L.Reader(tok, model, layers, norm, head, cap)
    layer = F.LAYER if not a.tiny else 1
    R1.LAYER = layer
    d = head.weight.shape[1]
    z = np.load(a.dirs)
    dirs = {k: z[k].astype(np.float64) / np.linalg.norm(z[k]) for k in z.files}
    dirs["fit_perp"], rank = perp(R, tok, layer, dirs["fit"], a.tiny)
    rng = np.random.default_rng(seed("SC-M22|rand") % 2**32)
    for j in range(N_RAND):
        g = rng.standard_normal(d)
        dirs[f"rand{j}"] = g / np.linalg.norm(g)
    letters = {x: tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B")}
    conds = [("none", 0.0)] + [(n, k) for n in dirs for k in DOSES]
    rows = []
    for p in RB.report_prompts():
        ids = prompt_ids(tok, p["messages"], None)
        span = I4.user_span(tok, ids)
        uni = next(x for x, r in p["amap"].items() if r == "unilateral")
        for n, k in conds:
            add = None if n == "none" else R1.adds(span, k * F.R_NORM * dirs[n])
            _, zz = R.forward(ids, [], pos=-1, add=add, logits=True)
            lp = torch.log_softmax(zz, -1)
            ab = torch.log_softmax(torch.stack([lp[letters["A"]], lp[letters["B"]]]), 0)
            rows.append({"block": "self", "family": p["family"], "cluster": p["cluster"], "pid": p["pid"],
                         "order": p["order"], "dir": n, "k": k, "p_uni": float(ab[0 if uni == "A" else 1].exp()),
                         "top1_letter": int(zz.argmax()) in letters.values()})
    print(f"[m22-read {time.time() - T0:5.0f}s] {a.cell} report done {len(rows)}", flush=True)
    for p in F.prompts(tok):
        for n, k in conds:
            vec = torch.zeros(d) if n == "none" else torch.tensor(k * F.R_NORM * dirs[n], dtype=torch.float32)
            lo = F.lo_with(model, layers, layer, p, vec.to(next(model.parameters()).device))
            rows.append({"block": "just", "pid": p["pid"], "train": p["train"], "dir": n, "k": k, "lo": float(lo)})
    (out / f"reads_{a.cell}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / f"info_{a.cell}.json").write_text(json.dumps({"cell": a.cell, "layer": layer, "perp_rank": rank,
                                                         "cos_fit_perp": float(dirs["fit"] @ dirs["fit_perp"]),
                                                         "n_rows": len(rows)}, indent=1))
    print(f"[m22-read {time.time() - T0:5.0f}s] done {a.cell} {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
