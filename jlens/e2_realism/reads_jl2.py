"""jl2 reads (pod): block outputs at layers 20, 24, ..., 60 and 63 at the read positions of data/prompts.jsonl
(build_prompts.py), read with the J-lens, J++ and the logit lens.
Saved per (prompt, lens, layer, position): top-20 token ids and log-probs, log-mass and best rank of EVAL and REAL.
Random-set log-masses (200 EVAL-sized + 200 REAL-sized) at 'last' and 'user_last', and their mean over the note
tokens (one value per prompt), so the floor is available for every position the analysis uses."""
import json
from pathlib import Path

import numpy as np
import torch

from jlens.e2_realism.lens_core import BAND, FINAL, LENSES, best_rank, log, np_save

HERE = Path(__file__).resolve().parent
TOPK = 20
CHUNK = 128


def run_jl2(R, out, limit=None):
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")][:limit]
    sets = json.load(open(HERE / "data" / "tokensets.json"))
    named = ["EVAL", "REAL"]
    rand = sorted(k for k in sets if k.startswith("rand"))
    layers = BAND + [FINAL]
    pos_lists = [[r["pos"]["last"], r["pos"]["user_last"]] + r["pos"]["note"] for r in rows]
    P = max(len(p) for p in pos_lists)
    n, nl, d = len(rows), len(layers), R.lm_head.weight.shape[1]
    resid = torch.zeros(n, nl, P, d, dtype=torch.bfloat16)
    for k, (r, pos) in enumerate(zip(rows, pos_lists)):
        acts = R.forward(r["ids"], layers, positions=pos)
        for j, L in enumerate(layers):
            resid[k, j, : len(pos)] = acts[L]
        if k % 50 == 0:
            log(f"jl2 forward {k}/{n} ({len(r['ids'])} tokens, {len(pos)} positions)")
    valid = np.array([[p < len(pl) for p in range(P)] for pl in pos_lists])
    n_note = valid[:, 2:].sum(1)
    top_ids = np.full((n, len(LENSES), nl, P, TOPK), -1, np.int32)
    top_lp = np.zeros((n, len(LENSES), nl, P, TOPK), np.float16)
    mass = np.full((n, len(LENSES), nl, P, len(named)), np.nan, np.float32)
    best = np.zeros((n, len(LENSES), nl, P, len(named)), np.int32)
    rmass = np.full((n, len(LENSES), nl, 2, len(rand)), np.nan, np.float32)        # at last, user_last
    rnote = np.zeros((n, len(LENSES), nl, len(rand)), np.float64)                   # sum over note tokens -> mean
    nidx = [torch.tensor(sets[s], device=R.dev) for s in named]
    ridx = torch.tensor([sets[s] for s in rand if s.startswith("randE")], device=R.dev)
    ridx_r = torch.tensor([sets[s] for s in rand if s.startswith("randR")], device=R.dev)
    assert [s[:5] for s in rand] == ["randE"] * len(ridx) + ["randR"] * len(ridx_r)
    for li, lens in enumerate(LENSES):
        for j, L in enumerate(layers):
            X = resid[:, j].reshape(n * P, d)
            for c0 in range(0, n * P, CHUNK):
                ix = np.arange(c0, min(c0 + CHUNK, n * P))
                a, b = ix // P, ix % P
                keep = valid[a, b]
                if not keep.any():
                    continue
                ix, a, b = ix[keep], a[keep], b[keep]
                lg = R.logits(X[torch.from_numpy(ix)], L, lens)
                lp = torch.log_softmax(lg, -1)
                tv, ti = lp.topk(TOPK)
                top_ids[a, li, j, b] = ti.cpu().numpy()
                top_lp[a, li, j, b] = tv.cpu().numpy()
                mass[a, li, j, b] = torch.stack([torch.logsumexp(lp[:, s], -1) for s in nidx], 1).cpu().numpy()
                best[a, li, j, b] = torch.stack([best_rank(lg, s) for s in nidx], 1).cpu().numpy()
                rm = torch.cat([torch.logsumexp(lp[:, ridx], -1), torch.logsumexp(lp[:, ridx_r], -1)], 1).cpu().numpy()
                head = b < 2
                rmass[a[head], li, j, b[head]] = rm[head]
                np.add.at(rnote, (a[~head], li, j), rm[~head])
            log(f"jl2 readout {lens} L{L} done")
    rnote = (rnote / np.maximum(n_note, 1)[:, None, None, None]).astype(np.float32)
    rnote[n_note == 0] = np.nan
    np_save(out / "reads_jl2.npz", top_ids=top_ids, top_lp=top_lp, mass=mass, best=best, rmass=rmass, rnote=rnote,
            valid=valid, layers=np.array(layers), lenses=np.array(LENSES), named=np.array(named),
            rand=np.array(rand))
    with open(out / "reads_jl2_meta.jsonl", "w") as f:
        for r, pl in zip(rows, pos_lists):
            f.write(json.dumps({k: v for k, v in r.items() if k != "ids"} | {"read_pos": pl, "n_tokens": len(r["ids"])},
                               ensure_ascii=False) + "\n")
    log(f"jl2 saved: {n} prompts, P={P}")
