"""jl3 pod stages.

run_jl3: for every prompt in data/prompts_jl3.jsonl (present prompts and candidate twins, build_pairs.py)
  - greedy continuation, 50 tokens, raw text (no chat template, as the reference evals and jlens G2); for association /
    typo also 12 greedy tokens of the verification prompt (prompt + fixed question). Used to verify twins and for the
    descriptive 'said later' share. Prompts are batched only with prompts of the same length (no padding).
  - block outputs at the read position, all 64 layers; readouts with the J-lens, J++ and the logit lens at every layer
    (no J at 63: the model's output there) and, exploratory, WorkspaceBench's J-lens readout of record ('jlens_cos':
    (W_U J h) / ||J^T W_U[t]||, no final norm) at layers 20-60 step 4.
  - per (prompt, lens, layer): top-10 ids and scores (log-probs; jlens_cos: its score), and the best rank of every
    probe set of the prompt (X_c and X'_c forms, build_pairs.py).
run_wsb: WorkspaceBench hallucination bank (camilablank/workspace-bench evals/hallucination, MIT), its own token ids
  (capture_rows.json) and read sites, layers 20-60 step 4: top-10 readouts in the benchmark's readouts format, one
  file per lens. Not judged here (the pinned judge is google/gemini-3.8-flash via OpenRouter; see README)."""
import json
from pathlib import Path

import numpy as np
import torch

from jlens.e2_realism.lens_core import BAND, FINAL, best_rank, log, np_save

HERE = Path(__file__).resolve().parent
LENSES3 = ["jlens", "jpp", "logit", "jlens_cos"]
TOPK = 10
CHUNK = 128


class CosReadout:
    """WorkspaceBench's J-lens readout of record (wsbench/produce/methods.py JLens): (W_U J h) / ||J^T W_U[t]||."""

    def __init__(self, R):
        self.R, self.denom = R, {}

    def scores(self, resid, layer):
        R = self.R
        J = R.lenses["jlens"][layer].float()
        W = R.lm_head.weight
        if layer not in self.denom:
            d = torch.empty(W.shape[0], device=R.dev)
            for c0 in range(0, W.shape[0], 16384):
                d[c0:c0 + 16384] = (W[c0:c0 + 16384].float() @ J).norm(dim=1).clamp_min(1e-9)
            self.denom[layer] = d
        x = resid.to(R.dev).float() @ J.T
        return (x.to(W.dtype) @ W.T).float() / self.denom[layer]


def eos_ids(tok):
    out = []
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i not in out:
            out.append(i)
    if tok.eos_token_id is not None and tok.eos_token_id not in out:
        out.append(tok.eos_token_id)
    return out


def rank_probes(lg, rows_probes):
    """lg [n, V]; rows_probes: per row a list of id lists -> per row a list of best ranks (1-based)."""
    K = max((len(p) for p in rows_probes), default=0)
    out = [[None] * len(p) for p in rows_probes]
    for k in range(K):
        ix = [i for i, p in enumerate(rows_probes) if len(p) > k and len(p[k]) > 0]     # empty set -> None
        if not ix:
            continue
        thr = torch.stack([lg[i, rows_probes[i][k]].max() for i in ix])
        cnt = (lg[ix] > thr[:, None]).sum(-1) + 1
        for i, c in zip(ix, cnt.tolist()):
            out[i][k] = c
    return out


def run_jl3(R, out, limit=None):
    out.mkdir(parents=True, exist_ok=True)
    tok = R.tok
    rows = [json.loads(l) for l in open(HERE / "data" / "prompts_jl3.jsonl")][:limit]
    n = len(rows)
    eos = eos_ids(tok)
    # 1. greedy continuations
    gen = R.greedy([r["ids"] for r in rows], 50, eos)
    vi = [i for i, r in enumerate(rows) if r["verify_ids"]]
    vgen = R.greedy([rows[i]["verify_ids"] for i in vi], 12, eos) if vi else []
    vmap = dict(zip(vi, vgen))
    with open(out / "jl3_gen.jsonl", "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"pid": r["pid"], "gen_ids": gen[i], "gen": tok.decode(gen[i]),
                                "verify_gen": tok.decode(vmap[i]) if i in vmap else None}, ensure_ascii=False) + "\n")
    log(f"jl3 greedy done ({n} prompts, {len(vi)} verification prompts)")
    # 2. reads at the read position, all layers
    layers = list(range(64))
    resid = torch.zeros(n, 64, R.lm_head.weight.shape[1], dtype=torch.bfloat16)
    for k, r in enumerate(rows):
        acts = R.forward(r["ids"], layers, positions=[r["read_pos"]])
        for L in layers:
            resid[k, L] = acts[L][0]
        if k % 200 == 0:
            log(f"jl3 forward {k}/{n}")
    probe_names = [sorted(r["probes"]) for r in rows]
    probe_ids = [[r["probes"][p] for p in names] for r, names in zip(rows, probe_names)]
    top_ids = np.full((n, len(LENSES3), 64, TOPK), -1, np.int32)
    top_sc = np.full((n, len(LENSES3), 64, TOPK), np.nan, np.float32)
    ranks = [{p: {lens: [None] * 64 for lens in LENSES3} for p in names} for names in probe_names]
    cos = CosReadout(R)
    for li, lens in enumerate(LENSES3):
        Ls = BAND if lens == "jlens_cos" else (list(range(63)) if lens != "logit" else layers)
        for L in Ls:
            for c0 in range(0, n, CHUNK):
                ix = list(range(c0, min(c0 + CHUNK, n)))
                X = resid[ix, L]
                if lens == "jlens_cos":
                    sc = cos.scores(X, L)
                    tv, ti = sc.topk(TOPK)
                else:
                    sc = R.logits(X, L, lens)
                    tv, ti = torch.log_softmax(sc, -1).topk(TOPK)
                top_ids[ix, li, L] = ti.cpu().numpy()
                top_sc[ix, li, L] = tv.cpu().numpy()
                rk = rank_probes(sc, [probe_ids[i] for i in ix])
                for i, rr in zip(ix, rk):
                    for p, v in zip(probe_names[i], rr):
                        ranks[i][p][lens][L] = v
        log(f"jl3 readout {lens} done ({len(Ls)} layers)")
    np_save(out / "reads_jl3.npz", top_ids=top_ids, top_sc=top_sc, lenses=np.array(LENSES3))
    with open(out / "reads_jl3_ranks.jsonl", "w") as f:
        for r, rk in zip(rows, ranks):
            f.write(json.dumps({"pid": r["pid"], "ranks": rk}) + "\n")
    log(f"jl3 saved: {n} prompts")


def run_wsb(R, out, limit=None):
    out.mkdir(parents=True, exist_ok=True)
    tok = R.tok
    rows = json.load(open(HERE / "data" / "wsb_hallucination" / "capture_rows.json"))[:limit]
    cos = CosReadout(R)
    files = {lens: open(out / f"wsb_hallucination_{lens}.tokens.jsonl", "w") for lens in LENSES3}

    def disp(ids):
        return [t.replace("Ġ", " ").replace("▁", " ") if isinstance(t, str) else ""
                for t in tok.convert_ids_to_tokens(ids)]
    nsite = 0
    for k, r in enumerate(rows):
        ids, pos = r["input_ids"], [p for p in r["read_positions"] if p < len(r["input_ids"])]
        if not pos:
            continue
        acts = R.forward(ids, BAND, positions=pos)
        site_tok = disp([ids[p] for p in pos])
        for lens in LENSES3:
            for L in BAND:
                sc = cos.scores(acts[L], L) if lens == "jlens_cos" else R.logits(acts[L], L, lens)
                tv, ti = sc.topk(TOPK)
                for j, p in enumerate(pos):
                    files[lens].write(json.dumps({"id": r["id"], "layer": L, "pos": p, "token": site_tok[j],
                                                  "tokens": disp(ti[j].tolist()),
                                                  "scores": [round(v, 4) for v in tv[j].tolist()]},
                                                 ensure_ascii=False) + "\n")
        nsite += len(pos)
        if k % 30 == 0:
            log(f"wsb {k}/{len(rows)}")
    for f in files.values():
        f.close()
    log(f"wsb saved: {len(rows)} items, {nsite} sites x {len(BAND)} layers x {len(LENSES3)} lenses")
