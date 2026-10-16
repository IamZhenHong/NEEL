"""T6 sigma and the T6p placebo direction (pilot pod, HF).

From R12's answers (data/r12_axis_set.jsonl: the 40 default and 400 role answers, thinking off, the r18 axis's own
data), read hidden state 20 at every answer token (teacher forced, no hook):
  sigma        sd of the per-token projections of the 40 default answers on the r18 axis (hidden state 20); the same
               projections r18 took tau from. Also recorded: sd of the per-answer mean projections.
  check        cos(recomputed diff-of-means, file axis) (expect ~1).
  placebo      labels shuffled (40 "default" drawn at random from the 440 answers, seed 64), diff-of-means of the
               per-answer means, orthogonalised to the axis, unit; applied at T6's alpha (same norm).
  python -m vibes.e64_layer_map.prep_vectors --out DIR [--model ...] [--axis ...] [--limit N]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vibes.e64_layer_map import conds as C
from vibes.e64_layer_map.hf_model import AXIS, H_STEER, HFModel

DATA = C.REPO / "vibes" / "e64_layer_map" / "data" / "r12_axis_set.jsonl"
PLACEBO_SEED = 64


def compute(m, out, limit=None):
    """Read the states with HFModel `m` (no hook active) and write out/vectors.pt + vectors_stats.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(DATA)]
    if limit:
        rows = [r for r in rows if r["label"] == "default"][:limit] + [r for r in rows if r["label"] == "role"][:limit]
    states = m.answer_states([r["prompt"] for r in rows], [r["completion"] for r in rows])
    ax = m.axis[H_STEER].cpu()
    lab = np.array([r["label"] == "default" for r in rows])
    means = torch.stack([s.mean(0) for s in states])                       # [n, D]
    tok_proj = torch.cat([s @ ax for s, d in zip(states, lab) if d])       # per-token, default answers
    ans_proj = means[torch.tensor(lab)] @ ax
    sigma = float(tok_proj.std())
    real = means[torch.tensor(lab)].mean(0) - means[torch.tensor(~lab)].mean(0)
    rng = np.random.default_rng(PLACEBO_SEED)
    perm = rng.permutation(lab)
    sh = means[torch.tensor(perm)].mean(0) - means[torch.tensor(~perm)].mean(0)
    orth = sh - (sh @ ax) * ax
    placebo = orth / orth.norm()
    stats = {"n_default": int(lab.sum()), "n_role": int((~lab).sum()), "n_default_tokens": int(tok_proj.numel()),
             "sigma_token": sigma, "sigma_answer": float(ans_proj.std()),
             "mean_default_token_proj": float(tok_proj.mean()), "tau20_file": m.tau[H_STEER],
             "cos_recomputed_vs_file_axis": float(torch.nn.functional.cosine_similarity(real, ax, dim=0)),
             "norm_real_diff": float(real.norm()), "norm_shuffled_diff": float(sh.norm()),
             "cos_shuffled_vs_axis_before_orth": float(torch.nn.functional.cosine_similarity(sh, ax, dim=0)),
             "cos_placebo_vs_axis": float(placebo @ ax), "placebo_seed": PLACEBO_SEED,
             "shuffled_default_overlap": int((perm & lab).sum()),
             "mean_resid_norm_default_tokens": float(torch.cat([s for s, d in zip(states, lab) if d]).norm(dim=-1).mean())}
    torch.save({"axis20": ax, "placebo": placebo, "sigma": sigma, "stats": stats}, out / "vectors.pt")
    (out / "vectors_stats.json").write_text(json.dumps(stats, indent=1))
    print("[e64-prep] " + json.dumps(stats), flush=True)
    return torch.load(out / "vectors.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--axis", default=str(AXIS))
    ap.add_argument("--limit", type=int, default=None, help="tiny: first N default and N role answers")
    a = ap.parse_args()
    compute(HFModel(a.model, a.axis), a.out, a.limit)


if __name__ == "__main__":
    main()
