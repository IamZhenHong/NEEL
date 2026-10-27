"""EL-H1 reads (DESIGN.md section 4; pod, project venv: transformers + torch): F2's q32 panel on sample 0 of every job.

Teacher-forced one sequence at a time (no padding), with F2's model loader and each job's own generation prompt
(reasoning on, gen.py's prompt_ids):
  pos 0  prompt end: the block output at the last prompt token;
  pos 1  own reply: the mean over the first READ_TOKENS of the job's own output (mostly reasoning);
  pos 2  fixed reply (primary): the mean over the first READ_TOKENS of the same item's `real` sample-0 output, read
         after this job's prompt, so the reply tokens are identical across kinds (real jobs: = pos 1).
Projections onto every vector at its layer: F2's directions, their placebos and Gaussian, and 20 extra seeded
Gaussians per layer (L2/L3's read.vectors). Writes OUT/proj_q32.npz: keys [N], names [V], vlayers [V], P [N, 3, V].
A job is skipped if its own or its reference output is empty. Stops (and writes what it has) at T0 + MAXMIN - 6 min.

  /root/venv/bin/python -m evalaware_lens.h1_inferred.read --path DIR --dirs DIR --out DIR
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from evalaware_lens.h1_inferred import items as I
from evalaware_lens.l2l3_cuereads.read import project, vectors

READ_TOKENS = 512
T0 = time.time()


def log(m):
    print(f"[el-h1 read {time.time() - T0:6.0f}s] {m}", flush=True)


def ref_key(job):
    return f"H1|{job['iid']}|real|s0"


def main(argv=None):
    import torch
    from evalaware_lens.f2_panel.read import DEV, load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    deadline = int(os.environ.get("T0", time.time())) + (int(os.environ.get("MAXMIN", 600)) - 6) * 60
    names, vlayers, V = vectors(Path(a.dirs) / "dirs_q32.npz")
    need = sorted(set(vlayers.tolist()))
    tok, model, blocks = load_model(a.path)
    gen = {r["key"]: r for r in map(json.loads, open(a.out / "gen.jsonl"))}
    jobs = [j for j in I.load() if j["sample"] == 0 and gen.get(j["key"], {}).get("ids")
            and gen.get(ref_key(j), {}).get("ids")]
    acc = {}

    def hook(i):
        def h(mod, inp, o):
            acc[i] = o if torch.is_tensor(o) else o[0]
        return h

    hs = [blocks[L].register_forward_hook(hook(L)) for L in need]

    def run(p, r):
        with torch.no_grad():
            model(input_ids=torch.tensor([p + r[:READ_TOKENS]], device=DEV), use_cache=False, logits_to_keep=1)
        return {L: acc[L][0].float().cpu().numpy() for L in need}

    P, keys = [], []
    try:
        for n, j in enumerate(jobs):
            if time.time() > deadline:
                log(f"deadline: stopping at {n} of {len(jobs)}")
                break
            g = gen[j["key"]]
            p = g["prompt_ids"]
            pe, own = project(run(p, g["ids"]), len(p), vlayers, V)
            fix = own if j["kind"] == "real" else project(run(p, gen[ref_key(j)]["ids"]), len(p), vlayers, V)[1]
            P.append(np.stack([pe, own, fix]))
            keys.append(j["key"])
            if n % 300 == 0:
                log(f"{n}/{len(jobs)}")
    finally:
        for h in hs:
            h.remove()
        np.savez(a.out / "proj_q32.npz", keys=np.array(keys), names=names, vlayers=vlayers,
                 P=np.asarray(P, np.float32).reshape(len(keys), 3, len(V)))
        log(f"wrote proj_q32.npz ({len(keys)} jobs x 3 reads x {len(V)} vectors)")


if __name__ == "__main__":
    main()
