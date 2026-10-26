"""EL-L2 + EL-L3 reads (DESIGN.md section 4; pod, project venv: transformers + torch). One model per process.

For every job, teacher-forced one sequence at a time (no padding; SC-M34's Qwen3.5 parity lesson), with F2's model
loader and thinking-off prompt:
  pos 0  prompt end: the block output at the last prompt token (the assistant turn's start);
  pos 1  own reply: the mean block output over the job's own reply tokens;
  pos 2  fixed reply: the mean over the tokens of the same task's NEUTRAL-cue reply (same wording; observer jobs use
         wording 0) read after this job's prompt, so the reply is identical across cue kinds (neutral jobs: = pos 1).
Each read is projected on the pod onto every vector at its layer: F2's 10 directions, their 2 on-manifold placebos
each, their Gaussian, and 20 extra seeded Gaussian directions per layer read. Writes OUT/proj_{model}.npz: keys [N],
names [V], vlayers [V], P [N, 3, V] float32.

  /root/venv/bin/python -m evalaware_lens.l2l3_cuereads.read --model-key q27 --path DIR --dirs DIR --out DIR
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from evalaware_lens.l2l3_cuereads import items as I

N_EXTRA = 20
T0 = time.time()


def log(m):
    print(f"[el-l2l3 read {time.time() - T0:6.0f}s] {m}", flush=True)


def vectors(dirs_npz):
    """F2's vectors plus the extra Gaussians: (names [V], layers [V], unit vectors [V, d])."""
    z = np.load(dirs_npz, allow_pickle=True)
    names, layers, vecs = [], [], []
    for i, n in enumerate(z["names"]):
        L = int(z["layers"][i])
        names += [f"dir:{n}", f"plac:{n}:0", f"plac:{n}:1", f"gauss:{n}"]
        layers += [L] * 4
        vecs += [z["u"][i], z["placebos"][i][0], z["placebos"][i][1], z["gauss"][i]]
    d = z["u"].shape[1]
    for L in sorted(set(int(x) for x in z["layers"])):
        g = np.random.default_rng(1000 + L).standard_normal((N_EXTRA, d))
        names += [f"extra:{L}:{k}" for k in range(N_EXTRA)]
        layers += [L] * N_EXTRA
        vecs += list(g)
    V = np.asarray(vecs, dtype=np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return np.array(names), np.array(layers), V


def project(acts, plen, layers, V):
    """acts: {layer: [T, d] float32 for prompt + reply}; -> (prompt-end [V], reply mean [V])."""
    pe, rm = np.zeros(len(V), np.float32), np.zeros(len(V), np.float32)
    for L in set(layers.tolist()):
        m = layers == L
        a = acts[L]
        pe[m] = V[m] @ a[plen - 1]
        rm[m] = V[m] @ a[plen:].mean(0)
    return pe, rm


def fixed_key(job):
    return f"L2L3|{job['tid']}|neutral|{job['w'] if not job['kind'].startswith('obs_') else 0}"


def main():
    import torch
    from evalaware_lens.f2_panel.read import DEV, load_model, prompt_ids
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    names, vlayers, V = vectors(Path(a.dirs) / f"dirs_{a.model_key}.npz")
    need = sorted(set(vlayers.tolist()))
    tok, model, blocks = load_model(a.path)
    gen = {r["key"]: r for r in map(json.loads, open(out / f"gen_{a.model_key}.jsonl"))}
    jobs = [j for j in I.load() if j["key"] in gen]
    acc = {}

    def hook(i):
        def h(mod, inp, o):
            acc[i] = o if torch.is_tensor(o) else o[0]
        return h

    hs = [blocks[L].register_forward_hook(hook(L)) for L in need]

    def run(p, r):
        r = (r or [tok.eos_token_id])[:512]
        with torch.no_grad():
            model(input_ids=torch.tensor([p + r], device=DEV), use_cache=False, logits_to_keep=1)
        return {L: acc[L][0].float().cpu().numpy() for L in need}

    P = np.zeros((len(jobs), 3, len(V)), np.float32)
    try:
        for n, j in enumerate(jobs):
            p = prompt_ids(tok, I.messages(j))
            pe, own = project(run(p, gen[j["key"]]["reply_ids"]), len(p), vlayers, V)
            if j["kind"] == "neutral":
                fix = own
            else:
                _, fix = project(run(p, gen[fixed_key(j)]["reply_ids"]), len(p), vlayers, V)
            P[n] = np.stack([pe, own, fix])
            if n % 300 == 0:
                log(f"{a.model_key}: {n}/{len(jobs)}")
    finally:
        for h in hs:
            h.remove()
    np.savez(out / f"proj_{a.model_key}.npz", keys=np.array([j["key"] for j in jobs]), names=names, vlayers=vlayers, P=P)
    log(f"{a.model_key}: wrote proj_{a.model_key}.npz ({len(jobs)} jobs x 3 reads x {len(V)} vectors)")


if __name__ == "__main__":
    main()
