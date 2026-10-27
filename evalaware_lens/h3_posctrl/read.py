"""EL-H3 reads (DESIGN.md section 4; pod, project venv: transformers + torch). One model per process.

L2/L3's read, unchanged in what it computes (evalaware_lens/l2l3_cuereads/read.py: its vector set and projection, F2's
loader and thinking-off prompt, one sequence at a time, no padding):
  pos 0  prompt end: the block output at the last prompt token;
  pos 1  own reply: the mean over the job's own reply tokens (neutral / eval_safety: L2/L3's reply; inducers: gen.py's);
  pos 2  fixed reply: the mean over L2/L3's NEUTRAL reply for the same task and wording, read after this job's prompt
         (neutral jobs: = pos 1).
Jobs run in priority order (neutral, the strong inducers and French, the mild inducers, then the eval_safety anchor),
and OUT/proj_{model}.npz is rewritten every 500 jobs and when the deadline (--deadline, epoch seconds) is 4 minutes
away, so a cut-off pod still returns every finished read. Keys, names, vlayers, P [N, 3, V] float32, as in L2/L3.

  /root/venv/bin/python -m evalaware_lens.h3_posctrl.read --model-key q27 --path DIR --dirs DIR --out DIR [--deadline T]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from evalaware_lens.h3_posctrl import items as I
from evalaware_lens.l2l3_cuereads.read import project, vectors

T0 = time.time()
SAVE_EVERY = 500


def log(m):
    print(f"[el-h3 read {time.time() - T0:6.0f}s] {m}", flush=True)


def order(jobs):
    def rank(j):
        k = j["kind"]
        return (0 if k == "neutral" else 1 if (k.endswith("_strong") or k == "french_mild") else 2 if k.endswith("_mild")
                else 3)
    return sorted(jobs, key=lambda j: (rank(j), j["kind"], j["ti"], j["w"]))


def save(out, key, jobs, P, n, names, vlayers):
    np.savez(Path(out) / f"proj_{key}.npz", keys=np.array([j["key"] for j in jobs[:n]]), names=names, vlayers=vlayers,
             P=P[:n])


def main():
    import torch
    from evalaware_lens.f2_panel.read import DEV, load_model, prompt_ids
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, default=float("inf"))
    a = ap.parse_args()
    out = Path(a.out)
    names, vlayers, V = vectors(Path(a.dirs) / f"dirs_{a.model_key}.npz")
    need = sorted(set(vlayers.tolist()))
    tok, model, blocks = load_model(a.path)
    l2l3 = I.load_replies()[a.model_key]
    own_gen = {r["key"]: r["reply_ids"] for r in map(json.loads, open(out / f"gen_{a.model_key}.jsonl"))}
    jobs = order([j for j in I.load() if j["own_from"] or j["key"] in own_gen])
    log(f"{a.model_key}: {len(jobs)} jobs ({sum(1 for j in jobs if not j['own_from'])} with generated replies)")
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
    n = 0
    try:
        for n, j in enumerate(jobs):
            if time.time() > a.deadline - 240:
                log(f"{a.model_key}: deadline near, stopping at {n}/{len(jobs)}")
                break
            p = prompt_ids(tok, I.messages(j))
            own_ids = l2l3[j["own_from"]] if j["own_from"] else own_gen[j["key"]]
            pe, own = project(run(p, own_ids), len(p), vlayers, V)
            if j["kind"] == "neutral":
                fix = own
            else:
                _, fix = project(run(p, l2l3[j["fixed_from"]]), len(p), vlayers, V)
            P[n] = np.stack([pe, own, fix])
            if n % 300 == 0:
                log(f"{a.model_key}: {n}/{len(jobs)} ({j['kind']})")
            if n and n % SAVE_EVERY == 0:
                save(out, a.model_key, jobs, P, n + 1, names, vlayers)
        else:
            n = len(jobs)
    finally:   # also on an exception: every finished read is kept
        for h in hs:
            h.remove()
        save(out, a.model_key, jobs, P, n, names, vlayers)
        log(f"{a.model_key}: wrote proj_{a.model_key}.npz ({n} of {len(jobs)} jobs x 3 reads x {len(V)} vectors)")


if __name__ == "__main__":
    main()
