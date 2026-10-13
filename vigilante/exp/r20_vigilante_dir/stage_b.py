"""R20 stage B on the pod (transformers, Gemma 4 31B): (1) teacher-forced activations at the read position for every
item in reads.jsonl, layers 20/30/40 -> acts.npz; (2) v_vig and v_perm at layer 30 by the pre-registered formula ->
vig.pt; (3) steering episodes (every generated token) from jobs_stage_b.jsonl -> episodes_agent.jsonl (checkpointed).

  python -m vigilante.exp.r20_vigilante_dir.stage_b --reads R --jobs J --out O
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from vigilante.src.episode import dumps, episode_record, run_batch
from vigilante.src.steer_run import Steer, find_layers, gen_safe, load
from vigilante.src.templates import to_template

LAYERS = (20, 30, 40)


def extract(tok, model, layers_mod, reads, out, batch=4):
    caps = {}
    hooks = []
    for L in LAYERS:
        def mk(L):
            def h(module, args, o):
                caps[L] = (o[0] if isinstance(o, tuple) else o)[:, -1, :].float().cpu()
            return h
        hooks.append(layers_mod[L].register_forward_hook(mk(L)))
    acts = {L: [] for L in LAYERS}
    order = sorted(range(len(reads)), key=lambda i: len(reads[i]["text"]))
    t0 = time.time()
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([reads[i]["text"] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            model(**enc, logits_to_keep=1)
        for L in LAYERS:
            for j, i in enumerate(idx):
                acts[L].append((i, caps[L][j].numpy()))
        if s % (batch * 10) == 0:
            print(f"[r20b] extracted {s + len(idx)}/{len(reads)} ({time.time() - t0:.0f}s)", flush=True)
    for h in hooks:
        h.remove()
    arr = {}
    for L in LAYERS:
        a = sorted(acts[L], key=lambda x: x[0])
        arr[f"L{L}"] = np.stack([v for _, v in a]).astype(np.float16)
    np.savez_compressed(out / "acts.npz", keys=np.array([r["key"] for r in reads]), **arr)
    return {L: np.stack([v for _, v in sorted(acts[L], key=lambda x: x[0])]) for L in LAYERS}


def directions(reads, acts, L=30):
    X = acts[L]
    by = defaultdict(lambda: {True: [], False: []})
    perm = defaultdict(lambda: {True: [], False: []})
    for i, r in enumerate(reads):
        if r["kind"] == "decision" and r["mixed"]:
            by[r["cell"]][r["acted"]].append(X[i])
        elif r["kind"] == "perm":
            perm[r["cell"]][r["permitted"]].append(X[i])
    v_vig = np.mean([np.mean(d[True], 0) - np.mean(d[False], 0) for d in by.values() if d[True] and d[False]], 0)
    v_perm = np.mean([np.mean(d[True], 0) - np.mean(d[False], 0) for d in perm.values() if d[True] and d[False]], 0)
    return torch.tensor(v_vig, dtype=torch.float32), torch.tensor(v_perm, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--max-rounds", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load(a.model)
    n_layers = model.config.get_text_config().num_hidden_layers
    name, layers_mod = find_layers(model, n_layers)
    reads = [json.loads(l) for l in open(a.reads)]
    print(f"[r20b] {len(reads)} reads; hooking {name}", flush=True)
    acts = extract(tok, model, layers_mod, reads, out)
    v_vig, v_perm = directions(reads, acts)
    g = torch.Generator().manual_seed(20261013)
    r = torch.randn(v_vig.shape[0], generator=g)
    torch.save({"vig": v_vig, "perm": v_perm, "random_unit": r / r.norm(), "layer": 30}, out / "vig.pt")
    cos = float(torch.nn.functional.cosine_similarity(v_vig, v_perm, dim=0))
    print(f"[r20b] |v_vig| {v_vig.norm():.2f} |v_perm| {v_perm.norm():.2f} cos {cos:.3f}", flush=True)

    steer = Steer()
    layers_mod[30].register_forward_hook(steer.hook)
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    by_key = {j["key"]: j for j in jobs}
    D = v_vig.shape[0]

    def vec_for(job):
        s = job.get("steer")
        if not s:
            return torch.zeros(D)
        if s["dir"] == "random":
            return r / r.norm() * v_vig.norm() * s["alpha"]
        return v_vig * s["alpha"]

    def generate(items):
        outs = [None] * len(items)
        order = sorted(range(len(items)), key=lambda i: len(json.dumps(items[i]["messages"])))
        for s in range(0, len(order), a.batch):
            idx = order[s:s + a.batch]
            texts = [tok.apply_chat_template(to_template(items[i]["messages"], "gemma4"), tools=items[i]["tools"] or None,
                                             tokenize=False, add_generation_prompt=True, enable_thinking=items[i]["thinking"])
                     for i in idx]
            vecs = torch.stack([vec_for(by_key[items[i]["key"]]) for i in idx]).to(model.device)
            seed = int(hashlib.sha256("|".join(items[i]["key"] + str(items[i]["seed"]) for i in idx).encode()).hexdigest()[:8], 16)
            res, fins = gen_safe(tok, model, steer, texts, vecs, max(items[i]["max_tokens"] for i in idx), seed, False, True)
            for i, rr, f in zip(idx, res, fins):
                outs[i] = (rr, f)
            print(f"[r20b] generated {min(s + a.batch, len(order))}/{len(order)} in this round", flush=True)
        return outs

    def checkpoint(eps, rnd=None):
        tmp = out / "episodes_agent.jsonl.tmp"
        with open(tmp, "w") as f:
            for e in eps:
                f.write(dumps(episode_record(e)) + "\n")
        tmp.replace(out / "episodes_agent.jsonl")

    eps = run_batch(jobs, generate, log=lambda m: print(m, flush=True), max_rounds=a.max_rounds, on_round=checkpoint)
    checkpoint(eps)
    print("[r20b] done", flush=True)


if __name__ == "__main__":
    main()
