"""R20 stage B2 on the pod (transformers, rules in README.md): (1) teacher-forced residuals at the read token of every
item in the reads file, at the listed layers -> acts.npz; (2) v_vig / v_auth / v_perm at the primary layer by the
pre-registered formulas -> dirs.pt; (3) decision-point steering: for every steer item, generate the decision turn
once per arm listed in the item ([name, dir, alpha, random, mode]; vector = sign * alpha * v, or a random unit
direction at the norm of alpha * |v|), adding it at the primary layer at the position that produces each generated
token (the last prompt position, then every new token). mode "all" steers the whole turn; "thought" only while the
turn is inside Gemma's thought block (R17 correction) -> steer_out.jsonl, appended per batch so a restart resumes.
--extra-dirs adds named directions from another dirs.pt (R17: batman, p4).

  python -m vigilante.exp.r20_vigilante_dir.pod_b2 --model M --fmt gemma4|qwen3 --reads R --steer S --out O \
      --layers 20 30 40 --primary 30 [--batch 6]
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from vigilante.src.steer_run import find_layers, load

CH_OPEN, CH_CLOSE = "<|channel>", "<channel|>"
STRIP = {"gemma4": ("<turn|>", "<eos>", "<|tool_response>", "<pad>"), "qwen3": ("<|im_end|>", "<|endoftext|>")}


def eos_ids(tok, fmt):
    if fmt == "gemma4":
        return [1, 106, 50]
    return [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]


class Add:
    """Adds gate[b] * vecs[b] to the residual at the last position of each forward: the last prompt token during
    prefill (left padding), then each newly generated token. vecs=None leaves the model untouched."""

    def __init__(self):
        self.vecs = None
        self.gate = None

    def hook(self, module, args, out):
        if self.vecs is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, -1, :] += self.gate.to(h.dtype)[:, None] * self.vecs.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def gate_processor(add, thought_rows, open_id, close_id):
    """Thought-mode rows: gate opens on a generated <|channel>, closes on <channel|> (one token behind)."""
    from transformers import LogitsProcessor

    class Gate(LogitsProcessor):
        def __call__(self, input_ids, scores):
            last = input_ids[:, -1]
            g = torch.where(last == open_id, torch.ones_like(add.gate),
                            torch.where(last == close_id, torch.zeros_like(add.gate), add.gate))
            add.gate = torch.where(thought_rows, g, add.gate)
            return scores
    return Gate()


def extract(tok, model, layers_mod, layers, reads, out, batch):
    caps, hooks = {}, []
    for L in layers:
        def mk(L):
            def h(module, args, o):
                caps[L] = (o[0] if isinstance(o, tuple) else o)[:, -1, :].float().cpu()
            return h
        hooks.append(layers_mod[L].register_forward_hook(mk(L)))
    acts = {L: [None] * len(reads) for L in layers}
    order = sorted(range(len(reads)), key=lambda i: len(reads[i]["text"]))
    t0 = time.time()
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([reads[i]["text"] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            model(**enc, logits_to_keep=1)
        for L in layers:
            for j, i in enumerate(idx):
                acts[L][i] = caps[L][j].numpy()
        print(f"[b2] extracted {s + len(idx)}/{len(reads)} ({time.time() - t0:.0f}s)", flush=True)
    for h in hooks:
        h.remove()
    arr = {f"L{L}": np.stack(acts[L]).astype(np.float32) for L in layers}
    np.savez_compressed(out / "acts.npz", keys=np.array([r["key"] for r in reads]), **arr)
    return arr


def directions(reads, X):
    """Pre-registered: mean over cells of mean(positive) - mean(negative), at the decision read."""
    g = defaultdict(lambda: {True: [], False: []})
    for i, r in enumerate(reads):
        if r["pos"] != "dec":
            continue
        if r["role"] in ("vig", "auth"):
            g[(r["role"], r["cell"])][r["acted"]].append(X[i])
        elif r["role"] == "perm":
            g[("perm", r["cell"])][r["permitted"]].append(X[i])
    dirs = {}
    for name in ("vig", "auth", "perm"):
        ds = [np.mean(d[True], 0) - np.mean(d[False], 0) for (n, _), d in g.items() if n == name and d[True] and d[False]]
        if ds:
            dirs[name] = torch.tensor(np.mean(ds, 0), dtype=torch.float32)
    return dirs


def generate(tok, model, add, texts, vecs, modes, max_new, seed, fmt):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    thought = torch.tensor([m == "thought" for m in modes], dtype=torch.bool, device=model.device)
    # thought rows start open when the prompt already opened the thought (Gemma turns after a tool reply)
    add.gate = torch.tensor([m == "all" or (m == "thought" and t.endswith(CH_OPEN + "thought\n"))
                             for m, t in zip(modes, texts)], dtype=torch.bool, device=model.device)
    add.vecs = vecs
    procs = [gate_processor(add, thought, tok.convert_tokens_to_ids(CH_OPEN), tok.convert_tokens_to_ids(CH_CLOSE))] \
        if bool(thought.any()) else []
    torch.manual_seed(seed)
    try:
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.95, top_k=20,
                                 eos_token_id=eos_ids(tok, fmt), pad_token_id=tok.pad_token_id, logits_processor=procs)
    finally:
        add.vecs = None
    res = []
    eos = set(eos_ids(tok, fmt))
    for row in out[:, enc["input_ids"].shape[1]:]:
        ids = row.tolist()
        while ids and ids[-1] == tok.pad_token_id:
            ids.pop()
        fin = "stop" if ids and ids[-1] in eos else "length"
        t = tok.decode(ids, skip_special_tokens=False)
        for s in STRIP[fmt]:
            t = t.replace(s, "")
        res.append((t, fin))
    return res


def generate_safe(tok, model, add, texts, vecs, modes, max_new, seed, fmt):
    try:
        return generate(tok, model, add, texts, vecs, modes, max_new, seed, fmt)
    except torch.OutOfMemoryError:
        pass
    # retry outside the except block: inside it the traceback keeps the failed batch's tensors alive, so every
    # retry ran out of memory too (287 rows lost on 10-13)
    import gc
    add.vecs = None
    gc.collect()
    torch.cuda.empty_cache()
    if len(texts) == 1:
        print("[b2] OOM on a single sequence; recorded as oom", flush=True)
        return [("", "oom")]
    h = len(texts) // 2
    print(f"[b2] OOM at batch {len(texts)}; splitting", flush=True)
    return (generate_safe(tok, model, add, texts[:h], vecs[:h], modes[:h], max_new, seed, fmt)
            + generate_safe(tok, model, add, texts[h:], vecs[h:], modes[h:], max_new, seed + 1, fmt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fmt", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--steer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--primary", type=int, required=True)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--read-batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N steer items (smoke)")
    ap.add_argument("--extra-dirs", default=None, help="dirs.pt with more named directions (R17: batman, p4)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load(a.model)
    n_layers = model.config.get_text_config().num_hidden_layers
    name, layers_mod = find_layers(model, n_layers)
    reads = [json.loads(l) for l in open(a.reads)]
    print(f"[b2] {a.model}: {n_layers} layers ({name}); {len(reads)} reads", flush=True)

    if (out / "dirs.pt").exists():
        saved = torch.load(out / "dirs.pt")
        dirs, rand = saved["dirs"], saved["random_unit"]
        print("[b2] reusing dirs.pt", flush=True)
    else:
        arr = extract(tok, model, layers_mod, a.layers, reads, out, a.read_batch)
        dirs = directions(reads, arr[f"L{a.primary}"])
        g = torch.Generator().manual_seed(20261013)
        r = torch.randn(arr[f"L{a.primary}"].shape[1], generator=g)
        rand = r / r.norm()
        torch.save({"dirs": dirs, "random_unit": rand, "layer": a.primary}, out / "dirs.pt")
    if a.extra_dirs:
        ex = torch.load(a.extra_dirs)
        for k, v in ex.items():          # R17: batman, p4; placebo file: <dir>_placebo
            if torch.is_tensor(v) and v.ndim == 1 and k != "random_unit":
                dirs[k] = v.float().cpu()
    for k, v in dirs.items():
        print(f"[b2] |v_{k}| = {v.norm():.2f}", flush=True)
    if "vig" in dirs and "perm" in dirs:
        print(f"[b2] cos(v_vig, v_perm) = {torch.nn.functional.cosine_similarity(dirs['vig'], dirs['perm'], dim=0):.3f}",
              flush=True)

    items = [json.loads(l) for l in open(a.steer) if l.strip()]
    items = [it for it in items if all(arm[1] is None or arm[1] in dirs for arm in it["arms"])][:a.limit]
    done = set()
    if (out / "steer_out.jsonl").exists():
        done = {(d["id"], d["arm"]) for d in map(json.loads, open(out / "steer_out.jsonl"))}
    work = [(it, arm) for it in items for arm in it["arms"] if (it["id"], arm[0]) not in done]
    cap = lambda w: w[1][5] if len(w[1]) > 5 else w[0]["max_tokens"]     # optional per-arm token cap
    work.sort(key=lambda w: (cap(w), len(w[0]["text"])))
    print(f"[b2] steering: {len(items)} items; {len(work)} generations to do", flush=True)
    add = Add()
    layers_mod[a.primary].register_forward_hook(add.hook)
    D = rand.shape[0]
    t0 = time.time()
    with open(out / "steer_out.jsonl", "a") as f:
        for s in range(0, len(work), a.batch):
            chunk = work[s:s + a.batch]
            vecs = []
            for it, (arm, dname, alpha, rnd, mode, *_) in chunk:
                if not alpha or dname is None:
                    vecs.append(torch.zeros(D))
                    continue
                v = dirs[dname]
                vecs.append((rand * v.norm() if rnd else v) * alpha * it["sign"])
            vecs = torch.stack(vecs).to(model.device)
            seed = int(hashlib.sha256("|".join(f"{w[0]['id']}:{w[1][0]}" for w in chunk).encode()).hexdigest()[:8], 16)
            res = generate_safe(tok, model, add, [w[0]["text"] for w in chunk], vecs, [w[1][4] for w in chunk],
                                max(cap(w) for w in chunk), seed, a.fmt)
            for (it, (arm, dname, alpha, rnd, mode, *_)), (txt, fin) in zip(chunk, res):
                f.write(json.dumps({"id": it["id"], "group": it["group"], "sign": it["sign"], "arm": arm, "dir": dname,
                                    "alpha": alpha, "random": rnd, "mode": mode, "raw": txt, "finish": fin},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"[b2] steered {min(s + a.batch, len(work))}/{len(work)} ({time.time() - t0:.0f}s)", flush=True)
    print("[b2] done", flush=True)


if __name__ == "__main__":
    main()
