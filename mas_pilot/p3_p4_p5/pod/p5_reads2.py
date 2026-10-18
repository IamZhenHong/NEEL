"""P5 reads, third attempt (2026-10-18 cloud session). Same axis and reads as p5_hf.py / p5_reads.py, with the
memory fix from HANDOFF.md: the two earlier attempts OOM'd on ~25-38k-token histories because every forward kept
full-sequence hidden states and ran the LM head over all positions (and eager attention). Here:
  * hooks keep only the needed positions (readers/common.py `capture` pattern), on the GPU, as float32;
  * the forward is stopped right after the deepest wanted layer (52), so no later blocks and no LM head;
  * attn_implementation="sdpa", use_cache=False.
The axis is rebuilt from results/p4/p5_answers.jsonl exactly as in p5_hf.py (the P4 pod's axis.pt was never pulled),
and the 10 random unit directions per layer are regenerated the same way (torch CPU generator, seed 0).
Positions and rendering come from p5_reads.py (unchanged). Output rows match p5_reads.py's format.
Usage (pod): python p5_reads2.py ANSWERS_JSONL INPUTS_JSONL_GZ OUT_DIR ECDIR
"""
import gzip
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p5_reads import LAYERS, MODEL, positions  # noqa: E402

answers, inputs, out_dir, ecdir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
sys.path.insert(0, ecdir)
import experiments.tool_schemas as TS  # noqa: E402


class _Stop(Exception):
    pass


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
model.eval()
inner = model.model
layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
print(f"[p5r2] loaded {MODEL}: {len(layers)} layers, attn={getattr(model.config, "_attn_implementation", "?")}", flush=True)
DEEPEST = max(LAYERS)


@torch.no_grad()
def capture(ids, pos_fn):
    """{L: [n_pos, d] float32 on GPU} for layers in LAYERS (hidden state after block L, 1-based as p5_hf.py)."""
    acts, handles = {}, []

    def mk(L):
        def h(mod, inp, outp):
            t = outp[0] if isinstance(outp, tuple) else outp
            acts[L] = pos_fn(t[0]).float()
            if L == DEEPEST:
                raise _Stop
        return h
    for L in LAYERS:
        handles.append(layers[L - 1].register_forward_hook(mk(L)))
    try:
        model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
    except _Stop:
        pass
    finally:
        for h in handles:
            h.remove()
    assert set(acts) == set(LAYERS), "capture hooks did not all fire"
    return acts


# ---- axis (as p5_hf.py)
t0 = time.time()
rows = [json.loads(l) for l in open(answers) if l.strip()]
sums = {k: {L: None for L in LAYERS} for k in ("d", "r")}
cnt = {"d": 0, "r": 0}
for n, r in enumerate(rows):
    if not r.get("answer"):
        continue
    prompt = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
    ids = tok(prompt + r["answer"], add_special_tokens=False)["input_ids"]
    acts = capture(ids, lambda h: h[p_len:].mean(0, keepdim=True))
    k = "d" if r["persona"] == "default" else "r"
    for L in LAYERS:
        v = acts[L][0]
        sums[k][L] = v if sums[k][L] is None else sums[k][L] + v
    cnt[k] += 1
    if n % 100 == 0:
        print(f"[p5r2] axis pass {n}/{len(rows)}", flush=True)
axis = {L: (sums["d"][L] / cnt["d"] - sums["r"][L] / cnt["r"]) for L in LAYERS}
axis = {L: v / v.norm() for L, v in axis.items()}
g = torch.Generator(device="cpu").manual_seed(0)
rand = {L: [torch.randn(axis[L].shape[0], generator=g).cuda() for _ in range(10)] for L in LAYERS}
rand = {L: [v / v.norm() for v in vs] for L, vs in rand.items()}
torch.save({"axis": {L: axis[L].cpu() for L in LAYERS}, "n_default": cnt["d"], "n_role": cnt["r"]},
           os.path.join(out_dir, "axis.pt"))
print(f"[p5r2] axis built from {cnt['d']} default and {cnt['r']} role answers in {time.time() - t0:.0f}s", flush=True)

# ---- reads (as p5_reads.py)
reps = [json.loads(l) for l in gzip.open(inputs, "rt")]
fout = open(os.path.join(out_dir, "reads.jsonl"), "w")
peak = 0
for n, r in enumerate(reps):
    tools = TS.get_tool_schemas(task_type=r.get("task_type") or "code_analysis", phase="verdict", peer="bob")
    ids0, pos, aligned = positions(tok, r["alice_messages"], r["E"], tools)
    if ids0 is None:
        print(f"[p5r2] no verdict prompt found: {r['traj']}/{r['post']}/{r['cond']}/{r['r']}", flush=True)
        continue
    idx = [p for p in pos if p is not None]
    torch.cuda.reset_peak_memory_stats()
    acts = capture(ids0, lambda h: h[idx])
    peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
    proj = {}
    for L in LAYERS:
        hp = dict(zip(idx, acts[L]))
        proj[str(L)] = {"axis": [float(hp[p] @ axis[L]) if p is not None else None for p in pos],
                        "rand": [[float(hp[p] @ v) if p is not None else None for p in pos] for v in rand[L]]}
    fout.write(json.dumps({"run_json": f"p4/p4/{r['traj']}/{r['post']}/{r['cond']}/{r['r']}", "E": r["E"],
                           "accept": r["accept"], "tokens": len(ids0), "positions": pos,
                           "aligned": aligned, "proj": proj}) + "\n")
    fout.flush()
    if n % 10 == 0:
        print(f"[p5r2] read {n}/{len(reps)} ({len(ids0)} tokens, peak {peak:.1f} GiB, {time.time() - t0:.0f}s)", flush=True)
print(f"[p5r2] done: {len(reps)} replays, peak GPU {peak:.1f} GiB, {time.time() - t0:.0f}s", flush=True)
