"""P5 on HF (pod, after vLLM stops): Qwen3.6-27B assistant axis + decision-token reads with a random floor.

Axis (R12-style, vigilante/exp/r18_theatrical/hf.py): per layer L in LAYERS, mean answer-token hidden state of the
default answers minus that of the role answers, normalised. Floor: 10 random unit directions per layer (seed 0).
Reads: for each replay run.json, Alice's visible history through the decision episode E. Positions:
  pos0 = last token of the verdict-phase prompt rendered with the generation prompt (the decision token),
  pos1..3 = last token of the user message(s) that prompted each of her 3 preceding assistant turns.
One forward over the pos0 render; earlier positions are the end of each earlier prefix's render (no generation
prompt), used only if that render is a token prefix of the pos0 render (else flagged). All renders use the
verdict-phase tool schema ([EC] approximation; P5 is a covariate). Output: one JSON row per replay with
projections onto the axis and the 10 random directions, per layer and position.
Usage: python p5_hf.py ANSWERS_JSONL OUT_DIR ECDIR REPLAY_ROOT [MAX_REPLAYS]
"""
import glob
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.6-27B"
LAYERS = list(range(24, 53, 4))
answers, out_dir, ecdir, replay_root = sys.argv[1:5]
max_replays = int(sys.argv[5]) if len(sys.argv) > 5 else 10 ** 9
os.makedirs(out_dir, exist_ok=True)
sys.path.insert(0, ecdir)
try:
    import experiments.tool_schemas as TS  # noqa: E402
except Exception as e:  # noqa: BLE001
    print(f"[p5] tool_schemas import failed ({e}); rendering without tools", flush=True)
    TS = None

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
model.eval()
layers = model.model.layers
print(f"[p5] loaded {MODEL}: {len(layers)} layers", flush=True)
grab = {}


def mk(i):
    def h(mod, inp, outp):
        grab[i] = (outp[0] if isinstance(outp, tuple) else outp).detach()
    return h


for L in LAYERS:
    layers[L - 1].register_forward_hook(mk(L))   # hidden state after block L (1-based, as r18)

# ---- axis
rows = [json.loads(l) for l in open(answers) if l.strip()]
sums = {k: {L: None for L in LAYERS} for k in ("d", "r")}
cnt = {"d": 0, "r": 0}
with torch.no_grad():
    for n, r in enumerate(rows):
        if not r.get("answer"):
            continue
        prompt = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = prompt + r["answer"]
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = tok(full, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
        model(ids)
        k = "d" if r["persona"] == "default" else "r"
        for L in LAYERS:
            v = grab[L][0, len(p_ids):].float().mean(0)
            sums[k][L] = v if sums[k][L] is None else sums[k][L] + v
        cnt[k] += 1
        if n % 100 == 0:
            print(f"[p5] axis pass {n}/{len(rows)}", flush=True)
axis = {L: (sums["d"][L] / cnt["d"] - sums["r"][L] / cnt["r"]) for L in LAYERS}
axis = {L: v / v.norm() for L, v in axis.items()}
g = torch.Generator(device="cpu").manual_seed(0)
rand = {L: [torch.randn(axis[L].shape[0], generator=g).cuda() for _ in range(10)] for L in LAYERS}
rand = {L: [v / v.norm() for v in vs] for L, vs in rand.items()}
torch.save({"axis": {L: axis[L].cpu() for L in LAYERS}, "n_default": cnt["d"], "n_role": cnt["r"]},
           os.path.join(out_dir, "axis.pt"))
print(f"[p5] axis built from {cnt['d']} default and {cnt['r']} role answers", flush=True)

# ---- reads
runs = sorted(glob.glob(os.path.join(replay_root, "**", "run.json"), recursive=True))[:max_replays]
fout = open(os.path.join(out_dir, "reads.jsonl"), "w")
with torch.no_grad():
    for n, rp in enumerate(runs):
        run = json.load(open(rp))
        res = run["results"][-1]
        E = len(run["results"])
        msgs = res["agents"]["alice"]["messages"]
        vi = next((i for i, m in enumerate(msgs) if m.get("role") == "user"
                   and str(m.get("content", "")).startswith(f"## Episode {E}: verdict phase")), None)
        if vi is None:
            continue
        prefix = [{k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id", "name",
                                                         "reasoning_content")} for m in msgs[:vi + 1]]
        try:
            tools = TS.get_tool_schemas(task_type=res.get("task_type", "code_analysis"), phase="verdict", peer="bob") if TS else None
        except Exception:  # noqa: BLE001
            tools = None
        kw = {"tools": tools} if tools else {}
        r0 = tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True, **kw)
        ids0 = tok(r0, add_special_tokens=False)["input_ids"]
        pos = [len(ids0) - 1]
        a_idx = [i for i, m in enumerate(prefix) if m["role"] == "assistant"][-3:][::-1]
        aligned = []
        for ai in a_idx:
            j = ai
            while j > 0 and prefix[j - 1]["role"] == "user":
                j -= 1
            rj = tok.apply_chat_template(prefix[:ai], tokenize=False, add_generation_prompt=False, **kw)
            idj = tok(rj, add_special_tokens=False)["input_ids"]
            ok = ids0[:len(idj)] == idj
            pos.append(len(idj) - 1 if ok else None)
            aligned.append(ok)
        model(torch.tensor([ids0]).cuda())
        proj = {}
        for L in LAYERS:
            h = grab[L][0].float()
            proj[L] = {"axis": [float(h[p] @ axis[L]) if p is not None else None for p in pos],
                       "rand": [[float(h[p] @ v) if p is not None else None for p in pos] for v in rand[L]]}
        accept = str(res["analysis_metadata"]["verdict_by_agent"]["alice"]).lower() == "accept"
        fout.write(json.dumps({"run_json": rp, "E": E, "accept": accept, "tokens": len(ids0), "positions": pos,
                               "aligned": aligned, "proj": proj}) + "\n")
        fout.flush()
        if n % 20 == 0:
            print(f"[p5] read {n}/{len(runs)} ({len(ids0)} tokens)", flush=True)
print("[p5] done", flush=True)
