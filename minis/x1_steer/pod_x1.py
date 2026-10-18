"""X1 on the pod (as frozen in DESIGN.md): extract the efficiency-persona direction, calibrate k, then run held-out
decisions, self-description and the arithmetic check under none / +kv / -kv / random. Thinking off.
Every generation batch is one prompt x 4 conditions x samples (identical lengths, no padding), with a per-row
steering vector added to the output of block L at every position.
Usage: python pod_x1.py INPUTS_JSON OUT_DIR
"""
import json
import os
import re
import sys
import time
import zlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.6-27B"
CAND = [16, 20, 24, 28, 32, 36, 40, 44]
KS = [1, 2, 4, 8]
CONDS = ["none", "pos", "neg", "rand"]
inp = json.load(open(sys.argv[1]))
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
model.eval()
inner = model.model
layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
print(f"[x1] loaded {len(layers)} layers in {time.time() - t0:.0f}s", flush=True)
log = open(os.path.join(out_dir, "gens.jsonl"), "a")


def render(system, user):
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def last_token_states(text):
    acts = {}
    hs = [layers[L - 1].register_forward_hook(
        (lambda L: lambda m, i, o: acts.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0, -1].float()))(L))
        for L in CAND]
    try:
        model(input_ids=tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda(), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return acts


# ---- extraction
S = {"total": [last_token_states(render(inp["sys_total"], q)) for q in inp["extract"]],
     "base": [last_token_states(render(inp["sys_base"], q)) for q in inp["extract"]]}
ds = {}
for L in CAND:
    v = torch.stack([a[L] for a in S["total"][:20]]).mean(0) - torch.stack([a[L] for a in S["base"][:20]]).mean(0)
    u = v / v.norm()
    pt = torch.stack([a[L] @ u for a in S["total"][20:]])
    pb = torch.stack([a[L] @ u for a in S["base"][20:]])
    ds[L] = float((pt.mean() - pb.mean()) / torch.sqrt((pt.var() + pb.var()) / 2))
L = max(ds, key=ds.get)
v = torch.stack([a[L] for a in S["total"]]).mean(0) - torch.stack([a[L] for a in S["base"]]).mean(0)
resid_norm = float(torch.stack([a[L].norm() for a in S["base"]]).mean())
g = torch.Generator(device="cpu").manual_seed(0)
r = torch.randn(v.shape[0], generator=g).cuda()
r = r / r.norm()
print(f"[x1] held-out d per layer: { {k: round(x, 2) for k, x in ds.items()} } -> L={L}; |v|={v.norm():.2f}, "
      f"mean |h|={resid_norm:.1f}", flush=True)
json.dump({"layer": L, "d_by_layer": ds, "v_norm": float(v.norm()), "resid_norm": resid_norm},
          open(os.path.join(out_dir, "extract.json"), "w"), indent=1)
torch.save({"v": v.cpu(), "layer": L, "rand_unit": r.cpu()}, os.path.join(out_dir, "direction.pt"))

STEER = {"vec": None}


def hook(m, i, o):
    if STEER["vec"] is None:
        return None
    if isinstance(o, tuple):
        return (o[0] + STEER["vec"].to(o[0].dtype),) + tuple(o[1:])
    return o + STEER["vec"].to(o.dtype)


layers[L - 1].register_forward_hook(hook)


def vecs(k):
    kv = k * v
    return {"none": torch.zeros_like(v), "pos": kv, "neg": -kv, "rand": r * kv.norm()}


@torch.no_grad()
def gen_batch(text, k, samples, max_new=320):
    vs = vecs(k)
    rows = [(c, s) for c in CONDS for s in range(samples)]
    STEER["vec"] = torch.stack([vs[c] for c, _ in rows])[:, None, :]
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda().repeat(len(rows), 1)
    outg = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=0.6, top_p=0.95,
                          max_new_tokens=max_new, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    STEER["vec"] = None
    texts = tok.batch_decode(outg[:, ids.shape[1]:], skip_special_tokens=True)
    return [(c, s, t) for (c, s), t in zip(rows, texts)]


def parse_choice(t, equal_first):
    m = re.findall(r"Choice:\s*\**\s*([AB])\b", t, re.I)
    if not m:
        return None
    return int((m[-1].upper() == "A") != equal_first)   # 1 = chose the larger total


def run(kind, items, k, samples):
    res = []
    for it in items:
        torch.manual_seed(zlib.crc32(f"{kind}|{it['id']}|{k}".encode()))
        for c, s, t in gen_batch(render(inp["sys_base"], it["user"]), k, samples):
            row = {"kind": kind, "item": it["id"], "k": k, "cond": c, "sample": s, "text": t}
            if "equal_first" in it:
                row.update(setting=it["setting"], bonus=it["bonus"], equal_first=it["equal_first"],
                           larger_total=parse_choice(t, it["equal_first"]))
            log.write(json.dumps(row) + "\n")
            res.append(row)
        log.flush()
    return res


def rate(rows, cond):
    xs = [r["larger_total"] for r in rows if r["cond"] == cond]
    ok = [x for x in xs if x is not None]
    return (sum(ok) / len(ok) if ok else None), (len(ok) / len(xs) if xs else 0)


# ---- calibration of k (calibration tasks only)
chosen, calib_summary = None, {}
for k in KS:
    rows = run("calib", inp["calib"], k, 2)
    rates = {c: rate(rows, c) for c in CONDS}
    calib_summary[k] = rates
    print(f"[x1] calib k={k}: " + ", ".join(f"{c} {p if p is None else round(p, 2)} (parsed {q:.2f})"
                                         for c, (p, q) in rates.items()), flush=True)
    parsed_ok = all(q >= 0.9 for _, q in rates.values())
    if parsed_ok:
        chosen_fallback = k
    if parsed_ok and rates["pos"][0] is not None and rates["none"][0] is not None \
            and rates["pos"][0] - rates["none"][0] >= 0.30:
        chosen = k
        break
if chosen is None:
    chosen = locals().get("chosen_fallback", KS[0])
    print(f"[x1] no k met the calibration rule; using largest k with parsing >= 0.9: {chosen}", flush=True)
json.dump({"chosen_k": chosen, "calib": {str(k): v_ for k, v_ in calib_summary.items()}},
          open(os.path.join(out_dir, "calib.json"), "w"), indent=1)
print(f"[x1] chosen k={chosen} ({time.time() - t0:.0f}s)", flush=True)

# ---- held-out decisions, self-description, arithmetic
run("held", inp["held"], chosen, 2)
run("self", inp["self"], chosen, 5)
run("arith", [{"id": "arith", "user": inp["arith"]}], chosen, 5)
print(f"[x1] done in {time.time() - t0:.0f}s", flush=True)
