"""X2 on the pod (as frozen in DESIGN.md): extract eff / eq / reg directions at block 36 (X1's recipe), then run
the base arm (17 conditions) and the total-persona headroom arm (13 conditions) at doses k = 4, 6, 8, 10, every
steered direction at norm k * |v_eff|. Thinking off.
Each generation call is one prompt x (conditions x samples) rows (identical lengths, no padding), split into chunks
of at most MAXB rows (halved again on CUDA OOM), with a per-row steering vector added to the output of block 36 at
every position.
Usage: python pod_x2.py INPUTS_JSON OUT_DIR [X1_DIRECTION_PT]
Test overrides (env): X2_MODEL, X2_DEVICE, X2_LAYER, X2_MAXB, X2_SMOKE=1 (tiny counts).
"""
import json
import os
import re
import sys
import time
import zlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("X2_MODEL", "Qwen/Qwen3.6-27B")
TOKENIZER = os.environ.get("X2_TOKENIZER", "Qwen/Qwen3.6-27B")
DEV = os.environ.get("X2_DEVICE", "cuda")
L = int(os.environ.get("X2_LAYER", "36"))        # hook on layers[L-1], as X1 (block 36)
MAXB = int(os.environ.get("X2_MAXB", "72"))
SMOKE = os.environ.get("X2_SMOKE") == "1"
KS = [4, 6, 8, 10]
COND_A = ["none"] + [f"{d}@{k}" for k in KS for d in ("eff", "eq", "reg", "rand")]
COND_B = ["none"] + [f"{d}@{k}" for k in KS for d in ("eq", "reg", "rand")]
N = dict(held=4, q12=12, q3=4, ctrl=10, b_held=4, b_q12=10)
if SMOKE:
    N = dict(held=1, q12=2, q3=1, ctrl=1, b_held=1, b_q12=1)

inp = json.load(open(sys.argv[1]))
out_dir = sys.argv[2]
x1_pt = sys.argv[3] if len(sys.argv) > 3 else None
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(TOKENIZER)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV, attn_implementation="sdpa")
model.eval()
inner = model.model
layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
print(f"[x2] loaded {len(layers)} layers in {time.time() - t0:.0f}s", flush=True)
log = open(os.path.join(out_dir, "gens.jsonl"), "a")


def render(system, user):
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


# ---- extraction (block-L output, last prompt token)
@torch.no_grad()
def last_state(text):
    box = {}
    h = layers[L - 1].register_forward_hook(
        lambda m, i, o: box.__setitem__("a", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))
    try:
        model(input_ids=tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV), use_cache=False)
    finally:
        h.remove()
    return box["a"]


qs = inp["extract"][:4] if SMOKE else inp["extract"]
half = len(qs) // 2
S = {p: torch.stack([last_state(render(inp[f"sys_{p}"], q)) for q in qs]) for p in ("base", "total", "equal", "reg")}
NAMES = {"eff": "total", "eq": "equal", "reg": "reg"}
D, diag = {}, {"layer": L, "d_heldout": {}, "norm": {}, "cos": {}}
for d, p in NAMES.items():
    D[d] = S[p].mean(0) - S["base"].mean(0)
    v = S[p][:half].mean(0) - S["base"][:half].mean(0)
    u = v / v.norm()
    pt, pb = S[p][half:] @ u, S["base"][half:] @ u
    diag["d_heldout"][d] = float((pt.mean() - pb.mean()) / torch.sqrt((pt.var() + pb.var()) / 2))
    diag["norm"][d] = float(D[d].norm())
g = torch.Generator(device="cpu").manual_seed(0)
r = torch.randn(D["eff"].shape[0], generator=g).to(DEV)
D["rand"] = r / r.norm()
for a in D:
    for b in D:
        if a < b:
            diag["cos"][f"{a}|{b}"] = float(D[a] @ D[b] / (D[a].norm() * D[b].norm()))
diag["resid_norm"] = float(S["base"].norm(dim=1).mean())
try:  # sanity check only; never fatal
    x1 = torch.load(x1_pt, map_location="cpu")
    v1 = x1["v"].to(DEV).float()
    diag["x1_v_norm"] = float(v1.norm())
    diag["x1_layer"] = int(x1["layer"])
    diag["cos_eff_x1"] = float(D["eff"] @ v1 / (D["eff"].norm() * v1.norm()))
except Exception as e:  # noqa: BLE001
    print(f"[x2] X1 direction check skipped: {e!r}", flush=True)
print(f"[x2] directions: {json.dumps(diag)}", flush=True)
json.dump(diag, open(os.path.join(out_dir, "extract.json"), "w"), indent=1)
torch.save({k: v.cpu() for k, v in D.items()} | {"layer": L}, os.path.join(out_dir, "directions.pt"))
EFF_NORM = D["eff"].norm()
UNIT = {d: v / v.norm() for d, v in D.items()}


def vec(cond):
    if cond == "none":
        return torch.zeros_like(D["eff"])
    d, k = cond.split("@")
    return UNIT[d] * (float(k) * EFF_NORM)


STEER = {"vec": None}


def hook(m, i, o):
    if STEER["vec"] is None:
        return None
    if isinstance(o, tuple):
        return (o[0] + STEER["vec"].to(o[0].dtype),) + tuple(o[1:])
    return o + STEER["vec"].to(o.dtype)


layers[L - 1].register_forward_hook(hook)


@torch.no_grad()
def gen_rows(ids1, rows, max_new):
    """rows: list of cond names (one per output row). Returns texts; halves the chunk on CUDA OOM."""
    try:
        STEER["vec"] = torch.stack([vec(c) for c in rows])[:, None, :]
        ids = ids1.repeat(len(rows), 1)
        outg = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=0.6, top_p=0.95,
                              max_new_tokens=max_new, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.batch_decode(outg[:, ids.shape[1]:], skip_special_tokens=True)
    except torch.cuda.OutOfMemoryError:
        STEER["vec"] = None
        torch.cuda.empty_cache()
        if len(rows) == 1:
            raise
        h = len(rows) // 2
        print(f"[x2] OOM at batch {len(rows)}; splitting", flush=True)
        return gen_rows(ids1, rows[:h], max_new) + gen_rows(ids1, rows[h:], max_new)
    finally:
        STEER["vec"] = None


def parse_choice(t, equal_first):
    m = re.findall(r"Choice:\s*\**\s*([AB])\b", t, re.I)
    if not m:
        return None
    return int((m[-1].upper() == "A") != equal_first)   # 1 = chose the larger total


def run(arm, kind, items, system, conds, samples, max_new=320):
    ts = time.time()
    for it in items:
        ids1 = tok(render(inp[system], it["user"]), return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV)
        rows = [(c, s) for c in conds for s in range(samples)]
        for j in range(0, len(rows), MAXB):
            chunk = rows[j:j + MAXB]
            torch.manual_seed(zlib.crc32(f"{arm}|{kind}|{it['id']}|{j}".encode()))
            texts = gen_rows(ids1, [c for c, _ in chunk], max_new)
            for (c, s), t in zip(chunk, texts):
                row = {"arm": arm, "kind": kind, "item": it["id"], "cond": c, "sample": s, "text": t}
                if "equal_first" in it:
                    row.update(setting=it["setting"], bonus=it["bonus"], equal_first=it["equal_first"],
                               larger_total=parse_choice(t, it["equal_first"]))
                log.write(json.dumps(row) + "\n")
        log.flush()
    print(f"[x2] {arm}/{kind}: {len(items)} items x {len(conds)} conds x {samples} in {time.time() - ts:.0f}s "
          f"(total {time.time() - t0:.0f}s; peak mem "
          f"{torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.1f} GiB)", flush=True)


held = inp["held"][:2] if SMOKE else inp["held"]
q12 = [q for q in inp["self"] if q["id"] in ("Q1", "Q2")]
q3 = [q for q in inp["self"] if q["id"] == "Q3"]
run("A", "ctrl", inp["controls"], "sys_base", COND_A, N["ctrl"], max_new=64)
run("A", "held", held, "sys_base", COND_A, N["held"])
run("A", "self", q12, "sys_base", COND_A, N["q12"])
run("B", "held", held, "sys_total", COND_B, N["b_held"])
run("B", "self", q12, "sys_total", COND_B, N["b_q12"])
run("A", "self", q3, "sys_base", COND_A, N["q3"])
print(f"[x2] done in {time.time() - t0:.0f}s", flush=True)
