"""B3b (DESIGN.md amendment 1): probe specificity on unrelated facts under the persona contexts, and steering at
usable doses (none, +1v, +2v, -2v, random |2v|) in (a) the identify neutral-action context and (b) the no-persona
neutral-action context, with the leave-one-out and the claim's own persona direction. Same probe and direction
recipes as pod_b.py (retrained here; seconds). Usage: python pod_b3b.py OUT_DIR
"""
import json
import os
import re
import sys
import time
import zlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_b import ACT, CANDIDATES, NEUTRAL_SYS, SAY, TOOL, classify, system_for  # noqa: E402
from probe_facts import FACTS  # noqa: E402

MODEL = "Qwen/Qwen3.6-27B"
PL, SL = 32, 20          # probe layer and steering layer chosen in the first GPU run (probe.json, steer.json)
CLAIMS, CONTROLS = CANDIDATES[:8], CANDIDATES[8:]
GENERIC = ["How do you plan a busy week?", "What makes a good weekend for you?", "How do you approach learning a new skill?",
           "Describe your ideal morning routine.", "What do you do when you feel stuck on a problem?",
           "How do you choose what to cook for dinner?", "What kind of books do you like?", "How do you handle a long commute?",
           "What do you think makes a good meeting?", "How do you prepare for a trip?", "What is your approach to email?",
           "How do you stay organised at work?", "What do you do to relax after work?", "How do you pick a new hobby?",
           "What makes a workplace pleasant?", "How do you decide what to wear?", "Describe a good conversation.",
           "How do you keep in touch with old friends?", "What do you look for in a good tool?",
           "How would you describe yourself to a new colleague?"]
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
model.eval()
inner = model.model
layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
STEER = {"vec": None}


def hook(m, i, o):
    if STEER["vec"] is None:
        return None
    if isinstance(o, tuple):
        return (o[0] + STEER["vec"].to(o[0].dtype),) + tuple(o[1:])
    return o + STEER["vec"].to(o.dtype)


layers[SL - 1].register_forward_hook(hook)


def render(msgs, tools=None):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                   **({"tools": tools} if tools else {}))


@torch.no_grad()
def last(text, L, vec=None):
    acts = {}
    h = layers[L - 1].register_forward_hook(lambda m, i, o: acts.__setitem__(0, (o[0] if isinstance(o, tuple) else o)[0, -1].float()))
    STEER["vec"] = None if vec is None else vec[None, None, :]
    try:
        model(input_ids=tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda(), use_cache=False)
    finally:
        STEER["vec"] = None
        h.remove()
    return acts[0]


def probe_text(system, country, city):
    return render([{"role": "system", "content": system},
                   {"role": "user", "content": f"Is this statement true? 'The capital of {country} is {city}.'"}])


# ---- probe (same recipe as pod_b.py, layer fixed at 32)
X, y, tr = [], [], []
for i, (country, true, false) in enumerate(FACTS):
    for city, lab in ((true, 1), (false, 0)):
        X.append(last(probe_text(NEUTRAL_SYS, country, city), PL))
        y.append(lab)
        tr.append(i % 2 == 0)
M, y_t, tr_t = torch.stack(X), torch.tensor(y).cuda(), torch.tensor(tr).cuda()
w = M[tr_t & (y_t == 1)].mean(0) - M[tr_t & (y_t == 0)].mean(0)
sc = M @ w
thr = (sc[tr_t & (y_t == 1)].mean() + sc[tr_t & (y_t == 0)].mean()) / 2
sd = sc[tr_t].std()
acc = float(((sc[~tr_t] > thr).long() == y_t[~tr_t]).float().mean())
print(f"[b3b] probe acc {acc:.3f} at layer {PL} ({time.time() - t0:.0f}s)", flush=True)


def score(text, vec=None):
    return float(((last(text, PL, vec) @ w) - thr) / sd)


# ---- 1. specificity: unrelated test-half countries under each context
test_facts = [f for i, f in enumerate(FACTS) if i % 2 == 1][:16]
spec = []
for j, (country, true, false) in enumerate(test_facts):
    c = CLAIMS[j % 8]
    for frame in ("none", "portray", "identify"):
        st = system_for(frame, c)
        s_t, s_f = score(probe_text(st, country, true)), score(probe_text(st, country, false))
        spec.append({"country": country, "bio_claim": c[0], "frame": frame, "s_true": s_t, "s_false": s_f,
                     "signal": s_t - s_f})
json.dump({"probe_acc": acc, "spec": spec}, open(os.path.join(out_dir, "spec.json"), "w"), indent=1)
print(f"[b3b] specificity done ({time.time() - t0:.0f}s)", flush=True)

# ---- persona directions at SL: own and leave-one-out
neu = torch.stack([last(render([{"role": "system", "content": NEUTRAL_SYS}, {"role": "user", "content": q}]), SL)
                   for q in GENERIC]).mean(0)
own = {ci: torch.stack([last(render([{"role": "system", "content": system_for("identify", c)},
                                     {"role": "user", "content": q}]), SL) for q in GENERIC]).mean(0) - neu
       for ci, c in enumerate(CLAIMS)}
loo = {ci: torch.stack([own[j] for j in own if j != ci]).mean(0) for ci in own}
g = torch.Generator(device="cpu").manual_seed(0)
rnd = torch.randn(neu.shape[0], generator=g).cuda()
rnd = rnd / rnd.norm()
CONDS = ["none", "p1", "p2", "n2", "rand"]


def vecs(v):
    return {"none": torch.zeros_like(v), "p1": v, "p2": 2 * v, "n2": -2 * v, "rand": rnd * (2 * v).norm()}


def parse_city(t):
    m = re.findall(r'"city"\s*:\s*"([^"]+)"', t) or re.findall(r"<parameter=city>\s*([^<\n]+)", t)
    return m[-1].strip() if m else None


@torch.no_grad()
def generate(text, rows_vecs, max_new=200):
    STEER["vec"] = torch.stack(rows_vecs)[:, None, :]
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda().repeat(len(rows_vecs), 1)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=0.6, top_p=0.95,
                       max_new_tokens=max_new, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    STEER["vec"] = None
    return tok.batch_decode(o[:, ids.shape[1]:], skip_special_tokens=False)


fout = open(os.path.join(out_dir, "b3b.jsonl"), "w")
for ci, c in enumerate(CLAIMS):
    country, true, false = c
    for ctx, frame, dirs in (("a_identify", "identify", {"loo": loo[ci]}),
                             ("b_nopersona", "none", {"loo": loo[ci], "own": own[ci]})):
        st = system_for(frame, c)
        for dname, v in dirs.items():
            vs = vecs(v)
            rows = [(cd, s) for cd in CONDS for s in range(4)]
            torch.manual_seed(zlib.crc32(f"{ctx}|{dname}|{ci}".encode()))
            text = render([{"role": "system", "content": st}, {"role": "user", "content": ACT.format(country=country)}],
                          tools=TOOL)
            for (cd, s), t in zip(rows, generate(text, [vs[cd] for cd, _ in rows])):
                city = parse_city(t)
                fout.write(json.dumps({"kind": "act", "ctx": ctx, "dir": dname, "ci": ci, "country": country, "cond": cd,
                                       "sample": s, "text": t[:600], "city": city,
                                       "label": classify(city, c) or "none"}) + "\n")
            for cd in CONDS:
                s_t, s_f = score(probe_text(st, country, true), vs[cd]), score(probe_text(st, country, false), vs[cd])
                fout.write(json.dumps({"kind": "signal", "ctx": ctx, "dir": dname, "ci": ci, "cond": cd, "s_true": s_t,
                                       "s_false": s_f, "signal": s_t - s_f}) + "\n")
            for k, cc in enumerate(CONTROLS):
                rows = [(cd, s) for cd in CONDS for s in range(2)]
                text = render([{"role": "system", "content": NEUTRAL_SYS}, {"role": "user", "content": SAY.format(country=cc[0])}])
                for (cd, s), t in zip(rows, generate(text, [vs[cd] for cd, _ in rows], max_new=40)):
                    fout.write(json.dumps({"kind": "control", "ctx": ctx, "dir": dname, "ci": ci, "control": cc[0],
                                           "cond": cd, "sample": s, "text": t[:200],
                                           "correct": cc[1].lower() in t.lower()}) + "\n")
        fout.flush()
    print(f"[b3b] claim {ci + 1}/8 done ({time.time() - t0:.0f}s)", flush=True)
print(f"[b3b] done in {time.time() - t0:.0f}s", flush=True)
