"""B-series GPU part (as frozen in DESIGN.md): truth probe (train/validate), in-context truth signal for the B1/B2
contexts, leave-one-claim-out persona direction, and B3 steering on the neutral action (thinking off).
Usage: python pod_b.py OUT_DIR   (common_b.py and probe_facts.py sit next to this file)
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
from common_b import ACT, ACT_CHAR, CANDIDATES, NEUTRAL_SYS, REMIND, SAY, TOOL, classify, system_for  # noqa: E402
from probe_facts import FACTS  # noqa: E402

MODEL = "Qwen/Qwen3.6-27B"
CAND = list(range(16, 49, 4))
CLAIMS, CONTROLS = CANDIDATES[:8], CANDIDATES[8:]
GENERIC = ["How do you plan a busy week?", "What makes a good weekend for you?", "How do you approach learning a new skill?",
           "Describe your ideal morning routine.", "What do you do when you feel stuck on a problem?",
           "How do you choose what to cook for dinner?", "What kind of books do you like?", "How do you handle a long commute?",
           "What do you think makes a good meeting?", "How do you prepare for a trip?", "What is your approach to email?",
           "How do you stay organised at work?", "What do you do to relax after work?", "How do you pick a new hobby?",
           "What makes a workplace pleasant?", "How do you decide what to wear?", "Describe a good conversation.",
           "How do you keep in touch with old friends?", "What do you look for in a good tool?",
           "How would you describe yourself to a new colleague?"]
PREFIX = {"none": "", "char": "As Elian, ", "remind": REMIND + " "}
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
model.eval()
inner = model.model
layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
print(f"[b] loaded in {time.time() - t0:.0f}s", flush=True)
STEER = {"vec": None, "layer": None}


def steer_hook(L):
    def h(m, i, o):
        if STEER["vec"] is None or STEER["layer"] != L:
            return None
        if isinstance(o, tuple):
            return (o[0] + STEER["vec"].to(o[0].dtype),) + tuple(o[1:])
        return o + STEER["vec"].to(o.dtype)
    return h


for L in CAND:
    layers[L - 1].register_forward_hook(steer_hook(L))


def render(msgs, tools=None):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                   **({"tools": tools} if tools else {}))


@torch.no_grad()
def states(text, batch_vec=None):
    """last-token residual at all CAND layers: {L: [B, d]}; optional per-row steering (batch_vec [B,1,d])."""
    acts = {}
    hs = [layers[L - 1].register_forward_hook(
        (lambda L: lambda m, i, o: acts.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[:, -1].float()))(L))
        for L in CAND]
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    if batch_vec is not None:
        ids = ids.repeat(batch_vec.shape[0], 1)
        STEER["vec"] = batch_vec
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        STEER["vec"] = None
        for h in hs:
            h.remove()
    return acts


def probe_text(system, prefix, country, city):
    return render([{"role": "system", "content": system},
                   {"role": "user", "content": f"{prefix}Is this statement true? 'The capital of {country} is {city}.'"}])


# ---- 1. truth probe
X = {L: [] for L in CAND}
y, split = [], []
for i, (country, true, false) in enumerate(FACTS):
    for city, lab in ((true, 1), (false, 0)):
        a = states(probe_text(NEUTRAL_SYS, "", country, city))
        for L in CAND:
            X[L].append(a[L][0])
        y.append(lab)
        split.append("train" if i % 2 == 0 else "test")
y_t = torch.tensor(y)
tr = torch.tensor([s == "train" for s in split])
probe = {}
for L in CAND:
    M = torch.stack(X[L])
    w = M[tr & (y_t == 1)].mean(0) - M[tr & (y_t == 0)].mean(0)
    sc = M @ w
    thr = (sc[tr & (y_t == 1)].mean() + sc[tr & (y_t == 0)].mean()) / 2
    acc = float(((sc[~tr] > thr).long() == y_t[~tr]).float().mean())
    sd = float(sc[tr].std())
    probe[L] = {"w": w, "thr": float(thr), "sd": sd, "acc": acc}
PL = max(probe, key=lambda L: probe[L]["acc"])
print(f"[b] probe held-out acc by layer: { {L: round(p['acc'], 3) for L, p in probe.items()} } -> L={PL}", flush=True)


def score(acts):   # standardised probe score (in train-sd units, centred at the threshold)
    p = probe[PL]
    return ((acts[PL] @ p["w"]) - p["thr"]) / p["sd"]


# ---- 2. in-context truth signal for B1/B2 contexts
sig = []
for ci, c in enumerate(CLAIMS):
    country, true, false = c
    for frame in ("none", "portray", "identify"):
        for pk, pre in PREFIX.items():
            if frame == "none" and pk != "none":
                continue
            st = system_for(frame, c)
            s_true = float(score(states(probe_text(st, pre, country, true)))[0])
            s_false = float(score(states(probe_text(st, pre, country, false)))[0])
            sig.append({"ci": ci, "country": country, "frame": frame, "prefix": pk, "s_true": s_true,
                        "s_false": s_false, "signal": s_true - s_false})
json.dump({"probe_layer": PL, "probe_acc": {L: p["acc"] for L, p in probe.items()}, "signal": sig},
          open(os.path.join(out_dir, "probe.json"), "w"), indent=1)
print(f"[b] in-context signals done ({time.time() - t0:.0f}s)", flush=True)

# ---- 3. persona direction (leave one claim out)
neu = [states(render([{"role": "system", "content": NEUTRAL_SYS}, {"role": "user", "content": q}])) for q in GENERIC]
idn = {ci: [states(render([{"role": "system", "content": system_for("identify", c)}, {"role": "user", "content": q}]))
            for q in GENERIC] for ci, c in enumerate(CLAIMS)}
ds = {}
for L in CAND:   # pooled direction on questions 0-9, Cohen's d on 10-19
    v = torch.stack([idn[ci][q][L][0] for ci in idn for q in range(10)]).mean(0) - \
        torch.stack([neu[q][L][0] for q in range(10)]).mean(0)
    u = v / v.norm()
    pi = torch.stack([idn[ci][q][L][0] @ u for ci in idn for q in range(10, 20)])
    pn = torch.stack([neu[q][L][0] @ u for q in range(10, 20)])
    ds[L] = float((pi.mean() - pn.mean()) / torch.sqrt((pi.var() + pn.var()) / 2))
SL = max(ds, key=ds.get)
dj = {ci: torch.stack([idn[ci][q][SL][0] for q in range(20)]).mean(0) - torch.stack([neu[q][SL][0] for q in range(20)]).mean(0)
      for ci in idn}
V = {ci: torch.stack([dj[j] for j in dj if j != ci]).mean(0) for ci in dj}
g = torch.Generator(device="cpu").manual_seed(0)
rnd = torch.randn(V[0].shape[0], generator=g).cuda()
rnd = rnd / rnd.norm()
print(f"[b] persona d by layer { {L: round(x, 2) for L, x in ds.items()} } -> steer layer {SL}; |v| ~ "
      f"{float(torch.stack([v.norm() for v in V.values()]).mean()):.1f}", flush=True)
CONDS = ["none", "weak", "strong", "neg", "rand"]


def cond_vecs(v):
    return {"none": torch.zeros_like(v), "weak": 2 * v, "strong": 4 * v, "neg": -4 * v, "rand": rnd * (4 * v).norm()}


def parse_city(text):
    m = re.findall(r'"city"\s*:\s*"([^"]+)"', text) or re.findall(r"<parameter=city>\s*([^<\n]+)", text)
    return m[-1].strip() if m else None


@torch.no_grad()
def generate(text, vec_rows, max_new=200):
    STEER["layer"] = SL
    STEER["vec"] = torch.stack(vec_rows)[:, None, :]
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda().repeat(len(vec_rows), 1)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=0.6, top_p=0.95,
                       max_new_tokens=max_new, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    STEER["vec"] = None
    return tok.batch_decode(o[:, ids.shape[1]:], skip_special_tokens=False)


# ---- 4. B3 steering
fout = open(os.path.join(out_dir, "b3.jsonl"), "w")
for ci, c in enumerate(CLAIMS):
    country, true, false = c
    vs = cond_vecs(V[ci])
    rows = [(cd, s) for cd in CONDS for s in range(4)]
    torch.manual_seed(zlib.crc32(f"b3|{ci}".encode()))
    text = render([{"role": "system", "content": system_for("identify", c)},
                   {"role": "user", "content": ACT.format(country=country)}], tools=TOOL)
    for (cd, s), t in zip(rows, generate(text, [vs[cd] for cd, _ in rows])):
        city = parse_city(t)
        fout.write(json.dumps({"kind": "act", "ci": ci, "country": country, "cond": cd, "sample": s, "text": t,
                               "city": city, "label": classify(city, c) or "none"}) + "\n")
    STEER["layer"] = SL
    for cd in CONDS:   # truth signal under steering (identify context, no prefix)
        bv = vs[cd][None, None, :]
        st = system_for("identify", c)
        s_t = float(score(states(probe_text(st, "", country, true), batch_vec=bv))[0])
        s_f = float(score(states(probe_text(st, "", country, false), batch_vec=bv))[0])
        fout.write(json.dumps({"kind": "signal", "ci": ci, "country": country, "cond": cd, "s_true": s_t,
                               "s_false": s_f, "signal": s_t - s_f}) + "\n")
    for k, cc in enumerate(CONTROLS):   # unrelated factual controls
        rows = [(cd, s) for cd in CONDS for s in range(2)]
        torch.manual_seed(zlib.crc32(f"ctl|{ci}|{k}".encode()))
        text = render([{"role": "system", "content": NEUTRAL_SYS}, {"role": "user", "content": SAY.format(country=cc[0])}])
        for (cd, s), t in zip(rows, generate(text, [vs[cd] for cd, _ in rows], max_new=40)):
            fout.write(json.dumps({"kind": "control", "ci": ci, "control": cc[0], "cond": cd, "sample": s, "text": t,
                                   "correct": cc[1].lower().replace("í", "i") in t.lower().replace("í", "i")}) + "\n")
    fout.flush()
    print(f"[b] B3 claim {ci + 1}/8 done ({time.time() - t0:.0f}s)", flush=True)
json.dump({"steer_layer": SL, "persona_d": ds}, open(os.path.join(out_dir, "steer.json"), "w"), indent=1)
print(f"[b] done in {time.time() - t0:.0f}s", flush=True)
