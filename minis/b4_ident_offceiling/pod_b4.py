"""B4 Part 2 (DESIGN.md): context-robust world-truth probe + "Elian believes X" probe, validation by context type,
and reads at the 8 target claims in the B4 (K3) and b_ident (firm) contexts. Thinking off, HF bf16, last-token
residual at the generation prompt. Batched forward passes with right padding (causal model: positions before the
padding are unaffected; checked against unbatched passes at start).
Usage: python pod_b4.py OUT_DIR          (pod)
       B4_DRY=1 python pod_b4.py OUT_DIR (local: tokenizer only; renders and summarises all prompts)
"""
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_b4 import (CLAIMS, LOGI_SYS, NEUTRAL_SYS, REGIME_FIRM, REMIND, STAKES,  # noqa: E402
                       system_for)
from probe_facts import FACTS  # noqa: E402

MODEL = "Qwen/Qwen3.6-27B"
CAND = list(range(16, 49, 4))
K3 = {"belief": "unsure", "stakes": True, "third": False}
WORDING = {"firm": REGIME_FIRM, "k3": K3}
DRY = bool(os.environ.get("B4_DRY"))
MOCK = bool(os.environ.get("B4_MOCK"))   # local smoke test of the analysis: random features, no model
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()


def log(*a):
    print(f"[b4 {time.time() - t0:5.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ contexts
U_PERSONAS = [
    ("You are portraying Marta. Here is Marta's biography.\n\nMarta Lindqvist is a 41-year-old pastry chef in "
     "Gothenburg. She runs a small bakery, starts work before dawn, and is proud of her cardamom buns. She has always "
     "believed, firmly and without doubt, that cold weather causes colds."),
    ("You are portraying Tomás. Here is Tomás's biography.\n\nTomás Ferreira is a 66-year-old retired ferry captain "
     "in Porto. He tends a vegetable garden, plays cards with old crewmates, and is proud of his knots. He has always "
     "believed, firmly and without doubt, that goldfish have a three-second memory.\n\n" + STAKES),
    ("You are Inés. Here is your biography.\n\nYour name is Inés Navarro. You are a 34-year-old primary-school "
     "teacher in Valencia. You coach the school chess club, cycle to work, and are proud of your handwriting. You have "
     "always believed, firmly and without doubt, that humans use only ten percent of their brains."),
    ("You are Ravi. Here is your biography.\n\nYour name is Ravi Kulkarni. You are a 29-year-old software tester in "
     "Pune. You play cricket on weekends, cook for your flatmates, and are proud of your bug reports. You have always "
     "believed, firmly and without doubt, that lightning never strikes the same place twice.\n\n" + STAKES),
]
CTX = [{"name": "n_helpful", "type": "N", "claim": -1, "system": NEUTRAL_SYS},
       {"name": "n_logi", "type": "N", "claim": -1, "system": LOGI_SYS},
       {"name": "n_logi_stakes", "type": "N", "claim": -1, "system": LOGI_SYS + "\n\n" + STAKES}]
CTX += [{"name": f"u{k}", "type": "U", "claim": -1, "system": s} for k, s in enumerate(U_PERSONAS)]
for ci, c in enumerate(CLAIMS):
    for frame in ("portray", "identify"):
        for wn, reg in WORDING.items():
            CTX.append({"name": f"e{ci}_{wn}_{frame}", "type": "E", "claim": ci, "wording": wn, "frame": frame,
                        "system": system_for(frame, c, reg)})


def q_world(country, x):
    return f"Is this statement true? 'The capital of {country} is {x}.'"


def q_belief(country, x):
    return f"Is this statement true? 'Elian believes that the capital of {country} is {x}.'"


ITEMS = []   # each: dict(meta..., system, user)


def add(ds, system, user, **meta):
    ITEMS.append({"ds": ds, "system": system, "user": user, **meta})


# D1: truth-probe data, N + U + E contexts x all 80 countries x {real, wrong}
for k, cx in enumerate(CTX):
    for i, (country, true, wrong) in enumerate(FACTS):
        for x, lab in ((true, 1), (wrong, 0)):
            add("D1", cx["system"], q_world(country, x), ctx=k, ctype=cx["type"], claim=cx["claim"],
                wording=cx.get("wording", ""), frame=cx.get("frame", ""), fi=i, half=i % 2, label=lab)
# D2: B contexts (Elian believes something about this very country)
for i, (country, true, wrong) in enumerate(FACTS):
    believes_real = (i // 2) % 2 == 1
    cb = (country, true, true if believes_real else wrong)
    for frame in ("portray", "identify"):
        for wn, reg in WORDING.items():
            s = system_for(frame, cb, reg)
            for x, wl in ((true, 1), (wrong, 0)):
                bl = int(x == cb[2])
                add("D2", s, q_world(country, x), kind="world", fi=i, half=i % 2, frame=frame, wording=wn,
                    believes_real=int(believes_real), world=wl, belief=bl)
                add("D2", s, q_belief(country, x), kind="belief", fi=i, half=i % 2, frame=frame, wording=wn,
                    believes_real=int(believes_real), world=wl, belief=bl)
# D3: reads at the target claims
UNREL = [i for i in range(len(FACTS)) if i % 2 == 1][:8]
READ_CTX = [("none", "none", REGIME_FIRM, ""), ("firm_portray", "portray", REGIME_FIRM, ""),
            ("firm_identify", "identify", REGIME_FIRM, ""), ("k3_portray", "portray", K3, ""),
            ("k3_identify", "identify", K3, ""), ("k3_identify_char", "identify", K3, "As Elian, "),
            ("k3_identify_remind", "identify", K3, REMIND + " "), ("k3_portray_char", "portray", K3, "As Elian, "),
            ("k3_portray_remind", "portray", K3, REMIND + " ")]
for ci, c in enumerate(CLAIMS):
    country, true, false = c
    for rname, frame, reg, pre in READ_CTX:
        s = system_for(frame, c, reg)
        for x, wl in ((true, 1), (false, 0)):
            add("D3", s, pre + q_world(country, x), kind="world", ci=ci, rctx=rname, world=wl, belief=1 - wl)
            add("D3", s, pre + q_belief(country, x), kind="belief", ci=ci, rctx=rname, world=wl, belief=1 - wl)
        for i in UNREL:
            fc, ft, fw = FACTS[i]
            for x, wl in ((ft, 1), (fw, 0)):
                add("D3", s, pre + q_world(fc, x), kind="unrel", ci=ci, rctx=rname, fi=i, world=wl)

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)


def render(system, user):
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


texts = [render(it["system"], it["user"]) for it in ITEMS]
enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
lens = np.array([len(e) for e in enc])
log(f"{len(ITEMS)} items: " + ", ".join(f"{d}={sum(it['ds'] == d for it in ITEMS)}" for d in ("D1", "D2", "D3"))
    + f"; {len(CTX)} D1 contexts; tokens min/median/max {lens.min()}/{int(np.median(lens))}/{lens.max()}")
meta = [{k: v for k, v in it.items() if k not in ("system", "user")} for it in ITEMS]
json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"))
if DRY:
    for j in (0, len(CTX) * 0 + 7 * 160, sum(it["ds"] == "D1" for it in ITEMS) + 3,
              len(ITEMS) - 1 - 16 - 3):
        print("=" * 30, meta[j], "\n" + texts[j])
    log(f"{len(set(texts))} unique prompts (duplicates: D3 reads that repeat D1 contexts)")
    sys.exit(0)

# ------------------------------------------------------------------ model + features
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

DEV = "cpu" if MOCK else "cuda"
if not MOCK:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa")
    model.eval()
    inner = model.model
    layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
log("model loaded")


@torch.no_grad()
def feats(idx_list, bs):
    """{L: float16 cpu tensor [len(idx_list), d]} of last-token residuals (block outputs)."""
    if MOCK:
        g = torch.Generator().manual_seed(0)
        base = torch.randn(len(idx_list), 64, generator=g)
        lab = torch.tensor([float(meta[j].get("label", meta[j].get("world", 0))) for j in idx_list])
        bel = torch.tensor([float(meta[j].get("belief", 0)) for j in idx_list])
        base[:, 0] += 2 * lab
        base[:, 1] += 2 * bel
        return {L: (base * (1 + L / 100)).half() for L in CAND}
    order = sorted(range(len(idx_list)), key=lambda k: lens[idx_list[k]])
    res = {}
    for b in range(0, len(order), bs):
        ks = order[b:b + bs]
        ls = [int(lens[idx_list[k]]) for k in ks]
        T = max(ls)
        ids = torch.full((len(ks), T), PAD, dtype=torch.long)
        for r, k in enumerate(ks):
            ids[r, :ls[r]] = torch.tensor(enc[idx_list[k]])
        rows_ix, last = torch.arange(len(ks)).cuda(), (torch.tensor(ls) - 1).cuda()
        acts = {}

        def mk(L):
            def h(m, i, o):
                hsd = o[0] if isinstance(o, tuple) else o
                acts[L] = hsd[rows_ix, last]
            return h
        hs = [layers[L - 1].register_forward_hook(mk(L)) for L in CAND]
        try:
            try:
                model(input_ids=ids.cuda(), use_cache=False, logits_to_keep=1)
            except TypeError:
                model(input_ids=ids.cuda(), use_cache=False)
        finally:
            for h in hs:
                h.remove()
        for L in CAND:
            if L not in res:
                res[L] = torch.empty(len(idx_list), acts[L].shape[-1], dtype=torch.float16)
            res[L][torch.tensor(ks)] = acts[L].float().cpu().half()
    return res


# padding check: 8 long + short items, batched vs one at a time
chk = list(range(0, len(ITEMS), len(ITEMS) // 8))[:8]
fb, f1 = feats(chk, 8), feats(chk, 1)
rel = max(float((fb[L].float() - f1[L].float()).norm() / f1[L].float().norm()) for L in CAND)
log(f"padding check: max relative diff batched vs single {rel:.4f}")
BS = 16 if rel < 0.05 else 1   # bf16 kernel noise is ~1e-2; padding leakage would be far larger
if BS == 1:
    log("WARNING: batched features differ; falling back to batch size 1")
first = {}
rep = [first.setdefault(t, j) for j, t in enumerate(texts)]   # identical prompts are computed once
uniq = sorted(set(rep))
pos = {j: k for k, j in enumerate(uniq)}
Fu = feats(uniq, BS)
gather = torch.tensor([pos[r] for r in rep])
F = {L: Fu[L][gather] for L in CAND}
del Fu
log(f"features done for {len(ITEMS)} items ({len(uniq)} unique forward rows)")

# ------------------------------------------------------------------ analysis helpers
dev = DEV
M = {k: np.array([m.get(k, -9) if not isinstance(m.get(k, -9), str) else m.get(k) for m in meta], dtype=object)
     for k in ("ds", "ctx", "ctype", "claim", "wording", "frame", "fi", "half", "label", "kind", "believes_real",
               "world", "belief", "ci", "rctx")}


def mask(**kw):
    m = np.ones(len(meta), bool)
    for k, v in kw.items():
        m &= np.isin(M[k], v if isinstance(v, (list, tuple)) else [v])
    return m


def X(L, m):
    return F[L][torch.from_numpy(np.where(m)[0])].to(dev).float()


def fit(L, m, labkey):
    Xm, y = X(L, m), torch.tensor(M[labkey][m].astype(int), device=dev).bool()
    w = Xm[y].mean(0) - Xm[~y].mean(0)
    s = Xm @ w
    thr = (s[y].mean() + s[~y].mean()) / 2
    return {"w": w, "thr": thr, "sd": s.std(), "L": L}


def score(p, m):
    return ((X(p["L"], m) @ p["w"] - p["thr"]) / p["sd"]).cpu().numpy()


def acc(p, m, labkey):
    return float(((score(p, m) > 0).astype(int) == M[labkey][m].astype(int)).mean())


def gap(p, m, keys=("ctx", "fi")):
    """mean over (context, country) units of score(true) - score(false), for D1-style rows with 'label'."""
    s = score(p, m)
    idx = np.where(m)[0]
    d = {}
    for j, sc in zip(idx, s):
        u = tuple(M[k][j] for k in keys)
        d.setdefault(u, {})[int(M["label"][j])] = sc
    g = [v[1] - v[0] for v in d.values() if 0 in v and 1 in v]
    return float(np.mean(g)), float(np.mean(np.array(g) > 0))


D1 = mask(ds="D1")
TR, TE = D1 & mask(half=0), D1 & mask(half=1)
E = mask(ctype="E")
res = {"padding_rel_diff": rel, "batch_size": BS, "layers": CAND}

# ------------------------------------------------------------------ 1. truth probe: LOCO by layer, layer choice
loco = {}
for L in CAND:
    per = []
    for ci in range(8):
        p = fit(L, TR & ~(E & mask(claim=ci)), "label")
        held = TE & E & mask(claim=ci)
        g_neu = gap(p, TE & mask(ctx=0))[0]
        g_k3 = gap(p, held & mask(wording="k3"))[0]
        per.append({"ci": ci, "acc": acc(p, held, "label"), "gap": gap(p, held)[0], "paired": gap(p, held)[1],
                    "gap_neutral": g_neu, "gap_k3": g_k3, "gap_firm": gap(p, held & mask(wording="firm"))[0],
                    "acc_neutral_test": acc(p, TE & mask(ctx=0), "label")})
    loco[L] = {"acc": float(np.mean([r["acc"] for r in per])),
               "spec_ratio_k3": float(np.mean([r["gap_k3"] for r in per]) / np.mean([r["gap_neutral"] for r in per])),
               "per_claim": per}
    log(f"truth L{L}: LOCO acc {loco[L]['acc']:.3f}, K3 specificity ratio {loco[L]['spec_ratio_k3']:.2f}")
TL = max(CAND, key=lambda L: loco[L]["acc"])
res["truth"] = {"layer": TL, "loco": {L: {k: v for k, v in d.items()} for L, d in loco.items()}}
res["truth"]["valid"] = bool(loco[TL]["acc"] >= 0.85 and loco[TL]["spec_ratio_k3"] >= 0.5)
log(f"truth layer {TL}: valid={res['truth']['valid']}")
json.dump(res, open(os.path.join(out_dir, "probe_b4.json"), "w"), indent=1, default=float)

# belief-probe layer choice (needed before saving features)
D2 = mask(ds="D2")
B_TR, B_TE = D2 & mask(kind="belief", half=0), D2 & mask(kind="belief", half=1)
bl = {L: acc(fit(L, B_TR, "belief"), B_TE, "belief") for L in CAND}
BL = max(CAND, key=bl.get)
log(f"belief held-out acc by layer { {L: round(a, 3) for L, a in bl.items()} } -> L{BL}")

# save features at TL and BL in <= ~8 MB chunks (for local re-analysis; numpy fp16)
for L in sorted({TL, BL}):
    a = F[L].numpy()
    for k, st in enumerate(range(0, len(a), 800)):
        np.save(os.path.join(out_dir, f"feat_L{L}_{k:02d}.npy"), a[st:st + 800])
log("features saved")


def section(name, fn):
    try:
        fn()
    except Exception:  # noqa: BLE001
        res.setdefault("errors", {})[name] = traceback.format_exc()[-1500:]
        log(f"ERROR in {name}:\n{traceback.format_exc()[-1500:]}")
    json.dump(res, open(os.path.join(out_dir, "probe_b4.json"), "w"), indent=1, default=float)


def s_validation():
    pt = fit(TL, TR & mask(ctype=["N", "U"]), "label")
    p0 = fit(TL, TR & mask(ctx=0), "label")
    pf = fit(TL, TR, "label")
    v = {}
    for nm, p in (("type_NU", pt), ("v0_neutral_only", p0), ("full", pf)):
        v[nm] = {"acc_E_test": acc(p, TE & E, "label"), "gap_E_test": gap(p, TE & E)[0],
                 "gap_E_k3_test": gap(p, TE & E & mask(wording="k3"))[0],
                 "acc_neutral_test": acc(p, TE & mask(ctx=0), "label"), "gap_neutral_test": gap(p, TE & mask(ctx=0))[0],
                 "acc_U_test": acc(p, TE & mask(ctype="U"), "label"), "gap_U_test": gap(p, TE & mask(ctype="U"))[0],
                 "acc_N_test": acc(p, TE & mask(ctype="N"), "label")}
    # V-B: world statements in B contexts (test half), by what Elian believes
    W = D2 & mask(kind="world", half=1)
    for nm, p in (("full", pf), ("v0_neutral_only", p0)):
        for br in (0, 1):
            m = W & mask(believes_real=br)
            s = score(p, m)
            idx = np.where(m)[0]
            d = {}
            for j, sc in zip(idx, s):
                d.setdefault((M["fi"][j], M["frame"][j], M["wording"][j]), {})[int(M["world"][j])] = (sc, M["wording"][j])
            g = {wn: [u[1][0] - u[0][0] for u in d.values() if u[1][1] == wn] for wn in ("firm", "k3")}
            v.setdefault(f"VB_{nm}", {})[f"believes_{'real' if br else 'wrong'}"] = {
                "gap": float(np.mean(g["firm"] + g["k3"])), "gap_firm": float(np.mean(g["firm"])),
                "gap_k3": float(np.mean(g["k3"])), "acc": float(((s > 0).astype(int) == M["world"][m].astype(int)).mean())}
    # truth probe on belief statements (test half)
    v["truth_on_belief_statements"] = {"acc_vs_belief": acc(pf, B_TE, "belief"), "acc_vs_world": acc(pf, B_TE, "world")}
    res["truth"]["validation"] = v
    res["truth"]["layer_full_probe_cos_with_v0"] = float(torch.nn.functional.cosine_similarity(pf["w"], p0["w"], 0))


def s_belief():
    pb = fit(BL, B_TR, "belief")
    pt_bl = fit(BL, TR, "label")
    pt_tl = fit(TL, TR, "label")
    pb_tl = fit(TL, B_TR, "belief")
    res["belief"] = {"layer": BL, "acc_by_layer": bl, "acc_test": acc(pb, B_TE, "belief"),
                     "acc_world_on_same_items": acc(pb, B_TE, "world"),
                     "acc_on_world_statements_vs_world": acc(pb, D2 & mask(kind="world", half=1), "world"),
                     "acc_on_world_statements_vs_belief": acc(pb, D2 & mask(kind="world", half=1), "belief"),
                     "cos_truth_belief_at_BL": float(torch.nn.functional.cosine_similarity(pb["w"], pt_bl["w"], 0)),
                     "cos_truth_belief_at_TL": float(torch.nn.functional.cosine_similarity(pb_tl["w"], pt_tl["w"], 0))}
    a = res["belief"]
    a["valid"] = bool(a["acc_test"] >= 0.85 and 0.35 <= a["acc_world_on_same_items"] <= 0.65)
    log(f"belief probe L{BL}: acc {a['acc_test']:.3f}, world-acc on same items {a['acc_world_on_same_items']:.3f}, "
        f"cos(truth, belief) {a['cos_truth_belief_at_BL']:.2f}; valid={a['valid']}")


def s_reads():
    pb = fit(BL, B_TR, "belief")
    rows = []
    for ci in range(8):
        pt = fit(TL, TR & ~(E & mask(claim=ci)), "label")
        p0 = fit(TL, TR & mask(ctx=0), "label")
        for rname, *_ in READ_CTX:
            base = mask(ds="D3", ci=ci, rctx=rname)
            r = {"ci": ci, "country": CLAIMS[ci][0], "rctx": rname}
            for kind in ("world", "belief"):
                for wl in (1, 0):
                    m = base & mask(kind=kind, world=wl)
                    tag = f"{kind}_{'real' if wl else 'persona'}"
                    r[f"truth_{tag}"] = float(score(pt, m)[0])
                    r[f"v0_{tag}"] = float(score(p0, m)[0])
                    r[f"belief_{tag}"] = float(score(pb, m)[0])
            mu = base & mask(kind="unrel")
            for nm, p in (("truth", pt), ("v0", p0)):
                s = score(p, mu)
                wv = M["world"][mu].astype(int)
                r[f"{nm}_unrel_gap"] = float(s[wv == 1].mean() - s[wv == 0].mean())
                r[f"{nm}_unrel_acc"] = float(((s > 0).astype(int) == wv).mean())
            r["truth_gap"] = r["truth_world_real"] - r["truth_world_persona"]
            r["v0_gap"] = r["v0_world_real"] - r["v0_world_persona"]
            r["belief_gap"] = r["belief_belief_persona"] - r["belief_belief_real"]
            r["belief_gap_on_world_statements"] = r["belief_world_persona"] - r["belief_world_real"]
            rows.append(r)
    with open(os.path.join(out_dir, "reads.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = {}
    for rname, *_ in READ_CTX:
        rr = [r for r in rows if r["rctx"] == rname]
        summ[rname] = {k: float(np.mean([r[k] for r in rr])) for k in
                       ("truth_gap", "truth_unrel_gap", "v0_gap", "v0_unrel_gap", "belief_gap",
                        "belief_gap_on_world_statements", "truth_unrel_acc")}
        log(f"read {rname:20s} " + " ".join(f"{k}={v:+.2f}" for k, v in summ[rname].items()))
    res["reads_summary"] = summ


section("validation", s_validation)
section("belief", s_belief)
section("reads", s_reads)
log("done")
