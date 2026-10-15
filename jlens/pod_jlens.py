"""J-lens reads on Qwen3.6-27B (prereg jlens/README.md). One process: self-checks, gate, then the claim reads.

Lenses (neuronpedia/jacobian-lens): 'jlens' = the Jacobian lens fitted by M. Piotrowski (n=1000 WikiText prompts);
'jpp' = the J++ lens (Ayonrinde & Lindsey); 'logit' = the logit lens (J = I). Readout at layer l for a block output h:
lm_head(final_norm(J_l h)), as the jlens reference implementation (anthropics/jacobian-lens, jlens/lens.py + hf.py).

Gate (before any claim read):
  G1  "Fact: The number of legs on the animal that spins webs is " -> ' spider' in the J-lens top 10 at some layer in
      20-60 at the last position, and the model's top-1 next token is '8'. Fail -> stop, no claim reads.
  G2  (reported) the paper's multihop lens eval, pass@10: J-lens vs logit lens.
  G3  (reported) swap ' spider' -> ' ant' with J-lens vectors at every layer 20-60, every prompt position (paper's
      h' = h + V(sigma(c) - c), c = V^+ h, sigma swaps the two coordinates, scaled by alpha); and the paper's 90-item
      probe-swap set. Lens vector for token t at layer l: J_l^T (g * W_U[t]), g = the final norm's gain.
Claim reads: jlens/data/claims.jsonl (build_claims.py), block outputs at layers 20, 24, ..., 60 and 63 at the read
positions; per lens: top-50 tokens, log-mass and best rank of the pre-registered token sets, log-mass of 200 random sets.
  python -m jlens.pod_jlens --out jlens/results [--limit N]"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

MODEL = "Qwen/Qwen3.6-27B"
REPO = "neuronpedia/jacobian-lens"
LENS_FILES = {"jlens": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
              "jpp": "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt"}
BAND = list(range(20, 61, 4))
SWAP_BAND = list(range(20, 61))
FINAL = 63
TOPK = 50
HERE = Path(__file__).resolve().parent
SPIDER = "Fact: The number of legs on the animal that spins webs is "
T0 = time.time()


def log(*a):
    print(f"[jlens {time.time() - T0:7.0f}s]", *a, flush=True)


def load_model(tiny=False):
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tiny:   # CPU test only: the real config shrunk, random weights
        cfg = AutoConfig.from_pretrained(MODEL)
        t = cfg.text_config
        t.hidden_size, t.intermediate_size, t.num_attention_heads, t.num_key_value_heads, t.head_dim = 64, 128, 2, 1, 32
        t.linear_num_value_heads, t.linear_num_key_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 2, 16, 16
        v = cfg.vision_config
        v.hidden_size, v.depth, v.num_heads, v.intermediate_size, v.out_hidden_size = 32, 1, 2, 64, 64
        torch.manual_seed(0)
        model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16).eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    text = None
    for path in ("model.language_model", "model", "language_model"):
        try:
            obj = model
            for a in path.split("."):
                obj = getattr(obj, a)
        except AttributeError:
            continue
        if all(hasattr(obj, a) for a in ("layers", "norm", "embed_tokens")):
            text = obj
            break
    assert text is not None and len(text.layers) == 64, "text decoder not found"
    return tok, model, text


class Reader:
    def __init__(self, tok, model, text):
        self.tok, self.model, self.text = tok, model, text
        self.blocks, self.norm, self.lm_head = text.layers, text.norm, model.lm_head
        self.lenses = {}
        self.dev = self.lm_head.weight.device

    def add_lens(self, name, path):
        ck = torch.load(path, map_location="cpu", weights_only=True) if isinstance(path, str) else path
        info = {k: (v if k != "J" else f"{len(v)} layers") for k, v in ck.items()}
        log(f"lens {name}: keys {sorted(ck)}; source_layers {ck.get('source_layers', sorted(ck['J']))[:3]}.."
            f"{ck.get('source_layers', sorted(ck['J']))[-1]}; n_prompts {ck.get('n_prompts')}; d_model {ck.get('d_model')}")
        self.lenses[name] = {int(k): v.to(self.dev, torch.float16) for k, v in ck["J"].items()}   # ~3.3 GB on GPU
        return {k: (str(v)[:2000]) for k, v in info.items()}

    def forward(self, ids, record, positions=None, hooks=()):
        """Block outputs at `record` layers, at `positions` (default all), fp32 on CPU: {layer: [n_pos, d]}."""
        acts, handles = {}, []

        def mk(i):
            def h(mod, inp, out):
                t = out if torch.is_tensor(out) else out[0]
                acts[i] = (t[0] if positions is None else t[0, positions]).float().cpu()
            return h
        for i, fn in hooks:
            handles.append(self.blocks[i].register_forward_hook(fn))
        for i in record:
            handles.append(self.blocks[i].register_forward_hook(mk(i)))
        try:
            with torch.no_grad():
                self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        return acts

    def logits(self, resid, layer, lens):
        """resid [n, d] fp32 -> lens logits [n, V] fp32 on GPU."""
        h = resid.to(self.dev)
        if lens != "logit":
            h = h @ self.lenses[lens][layer].float().T
        return self.lm_head(self.norm(h.to(self.lm_head.weight.dtype))).float()

    def lens_vectors(self, layer, token_ids, lens="jlens"):
        """[len(token_ids), d] fp32: J_l^T (g * W_U[t])."""
        d = self.lm_head.weight.shape[1]
        g = self.norm(torch.ones(1, d, device=self.dev, dtype=self.lm_head.weight.dtype)).float()[0]
        wu = self.lm_head.weight[token_ids].float() * g
        return wu @ self.lenses[lens][layer].float()


def best_rank(logits, ids):
    """1-based rank of the best-ranked of `ids` in each row: [n]."""
    return (logits > logits[:, ids].max(1, keepdim=True).values).sum(-1) + 1


def single_forms(tok, w):
    out = []
    for c in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in out:
            out.append(ids[0])
    return out


def selfcheck(R):
    ids = R.tok(SPIDER, add_special_tokens=False)["input_ids"]
    acts = R.forward(ids, [FINAL], positions=[len(ids) - 1])
    mine = R.logits(acts[FINAL], FINAL, "logit")[0]
    with torch.no_grad():
        ref = R.model(input_ids=torch.tensor([ids], device=R.dev)).logits[0, -1].float()
    diff = (mine - ref).abs().max().item()
    log(f"selfcheck: hook+unembed vs model logits max |diff| {diff:.4f} (top1 {R.tok.decode([int(mine.argmax())])!r} vs "
        f"{R.tok.decode([int(ref.argmax())])!r})")
    assert int(mine.argmax()) == int(ref.argmax()) and diff < 0.5, "hooked readout does not match the model"
    return diff


def gate(R, out):
    tok, res = R.tok, {}
    ids = tok(SPIDER, add_special_tokens=False)["input_ids"]
    layers = sorted(R.lenses["jlens"])
    acts = R.forward(ids, layers + [FINAL], positions=[len(ids) - 1])
    model_top = R.logits(acts[FINAL], FINAL, "logit")[0].topk(5).indices.tolist()
    res["spider_model_top5"] = [tok.decode([t]) for t in model_top]
    sp = [tok.encode(" spider", add_special_tokens=False)[0], tok.encode(" spiders", add_special_tokens=False)[0]]
    res["spider"] = {}
    for lens in ("jlens", "jpp", "logit"):
        rows = {}
        for L in layers:
            lg = R.logits(acts[L], L, lens)
            rows[L] = {"rank_spider": int(best_rank(lg, sp)[0]),
                       "top10": [tok.decode([t]) for t in lg[0].topk(10).indices.tolist()]}
        res["spider"][lens] = rows
    best = min(res["spider"]["jlens"][L]["rank_spider"] for L in BAND)
    res["G1"] = {"best_rank_band": best, "model_top1": res["spider_model_top5"][0],
                 "pass": best <= 10 and res["spider_model_top5"][0].strip() == "8"}
    log("G1", res["G1"])
    for L in BAND:
        log(f"  L{L} jlens rank(spider) {res['spider']['jlens'][L]['rank_spider']:>6}  logit {res['spider']['logit'][L]['rank_spider']:>6}"
            f"  jlens top10 {res['spider']['jlens'][L]['top10']}")

    # G2: multihop lens eval (pass@10 over all source layers and over the band)
    items = json.load(open(HERE / "data" / "lens-eval-multihop.json"))["items"]
    hits = {lens: {"all": [], "band": []} for lens in ("jlens", "jpp", "logit")}
    for k, it in enumerate(items):
        ids = tok(it["prompt"], add_special_tokens=False)["input_ids"]
        acts = R.forward(ids, layers, positions=[len(ids) - 1])
        inter = [single_forms(tok, w) for w in it["intermediates"]]
        inter = [f for f in inter if f]
        if not inter:
            continue
        for lens in hits:
            rk = {}
            for L in layers:
                lg = R.logits(acts[L], L, lens)
                rk[L] = [int(best_rank(lg, f)[0]) for f in inter]
            for scope, Ls in (("all", layers), ("band", BAND)):
                hits[lens][scope].append(np.mean([min(rk[L][j] for L in Ls) <= 10 for j in range(len(inter))]))
        if k % 20 == 0:
            log(f"multihop {k}/{len(items)}")
    res["G2"] = {lens: {s: float(np.mean(v)) for s, v in d.items()} | {"n": len(d["all"])} for lens, d in hits.items()}
    log("G2 multihop pass@10", res["G2"])

    # G3: swaps (lens vectors cached per layer for every token the swaps use)
    items_ps = json.load(open(HERE / "data" / "probe-swap.json"))["items"]
    need = {tok.encode(" spider", add_special_tokens=False)[0], tok.encode(" ant", add_special_tokens=False)[0]}
    for it in items_ps:
        for w in (it["intermediate"], it["swap_to"]):
            e = tok.encode(" " + w, add_special_tokens=False)
            if len(e) == 1:
                need.add(e[0])
    need = sorted(need)
    col = {t: i for i, t in enumerate(need)}
    vec = {L: R.lens_vectors(L, need) for L in SWAP_BAND}         # [len(need), d] per layer

    def swap_hooks(s, t, alpha):
        hooks = []
        for L in SWAP_BAND:
            V = vec[L][[col[s], col[t]]].T                       # [d, 2]
            P = torch.linalg.pinv(V)                             # [2, d]

            def fn(mod, inp, out, V=V, P=P):
                x = out if torch.is_tensor(out) else out[0]
                h = x.float()
                c = h @ P.T                                      # [1, n, 2]
                h = h + (alpha * c.flip(-1) - c) @ V.T
                h = h.to(x.dtype)
                return h if torch.is_tensor(out) else (h,) + tuple(out[1:])
            hooks.append((L, fn))
        return hooks

    def next_tok(ids, hooks=()):
        acts = R.forward(ids, [FINAL], positions=[len(ids) - 1], hooks=hooks)
        return torch.log_softmax(R.logits(acts[FINAL], FINAL, "logit")[0], -1)

    ids = tok(SPIDER, add_special_tokens=False)["input_ids"]
    s, t = sp[0], tok.encode(" ant", add_special_tokens=False)[0]
    eight, six = tok.encode("8", add_special_tokens=False)[0], tok.encode("6", add_special_tokens=False)[0]
    res["G3_spider"] = {}
    for alpha in (0.0, 1.0, 2.0, 4.0):
        lp = next_tok(ids, swap_hooks(s, t, alpha) if alpha else ())
        res["G3_spider"][str(alpha)] = {"top5": [tok.decode([i]) for i in lp.topk(5).indices.tolist()],
                                        "p8": float(lp[eight].exp()), "p6": float(lp[six].exp())}
        log(f"G3 spider->ant alpha {alpha}: {res['G3_spider'][str(alpha)]}")
    items = items_ps
    tally = {a: {"n": 0, "base": 0, "swap": 0} for a in (1.0, 2.0)}
    for it in items:
        sid = tok.encode(" " + it["intermediate"], add_special_tokens=False)
        tid = tok.encode(" " + it["swap_to"], add_special_tokens=False)
        if len(sid) != 1 or len(tid) != 1:
            continue
        sp_ = "" if it["prompt"].endswith(" ") else " "
        a_id = tok.encode(sp_ + it["answer"], add_special_tokens=False)[0]
        b_id = tok.encode(sp_ + it["swap_answer"], add_special_tokens=False)[0]
        ids = tok(it["prompt"], add_special_tokens=False)["input_ids"]
        base_ok = int(next_tok(ids).argmax()) == a_id
        for a in tally:
            tally[a]["n"] += 1
            tally[a]["base"] += base_ok
            tally[a]["swap"] += base_ok and int(next_tok(ids, swap_hooks(sid[0], tid[0], a)).argmax()) == b_id
    res["G3_probe_swap"] = {str(a): v for a, v in tally.items()}
    log("G3 probe-swap (single-token items; swap counted among baseline-correct)", res["G3_probe_swap"])
    (out / "gate.json").write_text(json.dumps(res, indent=1))
    return res["G1"]["pass"]


def claim_reads(R, out, limit):
    rows = [json.loads(l) for l in open(HERE / "data" / "claims.jsonl")][:limit]
    sets = json.load(open(HERE / "data" / "tokensets.json"))
    named = ["NEG", "DECEIT", "DONE", "EVAL"]
    set_names = named + sorted(k for k in sets if k.startswith("rand"))
    layers = BAND + [FINAL]
    P = max(1 + len(r["window"]) for r in rows)
    resid = torch.zeros(len(rows), len(layers), P, R.lm_head.weight.shape[1])
    meta = []
    for k, r in enumerate(rows):
        pos = [r["pos"]["turn_start"]] + r["window"]
        acts = R.forward(r["ids"], layers, positions=pos)
        for j, L in enumerate(layers):
            resid[k, j, : len(pos)] = acts[L]
        meta.append({kk: r[kk] for kk in r if kk not in ("ids",)} | {"read_pos": pos})
        if k % 10 == 0:
            log(f"reads {k}/{len(rows)} ({len(r['ids'])} tokens)")
    lenses = ["jlens", "jpp", "logit"]
    n, nl = len(rows), len(layers)
    top_ids = np.full((n, len(lenses), nl, P, TOPK), -1, np.int32)
    top_lp = np.zeros((n, len(lenses), nl, P, TOPK), np.float16)
    mass = np.full((n, len(lenses), nl, P, len(set_names)), np.nan, np.float32)
    best = np.zeros((n, len(lenses), nl, P, len(named)), np.int32)
    set_idx = [torch.tensor(sets[s], device=R.dev) for s in set_names]
    for li, lens in enumerate(lenses):
        for j, L in enumerate(layers):
            if lens != "logit" and L == FINAL:
                lens_ = "logit"                                # no J for the final block: the model's own output
            else:
                lens_ = lens
            X = resid[:, j].reshape(n * P, -1)
            for c0 in range(0, n * P, 128):
                lg = R.logits(X[c0:c0 + 128], L, lens_)
                lp = torch.log_softmax(lg, -1)
                tv, ti = lp.topk(TOPK)
                ix = np.arange(c0, min(c0 + 128, n * P))
                a, b = ix // P, ix % P
                top_ids[a, li, j, b] = ti.cpu().numpy()
                top_lp[a, li, j, b] = tv.cpu().numpy()
                mass[a, li, j, b] = torch.stack([torch.logsumexp(lp[:, si], -1) for si in set_idx], 1).cpu().numpy()
                best[a, li, j, b] = torch.stack([best_rank(lg, sets[s]) for s in named], 1).cpu().numpy()
            log(f"readout {lens} L{L} done")
    valid = np.array([[p < 1 + len(r["window"]) for p in range(P)] for r in rows])
    np.savez_compressed(out / "reads.npz", top_ids=top_ids, top_lp=top_lp, mass=mass, best=best, valid=valid,
                        layers=np.array(layers), lenses=np.array(lenses), set_names=np.array(set_names))
    with open(out / "reads_meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    log(f"claim reads saved: {n} reads, P={P}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tiny", action="store_true", help="CPU test on a shrunk random model with random lenses")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import transformers
    tok, model, text = load_model(a.tiny)
    R = Reader(tok, model, text)
    info = {"model": MODEL, "model_revision": getattr(model.config, "_commit_hash", None), "torch": torch.__version__,
            "transformers": transformers.__version__, "git": os.environ.get("GIT_REV"), "lenses": {},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    if a.tiny:
        d = text.config.hidden_size
        for name in LENS_FILES:
            R.add_lens(name, {"J": {L: torch.eye(d) + 0.1 * torch.randn(d, d) for L in range(63)}, "n_prompts": 0,
                              "d_model": d, "source_layers": list(range(63))})
        info["selfcheck_maxdiff"] = selfcheck(R)
        gate(R, out)
        claim_reads(R, out, a.limit)
        log("tiny test done")
        return
    for name, fn in LENS_FILES.items():
        p = hf_hub_download(REPO, fn)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        info["lenses"][name] = {"file": fn, "sha256": h.hexdigest(), "meta": R.add_lens(name, p)}
    info["selfcheck_maxdiff"] = selfcheck(R)
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    ok = gate(R, out)
    if not ok:
        log("GATE G1 FAILED: no claim reads")
        raise SystemExit(3)
    claim_reads(R, out, a.limit)
    log("all done")


if __name__ == "__main__":
    main()
