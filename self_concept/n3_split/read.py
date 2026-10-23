"""SC-N3 reads (DESIGN.md) on one cell's merged model (HF), from the replies V1's runner generated on that cell.

For each record whose reply holds a choose_option call and was not nudged:
  P0, D    the last prompt token, and the decision position (the last token shared by the reply's prefix + each option
           id; the next token is the option id's first token)
  resid    block outputs at items.PROBE_LAYERS at P0 and D (fp16), for the habit probe
  lens     each lens in LENS (+ the logit lens), every source layer, at P0 and D: log-mass of UNI / REF and of 200
           random sets of each size
  lo       log p(penalise) - log p(refer) at D (first tokens), teacher-forced on the record's own prefix
  mean_lp  NZ cells: mean primary-lens log-prob over the selection-half records at P0, per deep-band layer (B2)
B2 (NZ-SAYU, after NZ-PLAC was read on the same pod): word-pool tokens ranked by the deep-band mean of mean_lp(NZ-SAYU)
- mean_lp(NZ-PLAC). Arms top5, top10, bottom10, rand10 (seeded): the span of their lens vectors is projected out of
every deep-band block output at positions P0..D; lo and the top-1 token at D on every test-half record.

  python -m self_concept.n3_split.read --cell NZ-SAYU_s1 --model DIR --evals DIR --out DIR [--tiny]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from persona_affect.common.util import seed
from persona_affect.v1_value_hook.fmt import prompt_ids
from self_concept.common import lens as L
from self_concept.n3_split import items as I

LENS = [("cb", "camilablank/workspace-lenses", "qwen3.5-9b/j-lens/lens.pt", L.CB[1])]
N_RAND = 200
MARK = "<parameter=option_id>"
T0 = time.time()


def log(*a):
    print(f"[n3-read {time.time() - T0:6.0f}s]", *a, flush=True)


def decision(tok, p_ids, reply):
    """(teacher-forced ids ending at D, {option: first token id}) or None when the reply has no option_id."""
    j = reply.find(MARK)
    if j < 0:
        return None
    pre = reply[:j + len(MARK)]
    joint = {o: tok(pre + "\n" + o, add_special_tokens=False)["input_ids"] for o in I.OPTIONS}
    seqs = list(joint.values())
    k = 0
    while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
        k += 1
    first = {o: s[k] for o, s in joint.items() if len(s) > k}
    if len(first) != len(I.OPTIONS) or len(set(first.values())) != len(first):
        return None
    return p_ids + seqs[0][:k], first


def records(evals, cell):
    path = Path(evals) / f"eval_{cell}.jsonl"
    out = []
    for x in path.read_text().splitlines():
        r = json.loads(x)
        if not r.get("nudged") and r.get("role") in ("unilateral", "report", "note"):
            out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="n3 read")
    ap.add_argument("--cell", required=True, choices=I.CELLS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--evals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, layers, norm, head, cap = L.load_model("qwen9", tiny=a.tiny, path=None if a.tiny else a.model)
    n_layers = len(layers)
    R = L.Reader(tok, model, layers, norm, head, cap)
    by_key = {it["key"]: it for it in I.items()}
    recs = [r for r in records(a.evals, a.cell) if r["key"] in by_key]
    sp = I.split()
    if a.tiny:
        d = head.weight.shape[1]
        lenses = [("cb", {"J": {i: torch.eye(d) for i in range(n_layers - 1)}, "source_layers":
                          list(range(n_layers - 1)), "n_prompts": 1, "d_model": d})]
    else:
        from huggingface_hub import hf_hub_download
        lenses = [(n, hf_hub_download(repo, fn, revision=rev)) for n, repo, fn, rev in LENS]
    metas = {n: R.add_lens(n, ck) for n, ck in lenses}
    src = {n: [x for x in m["source_layers"] if x < n_layers - 1] for n, m in metas.items()}
    src["logit"] = list(range(n_layers - 1))
    prim = LENS[0][0]
    deep = L.deep_band(src[prim], n_layers)
    probe_layers = [x for x in I.PROBE_LAYERS if x < n_layers] if not a.tiny else [1, 2]
    set_ids = {k: L.word_ids(tok, w) for k, w in I.SETS.items()}
    pool = L.word_pool(tok)
    sizes = sorted({len(v) for v in set_ids.values()})
    rand = {s: L.random_sets(pool, s, N_RAND, seed(f"SC-N3|rand|{s}")) for s in sizes}
    record_layers = sorted(set(range(n_layers)))
    meta, lo, resid = [], [], []
    fixed = {n: np.zeros((len(recs), 2, len(src[n]), 2), np.float32) for n in src}
    rnd = {n: np.zeros((len(recs), 2, len(src[n]), N_RAND * len(sizes)), np.float32) for n in src}
    mean_lp, n_sel = None, 0
    kept = []
    for j, r in enumerate(recs):
        it = by_key[r["key"]]
        p_ids = prompt_ids(tok, it["messages"], it["tools"])
        dec = decision(tok, p_ids, r["text"])
        if dec is None:
            continue
        ids, first = dec
        p0, dpos = len(p_ids) - 1, len(ids) - 1
        acts, z = R.forward(ids, record_layers, pos=[p0, dpos], logits=True)
        lp_next = torch.log_softmax(z, -1)
        kept.append(j)
        lo.append(float(lp_next[first["penalise"]] - lp_next[first["refer"]]))
        resid.append(np.stack([acts[x].numpy().astype(np.float16) for x in probe_layers], 1))   # [2, n_probe, d]
        meta.append({"key": r["key"], "tid": r["tid"], "role": r["role"], "half": sp[r["tid"]], "p0": p0, "d": dpos,
                     "reason": r["text"], "first": first})
        for n in src:
            for li, layer in enumerate(src[n]):
                lp = R.logprobs(acts[layer], layer, n)                           # [2, V]
                for si, k in enumerate(("UNI", "REF")):
                    fixed[n][j, :, li, si] = L.logmass(lp, set_ids[k]).cpu().numpy()
                col = 0
                for s in sizes:
                    idx = torch.tensor(rand[s], device=lp.device)
                    rnd[n][j, :, li, col:col + N_RAND] = torch.logsumexp(lp[:, idx], -1).cpu().numpy()
                    col += N_RAND
                if n == prim and layer in deep and a.cell.startswith("NZ") and sp[r["tid"]] == "sel":
                    if mean_lp is None:
                        mean_lp = torch.zeros(len(deep), lp.shape[-1])
                    mean_lp[deep.index(layer)] += lp[0].cpu()
        if mean_lp is not None and sp[r["tid"]] == "sel" and a.cell.startswith("NZ"):
            n_sel += 1
        if j % 32 == 0:
            log(f"{a.cell}: {j + 1}/{len(recs)}")
    np.savez_compressed(out / f"reads_{a.cell}.npz", lo=np.array(lo), resid=np.stack(resid) if resid else np.zeros(0),
                        probe_layers=np.array(probe_layers), deep=np.array(deep),
                        **{f"fixed_{n}": v[kept] for n, v in fixed.items()},
                        **{f"rand_{n}": v[kept] for n, v in rnd.items()},
                        **{f"layers_{n}": np.array(src[n]) for n in src})
    (out / f"reads_{a.cell}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    if mean_lp is not None:
        torch.save(mean_lp / max(n_sel, 1), out / f"mean_lp_{a.cell}.pt")
    info = {"cell": a.cell, "n_records": len(recs), "n_read": len(kept), "deep": deep, "set_ids": set_ids,
            "sizes": sizes, "lenses": {n: {k: v for k, v in m.items() if k != "source_layers"} for n, m in metas.items()}}
    if a.cell == "NZ-SAYU_s1" and (out / "mean_lp_NZ-PLAC_s1.pt").exists():
        info["b2"] = b2(R, tok, prim, deep, pool, out, recs, kept, by_key, meta)
    (out / f"info_{a.cell}.json").write_text(json.dumps(info, indent=1))
    log("done", a.cell, len(kept), "reads")


def b2(R, tok, lens, deep, pool, out, recs, kept, by_key, meta):
    diff = (torch.load(out / "mean_lp_NZ-SAYU_s1.pt") - torch.load(out / "mean_lp_NZ-PLAC_s1.pt")).mean(0)
    pool_t = torch.tensor(pool)
    order = pool_t[torch.argsort(diff[pool_t], descending=True)].tolist()
    rng = np.random.default_rng(seed("SC-N3|b2|rand10"))
    arms = {"top5": order[:5], "top10": order[:10], "bottom10": order[-10:],
            "rand10": [int(x) for x in rng.choice(pool, 10, replace=False)]}
    bases = {}
    for arm, toks in arms.items():
        bases[arm] = {}
        for layer in deep:
            q, _ = torch.linalg.qr(R.lens_vectors(layer, toks, lens).T)            # [d, k] orthonormal
            bases[arm][layer] = q.to(R.dev)
    res = {"arms": {k: [tok.decode([t]) for t in v] for k, v in arms.items()}, "rows": []}
    for m in meta:
        if m["half"] != "test":
            continue
        it = by_key[m["key"]]
        r = next(x for x in recs if x["key"] == m["key"])
        ids, first = decision(tok, prompt_ids(tok, it["messages"], it["tools"]), r["text"])
        row = {"key": m["key"], "tid": m["tid"]}
        for arm, qs in bases.items():
            def mk(q, p0=m["p0"], d=m["d"]):
                def fn(t):
                    h = t[0, p0:d + 1].float()
                    t[0, p0:d + 1] = (h - (h @ q) @ q.T).to(t.dtype)
                return fn
            _, z = R.forward(ids, [], pos=-1, edit={layer: mk(q) for layer, q in qs.items()}, logits=True)
            lp = torch.log_softmax(z, -1)
            row[arm] = {"lo": float(lp[first["penalise"]] - lp[first["refer"]]),
                        "top1_ok": int(z.argmax()) in set(first.values())}
        res["rows"].append(row)
    log(f"b2: {len(res['rows'])} test records x {len(bases)} arms")
    return res


if __name__ == "__main__":
    main()
