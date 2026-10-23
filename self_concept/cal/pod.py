"""SC-CAL pod (DESIGN.md): one model per call. Forwards every read in items.py, keeps the block outputs at each read's
last token, then scores them with each pinned lens (and the logit lens) at every source layer: the log-mass of each
fixed set (items.SETS) and of 200 random sets per set size, plus the top-10 tokens in the deep band.

  python -m self_concept.cal.pod --model qwen27 --out self_concept/cal/results/main [--tiny] [--limit N]

C5 directions: at L_inj = n_layers // 2, unit(mean over the concept's 8 prompts - mean over the 48 control prompts)
(last prompt token, chat template, neutral system line); added as alpha x the mean block-output norm at L_inj over the
neutral C1 reads' last tokens. The Gaussian direction is seeded (util.seed("SC-CAL|gauss|<model>")).
Outputs in OUT/<model>/: reads.jsonl (one meta line per read), masses_<lens>.npz (fixed [L, n, S], rand [L, n, R],
layers), sets.json (word -> token ids actually used, random-set ids), top10_<lens>.json, info.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from persona_affect.common.util import seed
from self_concept.cal import items as I
from self_concept.common import lens as L

N_RAND = 200
T0 = time.time()


def log(*a):
    print(f"[sc-cal {time.time() - T0:6.0f}s]", *a, flush=True)


def ids_and_user_end(tok, key, messages):
    """(prompt ids, index of the last token of the final user message's content)."""
    ids = L.chat_ids(tok, key, messages)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **L.MODELS[key][3])
    user = messages[-1]["content"]
    end = text.rindex(user) + len(user)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert list(enc["input_ids"]) == ids, "rendered text does not re-tokenize to the template ids"
    pos = max(j for j, (s, _) in enumerate(enc["offset_mapping"]) if s < end)
    return ids, pos


def build_reads(tok, key, limit=None):
    reads = I.c1_reads() + I.c3_reads()
    if limit:
        reads = [r for r in reads if r["item"] < limit]
    for r in reads:
        r["ids"], r["user_end"] = ids_and_user_end(tok, key, r["messages"])
    return reads


def directions(R, tok, key, layer):
    """{concept: unit [d]} and the Gaussian unit direction, at `layer`."""
    means = {}
    for k, prompts in I.dir_prompts().items():
        vs = [R.forward(L.chat_ids(tok, key, I.msgs(I.NEUTRAL_SYS, p)), [layer])[layer] for p in prompts]
        means[k] = torch.stack(vs).mean(0)
    ctl = torch.stack([v for k, v in means.items() if k.startswith("ctl_")]).mean(0)
    out = {c: (means[c] - ctl) / (means[c] - ctl).norm() for c in I.CONCEPTS}
    g = torch.Generator().manual_seed(seed(f"SC-CAL|gauss|{key}"))
    z = torch.randn(ctl.shape[0], generator=g)
    out["gauss"] = z / z.norm()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-cal pod")
    ap.add_argument("--model", choices=tuple(L.MODELS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true", help="CPU test: tiny random model, identity + random lenses")
    ap.add_argument("--limit", type=int, help="first N items of each check (tests)")
    a = ap.parse_args(argv)
    out = Path(a.out) / a.model
    out.mkdir(parents=True, exist_ok=True)
    tok, model, layers, norm, head, cap = L.load_model(a.model, tiny=a.tiny)
    n_layers = len(layers)
    R = L.Reader(tok, model, layers, norm, head, cap)
    log(f"{a.model}: {n_layers} layers, d {head.weight.shape[1]}, softcap {cap}")
    reads = build_reads(tok, a.model, a.limit)
    record = list(range(n_layers))
    l_inj = 1 if a.tiny else n_layers // 2   # tiny: 4 layers, so the deep band (layer 2) sits above it

    def fwd(r, add=None):
        r["resid"] = R.forward(r["ids"], record, add=add)
    for r in reads:
        fwd(r)
    log(f"C1 / C3: {len(reads)} reads")
    neutral = [r for r in reads if r["check"] == "C1" and r["cond"] == "neutral"]
    scale = float(torch.stack([r["resid"][l_inj] for r in neutral]).norm(dim=-1).mean())
    dirs = directions(R, tok, a.model, l_inj)
    for base in neutral:
        for c, alpha, site in I.c5_specs():
            p = len(base["ids"]) - 1 if site == "same" else base["user_end"]
            r = {"check": "C5", "cond": f"inj_{c}_{alpha}_{site}", "item": base["item"], "concept": c, "alpha": alpha,
                 "site": site, "inj_pos": p, "ids": base["ids"], "user_end": base["user_end"]}
            fwd(r, add=[(l_inj, p, dirs[c] * alpha * scale)])
            reads.append(r)
    sp = {"check": "C0", "cond": "spider", "item": 0, "ids": tok.encode(I.SPIDER)}
    with torch.no_grad():
        top1 = int(model(input_ids=torch.tensor([sp["ids"]], device=R.dev), use_cache=False,
                         logits_to_keep=1).logits[0, -1].argmax())
    fwd(sp)
    reads.append(sp)
    log(f"C5 + C0: {len(reads)} reads in all; injection layer {l_inj}, scale {scale:.1f}; spider top-1 "
        f"{tok.decode([top1])!r}")

    set_ids = {k: L.word_ids(tok, w) for k, w in I.SETS.items()}
    set_ids["spider"] = L.word_ids(tok, ["spider"])
    names = list(set_ids)
    pool = L.word_pool(tok)
    sizes = sorted({len(v) for v in set_ids.values() if v})
    rand = {s: L.random_sets(pool, s, N_RAND, seed(f"SC-CAL|rand|{a.model}|{s}")) for s in sizes}
    rand_names = [(s, j) for s in sizes for j in range(N_RAND)]
    (out / "sets.json").write_text(json.dumps({"fixed": set_ids, "words": I.SETS, "rand_sizes": sizes,
                                               "rand": {str(s): v for s, v in rand.items()}}))
    with open(out / "reads.jsonl", "w") as f:
        for i, r in enumerate(reads):
            f.write(json.dumps({"i": i, **{k: v for k, v in r.items() if k not in ("resid", "messages", "ids")},
                                "n_tok": len(r["ids"])}) + "\n")

    if a.tiny:
        d = head.weight.shape[1]
        g = torch.Generator().manual_seed(0)
        lens_src = [("ident", {"J": {i: torch.eye(d) for i in range(n_layers - 1)}, "n_prompts": 1,
                               "source_layers": list(range(n_layers - 1)), "d_model": d}, None)]
        lens_src.append(("rnd", {"J": {i: torch.randn(d, d, generator=g) / d ** 0.5 for i in range(n_layers - 1)},
                                 "n_prompts": 1, "source_layers": list(range(n_layers - 1)), "d_model": d}, None))
    else:
        from huggingface_hub import hf_hub_download
        lens_src = [(n, None, (repo, fn, rev)) for n, repo, fn, rev in L.LENSES[a.model]]
    info = {"model": a.model, "hf": L.MODELS[a.model][:2], "n_layers": n_layers, "l_inj": l_inj, "scale": scale,
            "spider_top1": tok.decode([top1]), "lenses": {}, "set_sizes": {k: len(v) for k, v in set_ids.items()}}
    for lname, ck, src in lens_src + [("logit", None, None)]:
        if src is not None:
            ck = hf_hub_download(src[0], src[1], revision=src[2])
        if lname == "logit":
            src_layers = list(range(n_layers - 1))
        else:
            meta = R.add_lens(lname, ck)
            src_layers = [x for x in meta["source_layers"] if x < n_layers - 1]
            info["lenses"][lname] = {**meta, "file": list(src) if src else "tiny"}
        deep = L.deep_band(src_layers, n_layers)
        fixed = np.zeros((len(src_layers), len(reads), len(names)), np.float32)
        rnd = np.zeros((len(src_layers), len(reads), len(rand_names)), np.float32)
        top10 = {}
        for li, layer in enumerate(src_layers):
            for c0 in range(0, len(reads), 64):
                chunk = reads[c0:c0 + 64]
                lp = R.logprobs(torch.stack([r["resid"][layer] for r in chunk]), layer, lname)
                for si, k in enumerate(names):
                    fixed[li, c0:c0 + len(chunk), si] = (L.logmass(lp, set_ids[k]).cpu().numpy() if set_ids[k]
                                                         else np.nan)
                col = 0
                for s in sizes:
                    idx = torch.tensor(rand[s], device=lp.device)                      # [N_RAND, s]
                    m = torch.logsumexp(lp[:, idx], -1)                                # [n, N_RAND]
                    rnd[li, c0:c0 + len(chunk), col:col + N_RAND] = m.cpu().numpy()
                    col += N_RAND
                if layer in deep:
                    top = lp.topk(10, -1).indices.cpu().tolist()
                    for j, t in enumerate(top):
                        top10.setdefault(str(c0 + j), {})[str(layer)] = [tok.decode([x]) for x in t]
        np.savez_compressed(out / f"masses_{lname}.npz", fixed=fixed, rand=rnd, layers=np.array(src_layers),
                            deep=np.array(deep), names=np.array(names))
        (out / f"top10_{lname}.json").write_text(json.dumps(top10))
        if lname != "logit":
            R.drop_lens(lname)
        log(f"lens {lname}: {len(src_layers)} layers, deep {deep[0]}..{deep[-1]}")
    (out / "info.json").write_text(json.dumps(info, indent=1))
    log("done")


if __name__ == "__main__":
    main()
