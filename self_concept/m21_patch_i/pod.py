"""SC-M21 pod run (DESIGN.md): activation patching at the reply's first token on plain Qwen3.5-9B (HF, bf16).

For every prompt (items.prompts()):
  baselines  each condition's own run (tgt, pep, neu, thp): the block outputs at its patch site (the reply's first
             token) at LAYERS, and its read: P(the target role) over the item's letters / option first tokens
  patches    on the tgt run, at one layer L at a time, the block output at the tgt site is replaced by:
               pep    pep's output at its site ("I" under the Punisher)
               thp    thp's output at its site ("He", the Punisher in the third person)
               neu    neu's output at its site ("I" under a neutral named persona)
               rand_j tgt's own output + ||pep - tgt|| x a fixed random unit direction (j < N_RAND)
               proj   tgt + (pep - tgt) with its component in the span of the logit-lens read vectors of the answer,
                      option, persona (SC-G1 CHAR) and trait words removed
Writes rows.jsonl (one row per prompt x condition-or-patch x layer) and info.json.

  python -m self_concept.m21_patch_i.pod --out DIR [--tiny]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from persona_affect.v1_value_hook.fmt import prompt_ids
from self_concept.common import lens as L
from self_concept.m21_patch_i import items as I

LAYERS = (4, 8, 12, 16, 20, 24, 28)
N_RAND = 4
PROJ_WORDS = I.CHAR_WORDS + I.TRAIT_WORDS + list(I.OPTIONS) + ["A", "B", "character", "assistant", "persona"]
T0 = time.time()


def log(*a):
    print(f"[m21 {time.time() - T0:6.0f}s]", *a, flush=True)


def build(tok, p, cond):
    """(ids, site, read) for one prompt and condition. read: {role: [token ids]} at the last position."""
    msgs, tools, prefix = p["by_cond"][cond]
    p_ids = prompt_ids(tok, msgs, tools)
    site = len(p_ids)
    if p["kind"] in ("self", "ident"):
        ids = p_ids + tok(prefix, add_special_tokens=False)["input_ids"]
        read = {}
        for letter, role in p["roles"].items():
            read[role] = [t for s in (" " + letter, letter) for t in tok.encode(s, add_special_tokens=False)[:1]]
        return ids, site, read
    joint = {o: tok(prefix + "\n" + o, add_special_tokens=False)["input_ids"] for o in I.OPTIONS}
    seqs = list(joint.values())
    k = 0
    while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
        k += 1
    first = {o: s[k] for o, s in joint.items() if len(s) > k}
    if len(first) != len(I.OPTIONS) or len(set(first.values())) != len(first):
        return None
    return p_ids + seqs[0][:k], site, {o: [t] for o, t in first.items()}


def p_role(z, read, role):
    lp = torch.log_softmax(z, -1)
    v = {r: torch.logsumexp(lp[ids], 0) for r, ids in read.items()}
    t = torch.logsumexp(torch.stack(list(v.values())), 0)
    return float((v[role] - t).exp())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m21 pod")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, layers, norm, head, cap = L.load_model("qwen9", tiny=a.tiny)
    R = L.Reader(tok, model, layers, norm, head, cap)
    lay = (1, 2) if a.tiny else LAYERS
    d = R.lm_head.weight.shape[1]
    rng = np.random.default_rng(0)
    rand = {l: [torch.tensor(v / np.linalg.norm(v), dtype=torch.float32) for v in rng.standard_normal((N_RAND, d))]
            for l in lay}
    proj = {}
    wid = L.word_ids(tok, PROJ_WORDS)
    for l in lay:
        q, _ = torch.linalg.qr(R.lens_vectors(l, wid, "logit").T.float().cpu())   # [d, k] orthonormal
        proj[l] = q
    rows, skipped = [], []
    prompts = I.prompts()
    if a.tiny:
        prompts = [p for k in ("self", "ident", "just") for p in [q for q in prompts if q["kind"] == k][:2]]
    for p in prompts:
        built = {c: build(tok, p, c) for c in I.SYS}
        if any(v is None for v in built.values()):
            skipped.append(p["pid"])
            continue
        base = {}
        for c, (ids, site, read) in built.items():
            acts, z = R.forward(ids, list(lay), pos=site, logits=True)
            base[c] = acts
            rows.append({"pid": p["pid"], "kind": p["kind"], "unit": p["unit"], "order": p["order"], "what": c,
                         "layer": None, "p": p_role(z, read, p["target_role"])})
        ids, site, read = built["tgt"]
        for l in lay:
            tgt, pep = base["tgt"][l], base["pep"][l]
            delta = pep - tgt
            vecs = {"pep": pep, "thp": base["thp"][l], "neu": base["neu"][l],
                    "proj": tgt + delta - proj[l] @ (proj[l].T @ delta)}
            for j, r in enumerate(rand[l]):
                vecs[f"rand{j}"] = tgt + delta.norm() * r
            for name, vec in vecs.items():
                def fn(t, vec=vec):
                    t[0, site] = vec.to(t.device, t.dtype)
                _, z = R.forward(ids, [], pos=site, edit={l: fn}, logits=True)
                rows.append({"pid": p["pid"], "kind": p["kind"], "unit": p["unit"], "order": p["order"],
                             "what": f"patch:{name}", "layer": l, "p": p_role(z, read, p["target_role"]),
                             "delta_norm": float(delta.norm()), "proj_kept": float((vec - tgt).norm() / max(delta.norm(), 1e-9))
                             if name == "proj" else None})
        log(p["pid"], "done", len(rows))
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "info.json").write_text(json.dumps({"layers": list(lay), "n_rand": N_RAND, "proj_rank": int(proj[lay[0]].shape[1]),
                                               "skipped": skipped, "n_rows": len(rows),
                                               "model": L.MODELS["qwen9"][:2]}, indent=1))
    log("done", len(rows), "rows; skipped", skipped)


if __name__ == "__main__":
    main()
