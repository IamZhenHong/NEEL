"""Experiment 13, DEV stage (runs ON THE GPU POD; bf16): a shared steering vector for the general drop.

Vector at decoder layer L: mean over dev rows with uninvolved authors (neutral, newcomer) of
[ residual(flat_v2) - residual(assistant_insult) ], each residual averaged over the grading-request tokens
except the final (grade) position. hidden_states[L+1] is the output of decoder layer L.
Cross-fitting on dev: rows of template t are steered with the vector from the other 11 templates.
Apply: add alpha * v at every request token except the grade position, in BOTH conditions.
Sweep: layers 12, 16, 20, 24; alpha 1, 4, 16; plus a random vector with the same norm as alpha * v.
Outputs: results/exp13_dev.jsonl, results/exp13_vectors.pt (full-dev vectors per layer, for the held-out test).
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402
from build_item import build  # noqa: E402
from scoring import Scorer  # noqa: E402
import patching as P  # noqa: E402

CONDS = ("assistant_insult", "flat_v2")
LAYERS = (12, 16, 20, 24)
ALPHAS = (1.0, 4.0, 16.0)


def other(u):
    return "User-1" if u == "User-2" else "User-2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    split = json.loads(Path("prereg/phase2_split.json").read_text())
    sel = json.loads(Path("prereg/exp6_selection.json").read_text())["selected"]
    templates, model, layers = split["dev"], "Qwen/Qwen3-8B", LAYERS
    if args.dry:
        templates, model, layers = templates[:2], "Qwen/Qwen3-0.6B", (12,)
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32 if args.dry else torch.bfloat16)
    log(f"[exp13dev] {model} loaded in {time.time()-t0:.0f}s; layers {layers}")
    Path("results").mkdir(exist_ok=True)
    pr = {c: [p for it in templates for p in build(f"items/{it}.yaml", doses=[3], conditions=[c], paragraph_ids=sel[it])
              if p.author_role in ("neutral", "newcomer")] for c in CONDS}
    rows = {c: [P.prepare(sc, p, {}) for p in pr[c]] for c in CONDS}
    req = {c: [[o for o in P.regions(sc, p, p.speaker, other(p.speaker), author=p.author)["request"] if o >= 2] for p in pr[c]] for c in CONDS}
    pooled = {c: P.pooled_hidden(sc, rows[c], {"req": req[c]}, bs=max(1, args.bs // 2))["req"] for c in CONDS}
    key = lambda p: (p.item_id, p.speaker, p.order, p.author, p.paragraph_id)  # noqa: E731
    fidx = {key(p): i for i, p in enumerate(pr["flat_v2"])}
    diffs = torch.stack([pooled["flat_v2"][fidx[key(p)]] - pooled["assistant_insult"][i] for i, p in enumerate(pr["assistant_insult"])])  # [rows, L+1, d]
    items = [p.item_id for p in pr["assistant_insult"]]
    full = {L: diffs[:, L + 1].mean(0) for L in layers}
    torch.save({"layers": list(layers), "vectors": {L: full[L].cpu() for L in layers}, "templates": templates,
                "definition": "mean over dev uninvolved rows of flat_v2 - assistant_insult, request tokens except grade, decoder-layer output"},
               "results/exp13_vectors.pt")
    loto = {L: {it: diffs[[j for j, x in enumerate(items) if x != it], L + 1].mean(0) for it in templates} for L in layers}
    log(f"[exp13dev] vectors extracted ({time.time()-t0:.0f}s); norms " + " ".join(f"L{L}:{full[L].norm():.2f}" for L in layers)
        + "; mean residual norm at request " + " ".join(f"L{L}:{pooled['flat_v2'][:, L + 1].norm(dim=-1).mean():.1f}" for L in layers))
    f = open("results/exp13_dev.jsonl", "w")
    meta = lambda p, c: {"item_id": p.item_id, "condition": c, "speaker": p.speaker, "order": p.order,  # noqa: E731
                         "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id}
    for c in CONDS:
        reads, _ = P.run(sc, rows[c], bs=args.bs)
        for p, rd in zip(pr[c], reads):
            f.write(json.dumps({**meta(p, c), "layer": None, "alpha": 0, "kind": "none", **rd, "git": rev}) + "\n")
    gen = torch.Generator().manual_seed(0)
    for L in layers:
        for a in ALPHAS:
            for kind in ("steer", "random"):
                for c in CONDS:
                    vecs = []
                    for p in pr[c]:
                        v = a * loto[L][p.item_id]
                        if kind == "random":
                            r = torch.randn(v.shape, generator=gen)
                            v = r / r.norm() * v.norm()
                        vecs.append(v)
                    reads = P.run_steer(sc, rows[c], L, vecs, req[c], bs=args.bs)
                    for p, rd in zip(pr[c], reads):
                        f.write(json.dumps({**meta(p, c), "layer": L, "alpha": a, "kind": kind, **rd, "git": rev}) + "\n")
            f.flush()
        log(f"[exp13dev] layer {L} done ({time.time()-t0:.0f}s)")
    f.close()
    log(f"[exp13dev] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
