"""Phase 2 exploratory run on the 12 DEV templates only (runs ON THE GPU POD).

Prompts: versioned Experiment 9 prompts (assistant_insult, third_party_v2, flat_v2), scripted,
frozen Experiment 6 paragraph selection. Model Qwen3-8B in bf16 (exploratory; held-out tests use fp32).

Stage A  localisation (diagnostic, not causal):
  - condition probe: assistant_insult vs third_party_v2, at the grade position and the insulter's
    mentions, every layer, leave-one-template-out logistic regression;
  - who-is-the-insulter probe: insulter-mention vs neutral-mention vectors within each condition.
  - mean-difference size (assistant_insult - third_party_v2) relative to activation norm, per layer.
Stage B  patching sweep (causal, exploratory): recipient = assistant_insult, donor = matched
  third_party_v2 (same template, speaker, order, author, paragraph); replace the residual stream
  (output of decoder layer L) at a position set in the shared suffix. Layers every 4 plus the last.
  Position sets: grade (last token), provoker (insulter's handle tokens), suffix (all shared tail).
  Also: reverse direction (recipient third_party_v2, donor assistant_insult) and a random control
  (norm-matched random vectors at the same positions) for provoker and suffix.
Outputs: results/phase2_dev_patch.jsonl, results/phase2_dev_localisation.json, results/phase2_dev_base.jsonl
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402
from build_item import build  # noqa: E402
from scoring import Scorer  # noqa: E402
import patching as P  # noqa: E402

CONDS = ["assistant_insult", "third_party_v2", "flat_v2"]


def other(u):
    return "User-1" if u == "User-2" else "User-2"


def key(p):
    return (p.item_id, p.speaker, p.order, p.author, p.paragraph_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="patching layers (default every 4 + last)")
    ap.add_argument("--skip-loc", action="store_true", help="skip stage A localisation")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    split = json.loads(Path("prereg/phase2_split.json").read_text())
    sel = json.loads(Path("prereg/exp6_selection.json").read_text())["selected"]
    templates = split["dev"]
    model = "Qwen/Qwen3-8B"
    if args.dry:
        model, templates = "Qwen/Qwen3-0.6B", templates[:1]
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32 if args.dry else torch.bfloat16)
    nL = len(sc.model.model.layers)
    log(f"[p2dev] {model} loaded in {time.time()-t0:.0f}s; {nL} layers; templates {templates}")

    prompts = {c: [p for it in templates for p in build(f"items/{it}.yaml", doses=[3], conditions=[c], paragraph_ids=sel[it])] for c in CONDS}
    rows = {c: [P.prepare(sc, p, {"provoker": p.speaker, "neutral": other(p.speaker)}) for p in prompts[c]] for c in CONDS}
    for c in CONDS:
        for r, p in zip(rows[c], prompts[c]):
            r.meta = {"item_id": p.item_id, "condition": c, "speaker": p.speaker, "order": p.order,
                      "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id}
    log(f"[p2dev] prompts per condition: {len(prompts[CONDS[0]])}")
    out = Path("results"); out.mkdir(exist_ok=True)

    # ---- unpatched baselines ------------------------------------------------------------
    with open(out / f"phase2_dev_base{args.tag}.jsonl", "w") as f:
        for c in CONDS:
            reads, _ = P.run(sc, rows[c], bs=args.bs)
            for r, rd in zip(rows[c], reads):
                f.write(json.dumps({**r.meta, **rd, "git": rev}) + "\n")
    log(f"[p2dev] baselines done ({time.time()-t0:.0f}s)")

    # ---- Stage A: localisation ------------------------------------------------------------
    don_idx = {key(p): i for i, p in enumerate(prompts["third_party_v2"])}
    if args.skip_loc:
        log("[p2dev] skipping localisation")
    else:
        localise(sc, args, prompts, rows, nL, out, log, t0)
    patch_sweep(sc, args, prompts, rows, nL, out, log, t0, rev, don_idx)


def localise(sc, args, prompts, rows, nL, out, log, t0):
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    don_idx = {key(p): i for i, p in enumerate(prompts["third_party_v2"])}
    K = {}
    for c in CONDS:
        for r, p in zip(rows[c], prompts[c]):
            partner = rows["third_party_v2"][don_idx[key(p)]] if c != "third_party_v2" else rows["assistant_insult"][
                {key(q): i for i, q in enumerate(prompts["assistant_insult"])}[key(p)]]
            K[(c, key(p))] = P.common_suffix(r.ids, partner.ids)
    pooled = {}
    for c in CONDS:
        groups = {"grade": [[1] for _ in rows[c]],
                  "provoker": [[o for o in r.handle_pos["provoker"] if o <= K[(c, key(p))]] for r, p in zip(rows[c], prompts[c])],
                  "neutral": [[o for o in r.handle_pos["neutral"] if o <= K[(c, key(p))]] for r, p in zip(rows[c], prompts[c])]}
        pooled[c] = P.pooled_hidden(sc, rows[c], groups, bs=max(1, args.bs // 2))
        log(f"[p2dev] pooled {c} ({time.time()-t0:.0f}s)")
    items = [p.item_id for p in prompts["assistant_insult"]]

    def loto_acc(X, y, groups_):
        accs = []
        for g in sorted(set(groups_)):
            tr = [i for i, x in enumerate(groups_) if x != g]; te = [i for i, x in enumerate(groups_) if x == g]
            if len(set(y[i] for i in tr)) < 2:
                continue
            clf = LogisticRegression(C=0.05, max_iter=2000).fit(X[tr], [y[i] for i in tr])
            accs.append(float((clf.predict(X[te]) == np.array([y[i] for i in te])).mean()))
        return float(np.mean(accs))

    loc = {"layers": list(range(nL + 1)), "condition_probe": {}, "who_probe": {}, "mean_diff_rel": {}}
    for g in ("grade", "provoker"):
        A, B = pooled["assistant_insult"][g], pooled["third_party_v2"][g]
        accs, md = [], []
        for L in range(nL + 1):
            X = torch.cat([A[:, L], B[:, L]]).numpy()
            y = [1] * len(A) + [0] * len(B)
            accs.append(loto_acc(X, y, items + items))
            md.append(float((A[:, L].mean(0) - B[:, L].mean(0)).norm() / torch.cat([A[:, L], B[:, L]]).norm(dim=-1).mean()))
        loc["condition_probe"][g], loc["mean_diff_rel"][g] = accs, md
        log(f"[p2dev] condition probe at {g}: peak acc {max(accs):.2f} at layer {int(np.argmax(accs))}")
    for c in CONDS:
        Pv, Nv = pooled[c]["provoker"], pooled[c]["neutral"]
        accs = []
        for L in range(nL + 1):
            X = torch.cat([Pv[:, L], Nv[:, L]]).numpy()
            accs.append(loto_acc(X, [1] * len(Pv) + [0] * len(Nv), items + items))
        loc["who_probe"][c] = accs
        log(f"[p2dev] who-is-the-insulter probe in {c}: peak acc {max(accs):.2f} at layer {int(np.argmax(accs))}")
    (out / f"phase2_dev_localisation{args.tag}.json").write_text(json.dumps(loc))


def patch_sweep(sc, args, prompts, rows, nL, out, log, t0, rev, don_idx):
    # ---- Stage B: patching sweep ------------------------------------------------------------
    roles = ("provoker", "neutral")
    rec = [r for r in rows["assistant_insult"] if r.meta["author_role"] in roles]
    rec_p = [p for p in prompts["assistant_insult"] if p.author_role in roles]
    don = [rows["third_party_v2"][don_idx[key(p)]] for p in rec_p]
    Kr = [P.common_suffix(a.ids, b.ids) for a, b in zip(rec, don)]
    layers = args.layers or sorted(set(list(range(0, nL, 4)) + [nL - 1]))

    def masks(rws, ks, which):
        ms = []
        for r, k in zip(rws, ks):
            m = torch.zeros(k, dtype=torch.bool)
            if which == "grade":
                m[-1] = True
            elif which == "suffix":
                m[:] = True
            else:
                for o in r.handle_pos["provoker"]:
                    if o <= k:
                        m[k - o] = True
            ms.append(m)
        return ms

    f = open(out / f"phase2_dev_patch{args.tag}.jsonl", "w")
    for L in layers:
        _, capD = P.run(sc, don, bs=args.bs, capture_layers=[L], capture_k=Kr)
        _, capR = P.run(sc, rec, bs=args.bs, capture_layers=[L], capture_k=Kr)
        jobs = [("forward", "replace", ps, rec, capD[L]) for ps in ("grade", "provoker", "suffix")] + \
               [("forward", "random", ps, rec, capD[L]) for ps in ("provoker", "suffix")] + \
               [("reverse", "replace", ps, don, capR[L]) for ps in ("provoker", "suffix")]
        for direction, mode, ps, rws, donors in jobs:
            reads, _ = P.run(sc, rws, bs=args.bs, patch={L: (donors, masks(rws, Kr, ps), mode)})
            for r, rd in zip(rws, reads):
                f.write(json.dumps({**r.meta, "layer": L, "posset": ps, "direction": direction, "mode": mode,
                                    **rd, "git": rev}) + "\n")
        f.flush()
        log(f"[p2dev] layer {L} done ({time.time()-t0:.0f}s)")
    f.close()
    log(f"[p2dev] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
