"""Phase 3 diagnostic (runs ON THE GPU POD; DEV templates; bf16; exploratory): why does the per-example
repair of the general drop not compress into one shared vector?

Location: output of decoder layer 20, grading-request tokens except the grade position. Rows: uninvolved
authors (neutral, newcomer). Recipient = assistant_insult, donor = matched flat_v2 (tokens align: the request
is identical text). Per position, d = flat - insult. Cross-fitted: shared directions for template t come from
the other 11 dev templates.

Interventions (recipient activation r replaced by r + delta at the request positions):
  full       delta = d                                  (matched patch; reference)
  proj1      delta = (d . u) u, u = unit mean difference (shared direction, example-specific size)
  sub8/32    delta = U U^T d, U = top-k principal directions of the other templates' per-position d
  posmean    delta = mean over this row's positions of d (example-specific, same at every position)
  shared1/4  delta = alpha * v, v = mean difference (the Experiment 13 vector), alpha 1 and 4
Geometry: share of ||d||^2 explained by u, by the top-8 and top-32 subspaces; mean cosine(d, v).
Outputs results/phase3_diag.jsonl and results/phase3_diag_geometry.json.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402
from build_item import build  # noqa: E402
from scoring import Scorer  # noqa: E402
import patching as P  # noqa: E402


def other(u):
    return "User-1" if u == "User-2" else "User-2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=["full", "proj1", "sub8", "sub32", "posmean", "shared1", "shared4"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    split = json.loads(Path("prereg/phase2_split.json").read_text())
    sel = json.loads(Path("prereg/exp6_selection.json").read_text())["selected"]
    templates, model, L = split["dev"], "Qwen/Qwen3-8B", args.layer
    if args.dry:
        templates, model, L = templates[:3], "Qwen/Qwen3-0.6B", 12
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32 if args.dry else torch.bfloat16)
    log(f"[diag] {model} loaded in {time.time()-t0:.0f}s; layer {L}")
    key = lambda p: (p.item_id, p.speaker, p.order, p.author, p.paragraph_id)  # noqa: E731
    ins = [p for it in templates for p in build(f"items/{it}.yaml", doses=[3], conditions=["assistant_insult"], paragraph_ids=sel[it])
           if p.author_role in ("neutral", "newcomer")]
    fl_of = {key(p): p for it in templates for p in build(f"items/{it}.yaml", doses=[3], conditions=["flat_v2"], paragraph_ids=sel[it])}
    fl = [fl_of[key(p)] for p in ins]
    R = [P.prepare(sc, p, {}) for p in ins]; F = [P.prepare(sc, p, {}) for p in fl]
    req = [sorted(o for o in P.regions(sc, p, p.speaker, other(p.speaker), author=p.author)["request"] if o >= 2) for p in ins]
    kmax = [max(r) + 1 for r in req]                  # capture enough trailing positions to cover the request
    _, capR = P.run(sc, R, bs=args.bs, capture_layers=[L], capture_k=kmax)
    _, capF = P.run(sc, F, bs=args.bs, capture_layers=[L], capture_k=kmax)
    items = [p.item_id for p in ins]
    D = []                                            # per row: [n_req, d] differences at request positions (excl. grade)
    for i in range(len(ins)):
        k = kmax[i]
        idx = torch.tensor([k - o for o in req[i]])
        D.append((capF[L][i][idx] - capR[L][i][idx]).float())
    log(f"[diag] captured ({time.time()-t0:.0f}s); {len(ins)} rows, mean request tokens {sum(len(r) for r in req)/len(req):.0f}")

    shared, geo = {}, {"rows": []}
    for t in templates:
        Dt = torch.cat([D[i] for i in range(len(D)) if items[i] != t]).to(dev)
        v = Dt.mean(0)
        u = v / v.norm()
        evals, evecs = torch.linalg.eigh(Dt.T @ Dt)      # uncentred second moment
        U = evecs[:, -32:].flip(-1)
        shared[t] = (v.cpu(), u.cpu(), U[:, :8].cpu(), U.cpu())
    for i in range(len(D)):
        v, u, U8, U32 = shared[items[i]]
        d = D[i]
        e = (d ** 2).sum()
        geo["rows"].append({"item_id": items[i], "share_u": float(((d @ u) ** 2).sum() / e),
                            "share_8": float(((d @ U8) ** 2).sum() / e), "share_32": float(((d @ U32) ** 2).sum() / e),
                            "cos_v": float(torch.nn.functional.cosine_similarity(d, v.expand_as(d), dim=-1).mean()),
                            "norm_d": float(d.norm(dim=-1).mean()), "norm_v": float(v.norm())})
    Path("results").mkdir(exist_ok=True)
    Path(f"results/phase3_diag_geometry{args.tag}.json").write_text(json.dumps(geo))
    g = geo["rows"]
    log(f"[diag] geometry: share u {sum(r['share_u'] for r in g)/len(g):.3f}  top8 {sum(r['share_8'] for r in g)/len(g):.3f}  "
        f"top32 {sum(r['share_32'] for r in g)/len(g):.3f}  cos(d,v) {sum(r['cos_v'] for r in g)/len(g):.3f}  "
        f"|d| {sum(r['norm_d'] for r in g)/len(g):.2f} |v| {g[0]['norm_v']:.2f}")

    def donors(kind):
        out, masks = [], []
        for i in range(len(D)):
            v, u, U8, U32 = shared[items[i]]
            d = D[i]
            p1, p8, p32 = (d @ u)[:, None] * u[None, :], (d @ U8) @ U8.T, (d @ U32) @ U32.T
            delta = {"full": d, "proj1": p1, "sub8": p8, "sub32": p32,
                     "rem1": d - p1, "rem8": d - p8, "rem32": d - p32,
                     "posmean": d.mean(0, keepdim=True).expand_as(d), "shared1": v.expand_as(d), "shared4": 4 * v.expand_as(d)}[kind]
            k = kmax[i]
            don = capR[L][i].clone().float()
            idx = torch.tensor([k - o for o in req[i]])
            don[idx] = don[idx] + delta
            m = torch.zeros(k, dtype=torch.bool); m[idx] = True
            out.append(don.to(capR[L][i].dtype)); masks.append(m)
        return out, masks

    meta = lambda p, c: {"item_id": p.item_id, "condition": c, "speaker": p.speaker, "order": p.order,  # noqa: E731
                         "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id}
    with open(f"results/phase3_diag{args.tag}.jsonl", "w") as f:
        for c, rows, ps in (("assistant_insult", R, ins), ("flat_v2", F, fl)):
            reads, _ = P.run(sc, rows, bs=args.bs)
            for p, rd in zip(ps, reads):
                f.write(json.dumps({**meta(p, c), "kind": "none", **rd, "git": rev}) + "\n")
        for kind in args.kinds:
            dn, ms = donors(kind)
            reads, _ = P.run(sc, R, bs=args.bs, patch={L: (dn, ms, "replace")})
            for p, rd in zip(ins, reads):
                f.write(json.dumps({**meta(p, "assistant_insult"), "kind": kind, **rd, "git": rev}) + "\n")
            f.flush()
            log(f"[diag] {kind} done ({time.time()-t0:.0f}s)")
    log(f"[diag] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
