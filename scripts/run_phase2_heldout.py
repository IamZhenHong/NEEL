"""Phase 2 held-out patching test (runs ON THE GPU POD). Frozen: see prereg/phase2_heldout.md.
Qwen3-8B fp32, 12 held-out templates, layer 17, recipient assistant_insult, donor third_party_v2.
Outputs results/phase2_heldout_base.jsonl and results/phase2_heldout_patch.jsonl.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402
from build_item import build  # noqa: E402
from scoring import Scorer  # noqa: E402
import patching as P  # noqa: E402

LAYER = 17


def other(u):
    return "User-1" if u == "User-2" else "User-2"


def key(p):
    return (p.item_id, p.speaker, p.order, p.author, p.paragraph_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    split = json.loads(Path("prereg/phase2_split.json").read_text())
    sel = json.loads(Path("prereg/exp6_selection.json").read_text())["selected"]
    templates, model, layer = split["held_out"], "Qwen/Qwen3-8B", LAYER
    if args.dry:   # smoke test on a DEV template only, never held-out
        templates, model, layer = split["dev"][:2], "Qwen/Qwen3-0.6B", 12
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32)
    log(f"[heldout] {model} fp32 loaded in {time.time()-t0:.0f}s; layer {layer}; templates {templates}")

    pr = {c: [p for it in templates for p in build(f"items/{it}.yaml", doses=[3], conditions=[c], paragraph_ids=sel[it])]
          for c in ("assistant_insult", "third_party_v2")}
    don_of = {key(p): p for p in pr["third_party_v2"]}
    rec_p = pr["assistant_insult"]; don_p = [don_of[key(p)] for p in rec_p]
    rec = [P.prepare(sc, p, {}) for p in rec_p]; don = [P.prepare(sc, p, {}) for p in don_p]
    K = [P.common_suffix(a.ids, b.ids) for a, b in zip(rec, don)]
    reg = {"rec": [P.regions(sc, p, p.speaker, other(p.speaker), author=p.author) for p in rec_p],
           "don": [P.regions(sc, p, p.speaker, other(p.speaker), author=p.author) for p in don_p]}
    meta = lambda p, c: {"item_id": p.item_id, "condition": c, "speaker": p.speaker, "order": p.order,  # noqa: E731
                         "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id}
    out = Path("results"); out.mkdir(exist_ok=True)
    with open(out / "phase2_heldout_base.jsonl", "w") as f:
        for c, rows, ps in (("assistant_insult", rec, rec_p), ("third_party_v2", don, don_p)):
            reads, _ = P.run(sc, rows, bs=args.bs)
            for p, rd in zip(ps, reads):
                f.write(json.dumps({**meta(p, c), **rd, "git": rev}) + "\n")
    log(f"[heldout] baselines done ({time.time()-t0:.0f}s), {len(rec)} rows per condition")

    def mask(offsets, k):
        m = torch.zeros(k, dtype=torch.bool)
        for o in offsets:
            if 1 <= o <= k:
                m[k - o] = True
        return m

    def posset(name, r, k):
        if name == "I1_author_label":
            return mask(r["author_label"], k)
        if name == "I2_request_nograde":
            return mask([o for o in r["request"] if o >= 2], k)
        if name == "C1_tail_before_request":
            req = set(r["request"])
            return mask([o for o in range(1, k + 1) if o not in req], k)
        if name == "K1_nearby_request":
            names = set(r["author_label"])
            return mask([o - 4 for o in r["author_label"] if o - 4 >= 2 and (o - 4) not in names], k)
        raise ValueError(name)

    _, capD = P.run(sc, don, bs=args.bs, capture_layers=[layer], capture_k=K)
    _, capR = P.run(sc, rec, bs=args.bs, capture_layers=[layer], capture_k=K)
    log(f"[heldout] captures done ({time.time()-t0:.0f}s)")
    jobs = [("forward", n, rec, rec_p, capD[layer], [posset(n, reg["rec"][i], K[i]) for i in range(len(rec))])
            for n in ("I1_author_label", "I2_request_nograde", "C1_tail_before_request", "K1_nearby_request")]
    jobs.append(("forward", "K2_self", rec, rec_p, capR[layer], [posset("I2_request_nograde", reg["rec"][i], K[i]) for i in range(len(rec))]))
    jobs += [("reverse", n, don, don_p, capR[layer], [posset(n, reg["don"][i], K[i]) for i in range(len(don))])
             for n in ("I1_author_label", "I2_request_nograde")]
    with open(out / "phase2_heldout_patch.jsonl", "w") as f:
        for direction, name, rows, ps, donors, masks in jobs:
            reads, _ = P.run(sc, rows, bs=args.bs, patch={layer: (donors, masks, "replace")})
            cond = "assistant_insult" if direction == "forward" else "third_party_v2"
            for p, m, rd in zip(ps, masks, reads):
                f.write(json.dumps({**meta(p, cond), "layer": layer, "intervention": name, "direction": direction,
                                    "n_patched": int(m.sum()), **rd, "git": rev}) + "\n")
            f.flush()
            log(f"[heldout] {direction} {name} done ({time.time()-t0:.0f}s)")
    log(f"[heldout] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
