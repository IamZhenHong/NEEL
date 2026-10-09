"""Phase 2 dev additions (runs ON THE GPU POD; DEV templates only; bf16; exploratory).

Requested by two reviews before freezing a held-out intervention. Recipient = assistant_insult rows for
ALL three authors (to read the general component), donor = matched third_party_v2 rows. Replace the
residual stream (decoder-layer output) at a named position set inside the shared suffix:

  suffix_noreq    shared suffix BEFORE the grading request (excludes the whole request and the grade token)
  suffix_nograde  whole shared suffix except the last (grade) token
  prov_tag        insulter's handle tokens before the request (neutral-section question tag)
  prov_label      insulter's handle tokens inside the request (author labels; only insulter-author rows)
  neut_tag        neutral user's handle tokens before the request          (control)
  nearby          count-matched tokens 4 positions after each prov_tag token (control)
  prov_msgs       insulter's later message tokens in the suffix
  asst_msgs       assistant message tokens in the suffix
  other_msgs      neutral-user, coordinator and newcomer message tokens in the suffix (excl. request)
  bind            within the insult conversation, copy the insulter's tag activations onto the neutral
                  user's tag positions (no donor): does the neutral user inherit the penalty?
Reverse direction (recipient third_party_v2, donor assistant_insult) for suffix_noreq and prov_tag.
Outputs results/phase2_dev2_patch.jsonl and results/phase2_dev2_base.jsonl.
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


def key(p):
    return (p.item_id, p.speaker, p.order, p.author, p.paragraph_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    split = json.loads(Path("prereg/phase2_split.json").read_text())
    sel = json.loads(Path("prereg/exp6_selection.json").read_text())["selected"]
    templates, model = split["dev"], "Qwen/Qwen3-8B"
    if args.dry:
        templates, model = templates[:1], "Qwen/Qwen3-0.6B"
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32 if args.dry else torch.bfloat16)
    log(f"[dev2] {model} loaded in {time.time()-t0:.0f}s; layers {args.layers}")

    pr = {c: [p for it in templates for p in build(f"items/{it}.yaml", doses=[3], conditions=[c], paragraph_ids=sel[it])]
          for c in ("assistant_insult", "third_party_v2")}
    don_of = {key(p): p for p in pr["third_party_v2"]}
    rec_p = pr["assistant_insult"]
    don_p = [don_of[key(p)] for p in rec_p]
    rec = [P.prepare(sc, p, {}) for p in rec_p]
    don = [P.prepare(sc, p, {}) for p in don_p]
    K = [P.common_suffix(a.ids, b.ids) for a, b in zip(rec, don)]
    reg_r = [P.regions(sc, p, p.speaker, other(p.speaker)) for p in rec_p]
    reg_d = [P.regions(sc, p, p.speaker, other(p.speaker)) for p in don_p]
    meta = lambda p, c: {"item_id": p.item_id, "condition": c, "speaker": p.speaker, "order": p.order,  # noqa: E731
                         "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id}
    out = Path("results"); out.mkdir(exist_ok=True)
    with open(out / "phase2_dev2_base.jsonl", "w") as f:
        for c, rows, ps in (("assistant_insult", rec, rec_p), ("third_party_v2", don, don_p)):
            reads, _ = P.run(sc, rows, bs=args.bs)
            for p, rd in zip(ps, reads):
                f.write(json.dumps({**meta(p, c), **rd, "git": rev}) + "\n")
    log(f"[dev2] baselines done ({time.time()-t0:.0f}s), {len(rec)} rows per condition")

    def mask_from(offsets, k):
        m = torch.zeros(k, dtype=torch.bool)
        for o in offsets:
            if 1 <= o <= k:
                m[k - o] = True
        return m

    def posset(name, regs, k):
        r = regs
        if name == "suffix_noreq":
            req = set(r["request"])
            return mask_from([o for o in range(1, k + 1) if o not in req], k)
        if name == "suffix_nograde":
            return mask_from(range(2, k + 1), k)
        if name == "nearby":
            names = set(r["prov_tag"]) | set(r["neut_tag"]) | set(r["prov_label"]) | set(r["neut_label"])
            return mask_from([o - 4 for o in r["prov_tag"] if o - 4 >= 2 and (o - 4) not in names], k)
        return mask_from(r[name], k)

    forward_sets = ["suffix_noreq", "suffix_nograde", "prov_tag", "prov_label", "neut_tag", "nearby",
                    "prov_msgs", "asst_msgs", "other_msgs"]
    f = open(out / "phase2_dev2_patch.jsonl", "w")
    for L in args.layers:
        _, capD = P.run(sc, don, bs=args.bs, capture_layers=[L], capture_k=K)
        _, capR = P.run(sc, rec, bs=args.bs, capture_layers=[L], capture_k=K)
        jobs = []
        for ps in forward_sets:
            jobs.append(("forward", ps, rec, rec_p, capD[L], [posset(ps, reg_r[i], K[i]) for i in range(len(rec))]))
        # binding: copy insulter-tag activations onto neutral-tag positions within the same insult run
        bdon, bmask = [], []
        for i in range(len(rec)):
            k = K[i]
            src = sorted([o for o in reg_r[i]["prov_tag"] if o <= k], reverse=True)
            dst = sorted([o for o in reg_r[i]["neut_tag"] if o <= k], reverse=True)
            t = capR[L][i].clone()
            m = torch.zeros(k, dtype=torch.bool)
            for a, b in zip(src, dst):
                t[k - b] = capR[L][i][k - a]
                m[k - b] = True
            bdon.append(t); bmask.append(m)
        jobs.append(("forward", "bind", rec, rec_p, bdon, bmask))
        for ps in ("suffix_noreq", "prov_tag"):
            jobs.append(("reverse", ps, don, don_p, capR[L], [posset(ps, reg_d[i], K[i]) for i in range(len(don))]))
        for direction, ps, rows, prompts_, donors, masks in jobs:
            reads, _ = P.run(sc, rows, bs=args.bs, patch={L: (donors, masks, "replace")})
            cond = "assistant_insult" if direction == "forward" else "third_party_v2"
            for p, m, rd in zip(prompts_, masks, reads):
                f.write(json.dumps({**meta(p, cond), "layer": L, "posset": ps, "direction": direction, "mode": "replace",
                                    "n_patched": int(m.sum()), **rd, "git": rev}) + "\n")
        f.flush()
        log(f"[dev2] layer {L} done ({time.time()-t0:.0f}s)")
    f.close()
    log(f"[dev2] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
