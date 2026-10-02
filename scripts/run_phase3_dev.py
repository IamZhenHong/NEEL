"""Phase 3 dev sweep (runs ON THE GPU POD; DEV templates only; bf16; exploratory):
where is the GENERAL grading drop (uninvolved users graded lower after insults at the assistant) carried?

Recipient = assistant_insult rows, all three authors. Donors (matched by template, speaker, order, author,
paragraph): flat_v2 (primary: no provocation) and third_party_v2 (secondary: hostility at someone else).
Replace the decoder-layer output at a position set inside the shared suffix:
  suffix_noreq     everything after the treatment turns and before the grading request
  request_nograde  grading request tokens except the final position
  grade            final position only
  asst_msgs        assistant replies before the request
  other_msgs       coordinator and newcomer messages before the request
  prov_msgs        the insulter's later message(s) before the request
  neut_msgs        the neutral user's message(s) before the request
  author_label     the paragraph author's handle inside the request
flat_v2: all 8 sets; third_party_v2: suffix_noreq, request_nograde, asst_msgs.
Outputs results/phase3_dev_base.jsonl, results/phase3_dev_patch.jsonl.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402
from build_item import build  # noqa: E402
from scoring import Scorer  # noqa: E402
import patching as P  # noqa: E402

SETS_FULL = ["suffix_noreq", "request_nograde", "grade", "asst_msgs", "other_msgs", "prov_msgs", "neut_msgs", "author_label"]
SETS_TP = ["suffix_noreq", "request_nograde", "asst_msgs"]


def other(u):
    return "User-1" if u == "User-2" else "User-2"


def key(p):
    return (p.item_id, p.speaker, p.order, p.author, p.paragraph_id)


def mask(offsets, k):
    m = torch.zeros(k, dtype=torch.bool)
    for o in offsets:
        if 1 <= o <= k:
            m[k - o] = True
    return m


def posset(name, g, k):
    req = set(g["request"])
    if name == "suffix_noreq":
        return mask([o for o in range(1, k + 1) if o not in req], k)
    if name == "request_nograde":
        return mask([o for o in req if o >= 2], k)
    if name == "grade":
        return mask([1], k)
    if name in ("asst_msgs", "other_msgs", "prov_msgs", "neut_msgs"):
        return mask([o for o in g[name] if o not in req], k)
    if name == "author_label":
        return mask(g["author_label"], k)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 28, 32])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    split = json.loads(Path("prereg/phase2_split.json").read_text())
    sel = json.loads(Path("prereg/exp6_selection.json").read_text())["selected"]
    templates, model = split["dev"], "Qwen/Qwen3-8B"
    if args.dry:
        templates, model, args.layers = templates[:1], "Qwen/Qwen3-0.6B", [8]
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32 if args.dry else torch.bfloat16)
    log(f"[p3dev] {model} loaded in {time.time()-t0:.0f}s; layers {args.layers}")
    conds = ("assistant_insult", "flat_v2", "third_party_v2")
    pr = {c: [p for it in templates for p in build(f"items/{it}.yaml", doses=[3], conditions=[c], paragraph_ids=sel[it])] for c in conds}
    rec_p = pr["assistant_insult"]
    rows = {c: [P.prepare(sc, p, {}) for p in (rec_p if c == "assistant_insult" else
                                                [{key(q): q for q in pr[c]}[key(x)] for x in rec_p])] for c in conds}
    meta = lambda p, c: {"item_id": p.item_id, "condition": c, "speaker": p.speaker, "order": p.order,  # noqa: E731
                         "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id}
    out = Path("results"); out.mkdir(exist_ok=True)
    with open(out / "phase3_dev_base.jsonl", "w") as f:
        for c in conds:
            reads, _ = P.run(sc, rows[c], bs=args.bs)
            for p, rd in zip(rec_p, reads):
                f.write(json.dumps({**meta(p, c), **rd, "git": rev}) + "\n")
    log(f"[p3dev] baselines done ({time.time()-t0:.0f}s), {len(rec_p)} rows per condition")
    regs = [P.regions(sc, p, p.speaker, other(p.speaker), author=p.author) for p in rec_p]
    K = {c: [P.common_suffix(a.ids, b.ids) for a, b in zip(rows["assistant_insult"], rows[c])] for c in ("flat_v2", "third_party_v2")}
    f = open(out / "phase3_dev_patch.jsonl", "w")
    for L in args.layers:
        for donor, sets in (("flat_v2", SETS_FULL), ("third_party_v2", SETS_TP)):
            k = K[donor]
            _, cap = P.run(sc, rows[donor], bs=args.bs, capture_layers=[L], capture_k=k)
            for name in sets:
                masks = [posset(name, regs[i], k[i]) for i in range(len(rec_p))]
                reads, _ = P.run(sc, rows["assistant_insult"], bs=args.bs, patch={L: (cap[L], masks, "replace")})
                for p, m, rd in zip(rec_p, masks, reads):
                    f.write(json.dumps({**meta(p, "assistant_insult"), "layer": L, "donor": donor, "posset": name,
                                        "n_patched": int(m.sum()), **rd, "git": rev}) + "\n")
            f.flush()
        log(f"[p3dev] layer {L} done ({time.time()-t0:.0f}s)")
    f.close()
    log(f"[p3dev] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
