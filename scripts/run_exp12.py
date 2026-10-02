"""Experiment 12 (runs ON THE GPU POD): all-layer path blocking. Frozen: prereg/exp12.md.

Which route carries the provocation information to the grade? Qwen3-8B fp32 eager, scripted versioned
prompts, conditions assistant_insult, flat_v2, third_party_v2, all three authors, frozen Exp 6 paragraph
selection, all 24 templates. Blocks apply at EVERY layer and in every condition alike:

  none      unmodified
  B_tail    positions after the treatment block and before the grading request cannot attend to the
            treatment block (the three treatment turns and the interim acknowledgements between them)
  B_req     grading-request tokens (incl. the grade position) cannot attend to the treatment block
  B_both    both of the above (everything after the treatment block is cut off from it)
  B_ctrl    tail positions cannot attend to a count-matched span of earlier non-treatment context
            (system prompt and opening messages, excluding the first 4 tokens)
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402
from build_item import build  # noqa: E402
from scoring import Scorer  # noqa: E402
import patching as P  # noqa: E402

CONDS = ("assistant_insult", "flat_v2", "third_party_v2")
BLOCKS = ("none", "B_tail", "B_req", "B_both", "B_ctrl")


def other(u):
    return "User-1" if u == "User-2" else "User-2"


def key(p):
    return (p.item_id, p.speaker, p.order, p.author, p.paragraph_id)


def spans(sc, p, partner):
    """Offsets-from-end for: treatment block, tail (after treatment, before request), request, control."""
    r = P.prepare(sc, p, {})
    K = P.common_suffix(r.ids, P.prepare(sc, partner, {}).ids)   # shared tail starts right after the treatment block
    g = P.regions(sc, p, p.speaker, other(p.speaker), author=p.author)
    n = len(r.ids)
    first_treat = max(o for o in g["prov_msgs"] if o > K)          # earliest treatment token (largest offset)
    treat = list(range(K + 1, first_treat + 1))                     # treatment turns + interim acks between them
    req = set(g["request"])
    tail = [o for o in range(1, K + 1) if o not in req]
    early = [o for o in range(first_treat + 1, n - 3)]              # before the treatment block, skip first 4 tokens
    early = sorted(early)[:len(treat)]                              # nearest to the treatment block first
    return r, treat, tail, sorted(req), early


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    sel = json.loads(Path("prereg/exp6_selection.json").read_text())["selected"]
    templates = sorted(k for k, v in sel.items() if v)
    model = "Qwen/Qwen3-8B"
    if args.dry:
        templates, model = templates[:2], "Qwen/Qwen3-0.6B"
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32, attn_implementation="eager")
    log(f"[exp12] {model} fp32 eager loaded in {time.time()-t0:.0f}s; {len(templates)} templates")
    pr = {c: [p for it in templates for p in build(f"items/{it}.yaml", doses=[3], conditions=[c], paragraph_ids=sel[it])] for c in CONDS}
    by = {c: {key(p): p for p in pr[c]} for c in CONDS}
    Path("results").mkdir(exist_ok=True)   # added after a failed first launch (directory missing on the pod)
    f = open(Path("results") / "exp12.jsonl", "w")
    for c in CONDS:
        partner_c = "flat_v2" if c != "flat_v2" else "assistant_insult"
        ps = pr[c]
        info = [spans(sc, p, by[partner_c][key(p)]) for p in ps]
        rows = [x[0] for x in info]
        sets = {"none": [None] * len(ps),
                "B_tail": [(x[2], x[1]) for x in info],
                "B_req": [(x[3], x[1]) for x in info],
                "B_both": [(x[2] + x[3], x[1]) for x in info],
                "B_ctrl": [(x[2], x[4]) for x in info]}
        for b in BLOCKS:
            reads = P.run_blocked(sc, rows, sets[b], bs=args.bs)
            for p, rd in zip(ps, reads):
                f.write(json.dumps({"item_id": p.item_id, "condition": c, "block": b, "speaker": p.speaker, "order": p.order,
                                    "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id,
                                    **rd, "git": rev}) + "\n")
            f.flush()
            log(f"[exp12] {c} {b} done ({time.time()-t0:.0f}s); treatment {sum(len(x[1]) for x in info)/len(info):.0f} tok, "
                f"tail {sum(len(x[2]) for x in info)/len(info):.0f}, request {sum(len(x[3]) for x in info)/len(info):.0f}")
    f.close()
    log(f"[exp12] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
