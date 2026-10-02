"""Experiment 10 runner (runs ON THE GPU POD): anonymised author + attention blocking. Frozen: prereg/exp10.md.

Qwen3-8B, float32, eager attention (needed for blocking), scripted versioned Exp 9 prompts,
conditions assistant_insult and third_party_v2, frozen Exp 6 paragraph selection, all 24 templates.

  named      unmodified review prompts, all three authors (baseline for everything)
  anon       the author's handle in the grading request replaced by "a team member"
             (one prompt per condition x speaker x order x paragraph)
  B1_label   every layer: author-label tokens in the request cannot attend to the treatment turns
  B2_request every layer: every token of the grading request (incl. the grade position) cannot attend
             to the treatment turns
  B3_control every layer: request tokens cannot attend to a count-matched span of other pre-request
             conversation tokens (excluding the insulter's messages)
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402
from build_item import build  # noqa: E402
from scoring import Scorer  # noqa: E402
import patching as P  # noqa: E402

CONDS = ("assistant_insult", "third_party_v2")


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
    log(f"[exp10] {model} fp32 eager loaded in {time.time()-t0:.0f}s; {len(templates)} templates")

    pr = {c: [p for it in templates for p in build(f"items/{it}.yaml", doses=[3], conditions=[c], paragraph_ids=sel[it])]
          for c in CONDS}
    partner = {c: {key(p): p for p in pr[o]} for c, o in (("assistant_insult", "third_party_v2"), ("third_party_v2", "assistant_insult"))}
    out = Path("results"); out.mkdir(exist_ok=True)
    meta = lambda p, c: {"item_id": p.item_id, "condition": c, "speaker": p.speaker, "order": p.order,  # noqa: E731
                         "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id}
    f = open(out / "exp10.jsonl", "w")

    def write(kind, c, ps, reads, extra=None):
        for p, rd in zip(ps, reads):
            f.write(json.dumps({**meta(p, c), "kind": kind, **(extra or {}), **rd, "git": rev}) + "\n")
        f.flush()

    for c in CONDS:
        ps = pr[c]
        rows = [P.prepare(sc, p, {}) for p in ps]
        reads, _ = P.run(sc, rows, bs=args.bs)
        write("named", c, ps, reads)
        log(f"[exp10] {c} named done ({time.time()-t0:.0f}s), {len(ps)} rows")

        anon_ps, seen = [], set()
        for p in ps:
            k = (p.item_id, p.speaker, p.order, p.paragraph_id)
            if k not in seen:
                seen.add(k); anon_ps.append(P.anonymise(p))
        reads, _ = P.run(sc, [P.prepare(sc, p, {}) for p in anon_ps], bs=args.bs)
        for p in anon_ps:
            p.author_role = "anonymous"
        write("anon", c, anon_ps, reads)
        log(f"[exp10] {c} anon done ({time.time()-t0:.0f}s), {len(anon_ps)} rows")

        # block position sets
        treat, lab, req, ctrl = [], [], [], []
        for p, r in zip(ps, rows):
            q = partner[c][key(p)]
            K = P.common_suffix(r.ids, P.prepare(sc, q, {}).ids)
            g = P.regions(sc, p, p.speaker, other(p.speaker), author=p.author)
            t = [o for o in g["prov_msgs"] if o > K]                        # treatment turns (before the shared suffix)
            pre_req = sorted([o for o in range(1, K + 1) if o not in set(g["request"]) and o not in set(g["prov_msgs"])],
                             reverse=True)                                # earliest first
            treat.append(t); lab.append(g["author_label"]); req.append(g["request"])
            ctrl.append(pre_req[:len(t)])
        for name, blocks in (("B1_label", [(lab[i], treat[i]) for i in range(len(ps))]),
                             ("B2_request", [(req[i], treat[i]) for i in range(len(ps))]),
                             ("B3_control", [(req[i], ctrl[i]) for i in range(len(ps))])):
            reads = P.run_blocked(sc, rows, blocks, bs=args.bs)
            write(name, c, ps, reads, {"n_query": None, "n_key": None})
            log(f"[exp10] {c} {name} done ({time.time()-t0:.0f}s); mean blocked keys "
                f"{sum(len(b[1]) for b in blocks)/len(blocks):.0f}, queries {sum(len(b[0]) for b in blocks)/len(blocks):.0f}")
    f.close()
    log(f"[exp10] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
