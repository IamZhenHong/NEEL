"""Experiment 13 held-out test (runs ON THE GPU POD; fp32). Frozen: prereg/exp13.md.
Vector: results/exp13_vectors.pt (all 12 dev templates), layer 20, alpha 4 (selected on dev by the fixed rule).
Conditions assistant_insult and flat_v2, all three authors, 12 held-out templates.
Interventions: none; steer (alpha * v at request tokens except grade); random (norm-matched, seed 0).
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402
from build_item import build  # noqa: E402
from scoring import Scorer  # noqa: E402
import patching as P  # noqa: E402

LAYER, ALPHA = 20, 4.0


def other(u):
    return "User-1" if u == "User-2" else "User-2"


def main():
    log = lambda s: print(s, flush=True)  # noqa: E731
    dry = "--dry" in sys.argv
    split = json.loads(Path("prereg/phase2_split.json").read_text())
    sel = json.loads(Path("prereg/exp6_selection.json").read_text())["selected"]
    vec = torch.load("prereg/exp13_vectors.pt")["vectors"][LAYER] * ALPHA
    templates, model, layer = split["held_out"], "Qwen/Qwen3-8B", LAYER
    if dry:
        templates, model, layer = split["dev"][:2], "Qwen/Qwen3-0.6B", 12
        vec = torch.zeros(1024)
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(vec.shape, generator=g); rnd = rnd / rnd.norm() * vec.norm()
    log(f"[exp13ho] {model} fp32 loaded in {time.time()-t0:.0f}s; |v| {vec.norm():.2f}")
    Path("results").mkdir(exist_ok=True)
    f = open("results/exp13_heldout.jsonl", "w")
    for c in ("assistant_insult", "flat_v2"):
        ps = [p for it in templates for p in build(f"items/{it}.yaml", doses=[3], conditions=[c], paragraph_ids=sel[it])]
        rows = [P.prepare(sc, p, {}) for p in ps]
        req = [[o for o in P.regions(sc, p, p.speaker, other(p.speaker), author=p.author)["request"] if o >= 2] for p in ps]
        for kind, v in (("none", None), ("steer", vec), ("random", rnd)):
            reads = P.run(sc, rows, bs=4)[0] if v is None else P.run_steer(sc, rows, layer, [v] * len(rows), req, bs=4)
            for p, rd in zip(ps, reads):
                f.write(json.dumps({"item_id": p.item_id, "condition": c, "kind": kind, "speaker": p.speaker, "order": p.order,
                                    "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id, **rd, "git": rev}) + "\n")
            f.flush()
            log(f"[exp13ho] {c} {kind} done ({time.time()-t0:.0f}s)")
    f.close()
    log(f"[exp13ho] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
