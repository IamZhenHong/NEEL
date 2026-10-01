"""Experiment 7 runner (runs ON THE GPU POD). Frozen: see prereg/exp7.md.

Scripted (Phase 1a) scoring of four conditions on the 24 usable Experiment 6 templates, using the
frozen Experiment 6 paragraph selection (no recalibration). Qwen3-8B, float32, batched scoring.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_item import iter_prefixes, load, review_prompts  # noqa: E402
from scoring import Scorer  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
DOSE = 3
CONDITIONS = ["flat", "assistant_insult", "third_party_insult", "negative_context"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/exp6_selection.json")
    ap.add_argument("--out", default="results/exp7_scripted.jsonl")
    ap.add_argument("--dry", action="store_true", help="CPU smoke test: Qwen3-0.6B, 2 templates")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731

    import torch
    sel = json.loads(Path(args.selection).read_text())["selected"]
    usable = {k: v for k, v in sel.items() if v}
    model = MODEL
    if args.dry:
        model, usable = "Qwen/Qwen3-0.6B", dict(list(usable.items())[:2])
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                                     capture_output=True, text=True).stdout.strip()
    common = {"model": model, "git": rev, "dtype": "fp32", "experiment": 7, "mode": "scripted"}
    t0 = time.time()
    scorer = Scorer(model, dtype=torch.float32)
    log(f"[exp7] {model} fp32 loaded in {time.time()-t0:.0f}s; {len(usable)} templates")

    prompts = []
    for iid, paras in usable.items():
        path = f"items/{iid}.yaml"
        it = load(path)
        missing = [c for c in CONDITIONS if c not in it["treatments"][DOSE]]
        if missing:
            sys.exit(f"{iid} missing conditions {missing}")
        for pre in iter_prefixes(path, doses=[DOSE], conditions=CONDITIONS):
            prompts += review_prompts(it, pre, pre.messages, paragraph_ids=paras)
    log(f"[exp7] scoring {len(prompts)} prompts")
    t1 = time.time()
    reads = scorer.score_batch([p.messages for p in prompts], batch_size=8)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for p, r in zip(prompts, reads):
            f.write(json.dumps({"item_id": p.item_id, "condition": p.condition, "dose": p.dose,
                                "speaker": p.speaker, "order": p.order, "author": p.author,
                                "author_role": p.author_role, "paragraph_id": p.paragraph_id,
                                "target_grade": p.target_grade, **common, **r.as_dict()}) + "\n")
    log(f"[exp7] scored {len(prompts)} in {time.time()-t1:.0f}s -> {out}; total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
