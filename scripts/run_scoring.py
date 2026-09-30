"""Score every prompt built from one or more item files. Runs ON THE GPU POD.

Usage:
  python scripts/run_scoring.py --items items/onboarding_001.yaml --out results/run.jsonl \
      [--model Qwen/Qwen3-8B] [--speakers User-1 User-2] [--complete-n 6]

Writes one JSON line per prompt with the readout (expected score, valid mass, top label,
top token, label probs) plus all condition metadata. `--complete-n` also runs greedy
completions on that many random prompts as the spot-check from design v4.1 §7.1.
"""
from __future__ import annotations

import argparse, json, os, random, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_item import build, render_plain  # noqa: E402
from scoring import Scorer  # noqa: E402


def git_rev() -> str:
    if os.environ.get("GIT_REV"):
        return os.environ["GIT_REV"]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--speakers", nargs="+", default=["User-1", "User-2"])
    ap.add_argument("--complete-n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-cpu", action="store_true", help="permit running without a GPU (local dry runs)")
    args = ap.parse_args()

    prompts = []
    for path in args.items:
        for sp in args.speakers:
            prompts += build(path, speaker=sp)
    print(f"{len(prompts)} prompts from {len(args.items)} item(s) x {len(args.speakers)} speaker(s)")

    import torch
    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit("CUDA not available; refusing to run an 8B model on CPU (pass --allow-cpu for dry runs)")
    t0 = time.time()
    scorer = Scorer(args.model)
    print(f"model loaded in {time.time()-t0:.0f}s on {scorer.device}")
    print("generation tail:", repr(scorer.render(prompts[0].messages)[-60:]))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rev = git_rev()
    with out.open("w") as f:
        for i, p in enumerate(prompts):
            r = scorer.score(p.messages)
            row = {"item_id": p.item_id, "condition": p.condition, "speaker": p.speaker,
                   "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id,
                   "target_grade": p.target_grade, "model": args.model, "git": rev, **r.as_dict()}
            f.write(json.dumps(row) + "\n")
            if i % 12 == 0:
                print(f"  [{i+1}/{len(prompts)}] {p.condition:20} {p.author_role:9} {p.paragraph_id:13} "
                      f"E={r.expected_score:.2f} M={r.valid_mass:.3f} top={r.top_label} tok={r.top_token!r}")
    print(f"scored {len(prompts)} prompts in {time.time()-t0:.0f}s -> {out}")

    if args.complete_n:
        random.seed(args.seed)
        sample = random.sample(prompts, min(args.complete_n, len(prompts)))
        comp_path = out.with_suffix(".completions.txt")
        with comp_path.open("w") as f:
            for p in sample:
                text = scorer.complete(p.messages, max_new_tokens=30)
                f.write(f"### {p.condition} | {p.author_role} | {p.paragraph_id} -> {text!r}\n")
        print(f"completions -> {comp_path}")


if __name__ == "__main__":
    main()
