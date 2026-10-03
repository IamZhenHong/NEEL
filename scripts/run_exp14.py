"""Experiment 14 runner (runs ON THE GPU POD): cross-family replication on Gemma 4 12B-it. Frozen: prereg/exp14.md.

  --stage calib      fresh-context + coordinator_flat grading on all templates/paragraphs (bf16).
                     Writes results/exp14_calibration.jsonl, results/exp14_selection.json, completions.
  --stage replicate  flat_v2 / assistant_insult / third_party_v2 x 2 speakers x 2 orders x 3 authors x
                     selected paragraphs (reads results/exp14_selection.json; fp32 by default).
                     Writes results/exp14_replicate.jsonl.
"""
from __future__ import annotations

import argparse, glob, json, os, random, subprocess, sys, time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402
from build_item import build_fresh, iter_prefixes, load, review_prompts  # noqa: E402
from scoring import Scorer  # noqa: E402

MODEL = "google/gemma-4-12B-it"
DOSE = 3
CONDITIONS = ["flat_v2", "assistant_insult", "third_party_v2"]
SELECT_THRESHOLD = 0.85
TEMPLATES = [f"items/onboarding_{i:03d}.yaml" for i in range(6, 31)]   # 008 included: selection is per model


def meta(p):
    return {"item_id": p.item_id, "condition": p.condition, "dose": p.dose, "speaker": p.speaker, "order": p.order,
            "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id, "target_grade": p.target_grade}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["calib", "replicate"], required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default=None)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    model, templates = args.model, TEMPLATES
    dtype = args.dtype or ("bf16" if args.stage == "calib" else "fp32")
    if args.dry:
        model, templates, dtype = "Qwen/Qwen3-0.6B", TEMPLATES[:2], "fp32"
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = Path("results"); out.mkdir(exist_ok=True)
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32 if dtype == "fp32" else torch.bfloat16)
    log(f"[exp14 {args.stage}] {model} {dtype} loaded in {time.time()-t0:.0f}s; {len(templates)} templates")
    log(f"[exp14] generation tail: {sc.render([{'role': 'user', 'content': 'x'}])[-80:]!r}")
    common = {"model": model, "git": rev, "dtype": dtype, "experiment": 14}

    if args.stage == "calib":
        prompts = []
        for path in templates:
            it = load(path)
            prompts += build_fresh(path)
            pre = next(iter_prefixes(path, doses=[DOSE], speakers=("User-1",), orders=("NS",), conditions=["coordinator_flat"]))
            prompts += review_prompts(it, pre, pre.messages)
        log(f"[exp14 calib] scoring {len(prompts)} prompts")
        reads = sc.score_batch([p.messages for p in prompts], batch_size=args.bs)
        with open(out / "exp14_calibration.jsonl", "w") as f:
            for p, r in zip(prompts, reads):
                f.write(json.dumps({**meta(p), **common, **r.as_dict()}) + "\n")
        sel = {}
        for path in templates:
            iid = load(path)["id"]
            by_p = {}
            for p, r in zip(prompts, reads):
                if p.item_id == iid and p.condition == "coordinator_flat":
                    by_p.setdefault(p.paragraph_id, []).append(r.top_label_prob)
            sel[iid] = [pid for pid, v in by_p.items() if mean(v) <= SELECT_THRESHOLD]
        (out / "exp14_selection.json").write_text(json.dumps({"threshold": SELECT_THRESHOLD, "model": model, "selected": sel}, indent=1))
        random.seed(0)
        sample = random.sample(prompts, min(12, len(prompts)))
        with open(out / "exp14_calibration.completions.txt", "w") as f:
            for p in sample:
                f.write(f"### {p.condition} | {p.author_role} | {p.paragraph_id} -> {sc.complete(p.messages, max_new_tokens=20)!r}\n")
        log(f"[exp14 calib] selected {sum(len(v) for v in sel.values())} paragraphs, {sum(bool(v) for v in sel.values())}/{len(sel)} templates; "
            f"done in {time.time()-t0:.0f}s")
    else:
        sel = json.loads((out / "exp14_selection.json").read_text())["selected"]
        prompts = []
        for path in templates:
            it = load(path)
            if not sel.get(it["id"]):
                continue
            for pre in iter_prefixes(path, doses=[DOSE], conditions=CONDITIONS):
                prompts += review_prompts(it, pre, pre.messages, paragraph_ids=sel[it["id"]])
        log(f"[exp14 replicate] scoring {len(prompts)} prompts")
        reads = sc.score_batch([p.messages for p in prompts], batch_size=args.bs)
        with open(out / "exp14_replicate.jsonl", "w") as f:
            for p, r in zip(prompts, reads):
                f.write(json.dumps({**meta(p), **common, **r.as_dict()}) + "\n")
        log(f"[exp14 replicate] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
