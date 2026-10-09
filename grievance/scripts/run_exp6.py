"""Experiment 6 (confirmatory) runner. Runs ON THE GPU POD. Frozen: see prereg/exp6.md.

Stage 1  calibration: scripted coordinator_flat (no user is the corrector), one cell
         (speaker User-1, order NS), 3 authors x 6 paragraphs per template. Paragraph p is
         selected iff mean top-label probability over the 3 authors <= SELECT_THRESHOLD.
Stage 2  scripted (Phase 1a): flat and assistant_insult x 2 speakers x 2 orders x 3 authors
         x selected paragraphs.
Stage 3  free replies (Phase 1b): the model writes all 6 assistant turns after the first
         treatment turn (greedy), then the same review prompts on its own history.
All scoring in float32, batched (verified equal to single-prompt scoring within 2e-4).
"""
from __future__ import annotations

import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_item import iter_prefixes, load, review_prompts  # noqa: E402
from generation import generate_batch  # noqa: E402
from scoring import Scorer  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
DOSE = 3
CONDITIONS = ["flat", "assistant_insult"]
SELECT_THRESHOLD = 0.85
ITEMS = [f"items/onboarding_{i:03d}.yaml" for i in range(6, 31)]


def meta(p):
    return {"item_id": p.item_id, "version": p.version, "condition": p.condition, "dose": p.dose,
            "speaker": p.speaker, "order": p.order, "author": p.author, "author_role": p.author_role,
            "paragraph_id": p.paragraph_id, "target_grade": p.target_grade}


def score_and_write(scorer, prompts, path, common, log):
    t0 = time.time()
    reads = scorer.score_batch([p.messages for p in prompts], batch_size=8)
    with open(path, "w") as f:
        for p, r in zip(prompts, reads):
            f.write(json.dumps({**meta(p), **common, **r.as_dict()}) + "\n")
    log(f"  scored {len(prompts)} prompts in {time.time()-t0:.0f}s -> {path}")
    return reads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--reply-tokens", type=int, default=100)
    ap.add_argument("--dry", action="store_true", help="tiny CPU smoke test with Qwen3-0.6B, 2 templates")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731

    import torch
    model, items, dtype = MODEL, ITEMS, torch.float32
    if args.dry:
        model, items = "Qwen/Qwen3-0.6B", ITEMS[:2]
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    missing = [p for p in items if not Path(p).exists()]
    if missing:
        sys.exit(f"missing templates: {missing}")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                                     capture_output=True, text=True).stdout.strip()
    common = {"model": model, "git": rev, "dtype": "fp32", "experiment": 6}
    t0 = time.time()
    scorer = Scorer(model, dtype=dtype)
    log(f"[exp6] {model} fp32 loaded in {time.time()-t0:.0f}s; {len(items)} templates")
    loaded = {load(p)["id"]: (p, load(p)) for p in items}

    # ---- stage 1: calibration and selection ------------------------------------------------
    calib = []
    for iid, (path, it) in loaded.items():
        pre = next(iter_prefixes(path, doses=[DOSE], speakers=("User-1",), orders=("NS",),
                                 conditions=["coordinator_flat"]))
        calib += review_prompts(it, pre, pre.messages)
    reads = score_and_write(scorer, calib, out / "exp6_calibration.jsonl", common, log)
    sel = {}
    for iid in loaded:
        by_p = {}
        for p, r in zip(calib, reads):
            if p.item_id == iid:
                by_p.setdefault(p.paragraph_id, []).append(r.top_label_prob)
        sel[iid] = [pid for pid, v in by_p.items() if mean(v) <= SELECT_THRESHOLD]
    (out / "exp6_selection.json").write_text(json.dumps({"threshold": SELECT_THRESHOLD, "selected": sel}, indent=1))
    n_usable = sum(bool(v) for v in sel.values())
    log(f"[exp6] selection: {sum(len(v) for v in sel.values())} paragraphs, {n_usable}/{len(sel)} templates usable")

    # ---- stage 2: scripted --------------------------------------------------------------------
    scripted = []
    for iid, (path, it) in loaded.items():
        if not sel[iid]:
            continue
        for pre in iter_prefixes(path, doses=[DOSE], conditions=CONDITIONS):
            scripted += review_prompts(it, pre, pre.messages, paragraph_ids=sel[iid])
    score_and_write(scorer, scripted, out / "exp6_scripted.jsonl", {**common, "mode": "scripted"}, log)

    # ---- stage 3: free replies --------------------------------------------------------------
    prefixes = [pre for iid, (path, _) in loaded.items() if sel[iid]
                for pre in iter_prefixes(path, doses=[DOSE], conditions=CONDITIONS, generate=True)]
    n_gen = max(sum(1 for m in pre.messages if m.get("gen")) for pre in prefixes)
    for k in range(n_gen):
        todo = []
        for pre in prefixes:
            g = [i for i, m in enumerate(pre.messages) if m.get("gen")]
            if g:
                todo.append((pre, g[0]))
        t1 = time.time()
        replies = generate_batch(scorer, [[{"role": m["role"], "content": m["content"]} for m in pre.messages[:i]]
                                          for pre, i in todo], max_new_tokens=args.reply_tokens)
        for (pre, i), r in zip(todo, replies):
            pre.messages[i] = {"role": "assistant", "content": r}
        log(f"  generated turn {k+1}/{n_gen} for {len(todo)} histories in {time.time()-t1:.0f}s")
    assert all(not m.get("gen") and m["content"] for pre in prefixes for m in pre.messages)
    with open(out / "exp6_freereply.transcripts.jsonl", "w") as f:
        for pre in prefixes:
            f.write(json.dumps({"item_id": pre.item_id, "condition": pre.condition, "speaker": pre.speaker,
                                "order": pre.order, "messages": pre.messages}) + "\n")
    free = []
    for pre in prefixes:
        free += review_prompts(loaded[pre.item_id][1], pre, pre.messages, paragraph_ids=sel[pre.item_id])
    score_and_write(scorer, free, out / "exp6_freereply.jsonl", {**common, "mode": "freereply"}, log)
    log(f"[exp6] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
