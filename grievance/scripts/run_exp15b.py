"""Experiment 15b (adapted) runner (runs ON THE GPU POD): cross-family replication on Phi-4. Frozen: prereg/exp15b.md.

  --stage calib      fresh-context + coordinator_flat grading on all templates/paragraphs (bf16).
                     Writes results/exp15b_calibration.jsonl, results/exp15b_selection.json, completions.
  --stage replicate  flat_v2 / assistant_insult / third_party_v2 x 2 speakers x 2 orders x 3 authors x
                     selected paragraphs (reads results/exp15b_selection.json; fp32 by default).
                     Writes results/exp15b_replicate.jsonl.
"""
from __future__ import annotations

import argparse, glob, json, os, random, subprocess, sys, time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402
from build_item import build_fresh, iter_prefixes, load, review_prompts  # noqa: E402
from scoring import Scorer  # noqa: E402

MODEL = "microsoft/phi-4"
DOSE = 3
CONDITIONS = ["flat_v2", "assistant_insult", "third_party_v2"]
SELECT_THRESHOLD = 0.85
STAGE2_CAP_TOTAL, STAGE2_CAP_PER_TEMPLATE = 60, 3
TEMPLATES = [f"items/onboarding_{i:03d}.yaml" for i in range(6, 31) if i != 8]   # 008 has no v2 control turns
# The one adaptation for Phi-4 (prereg/exp15b.md): appended to every system message, both stages.
FORMAT_LINE = "When asked for a grade, reply with the single digit only, nothing else."


def adapt(prompts):
    for p in prompts:
        assert p.messages[0]["role"] == "system"
        p.messages[0]["content"] = p.messages[0]["content"].rstrip() + "\n" + FORMAT_LINE
    return prompts



def meta(p):
    return {"item_id": p.item_id, "condition": p.condition, "dose": p.dose, "speaker": p.speaker, "order": p.order,
            "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id, "target_grade": p.target_grade}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["calib", "replicate"], required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default=None)
    ap.add_argument("--bs", type=int, default=2)   # 14.7B fp32 on 80 GB: leave activation headroom
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
    if dtype == "fp32" and torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 60e9:
        dtype = "bf16"   # fp32 weights do not fit a 48 GB card; recorded in every row's dtype field
        log("[exp15b] GPU under 60 GB: falling back to bf16")
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32 if dtype == "fp32" else torch.bfloat16)
    log(f"[exp15b {args.stage}] {model} {dtype} loaded in {time.time()-t0:.0f}s; {len(templates)} templates")
    log(f"[exp15b] generation tail: {sc.render([{'role': 'user', 'content': 'x'}])[-80:]!r}")
    common = {"model": model, "git": rev, "dtype": dtype, "experiment": "15b"}

    if args.stage == "calib":
        prompts = []
        for path in templates:
            it = load(path)
            prompts += build_fresh(path)
            pre = next(iter_prefixes(path, doses=[DOSE], speakers=("User-1",), orders=("NS",), conditions=["coordinator_flat"]))
            prompts += review_prompts(it, pre, pre.messages)
        adapt(prompts)
        log(f"[exp15b calib] scoring {len(prompts)} prompts (format line appended to system message)")
        reads = sc.score_batch([p.messages for p in prompts], batch_size=args.bs)
        with open(out / "exp15b_calibration.jsonl", "w") as f:
            for p, r in zip(prompts, reads):
                f.write(json.dumps({**meta(p), **common, **r.as_dict()}) + "\n")
        sel, ptop = {}, {}
        for path in templates:
            iid = load(path)["id"]
            by_p = {}
            for p, r in zip(prompts, reads):
                if p.item_id == iid and p.condition == "coordinator_flat":
                    by_p.setdefault(p.paragraph_id, []).append(r.top_label_prob)
            sel[iid] = [pid for pid, v in by_p.items() if mean(v) <= SELECT_THRESHOLD]
            ptop[iid] = {pid: mean(v) for pid, v in by_p.items()}
        # prereg/exp15b.md: if more than 60 paragraphs pass, Stage 2 uses at most 3 per template, the 3 with
        # mean p(top) closest to 0.5 (calibration data only; fixed here, before Stage 2).
        stage2 = sel
        if sum(len(v) for v in sel.values()) > STAGE2_CAP_TOTAL:
            stage2 = {iid: sorted(v, key=lambda pid: abs(ptop[iid][pid] - 0.5))[:STAGE2_CAP_PER_TEMPLATE] for iid, v in sel.items()}
        (out / "exp15b_selection.json").write_text(json.dumps({"threshold": SELECT_THRESHOLD, "model": model, "selected": sel,
                                                               "stage2": stage2, "format_line": FORMAT_LINE}, indent=1))
        log(f"[exp15b calib] stage-2 set: {sum(len(v) for v in stage2.values())} paragraphs")
        random.seed(0)
        sample = random.sample(prompts, min(12, len(prompts)))
        with open(out / "exp15b_calibration.completions.txt", "w") as f:
            for p in sample:
                f.write(f"### {p.condition} | {p.author_role} | {p.paragraph_id} -> {sc.complete(p.messages, max_new_tokens=20)!r}\n")
        log(f"[exp15b calib] selected {sum(len(v) for v in sel.values())} paragraphs, {sum(bool(v) for v in sel.values())}/{len(sel)} templates; "
            f"done in {time.time()-t0:.0f}s")
    else:
        sel_path = Path("prereg/exp15b_selection.json") if Path("prereg/exp15b_selection.json").exists() else out / "exp15b_selection.json"
        sj = json.loads(sel_path.read_text()); sel = sj.get("stage2") or sj["selected"]
        prompts = []
        for path in templates:
            it = load(path)
            if not sel.get(it["id"]):
                continue
            for pre in iter_prefixes(path, doses=[DOSE], conditions=CONDITIONS):
                prompts += review_prompts(it, pre, pre.messages, paragraph_ids=sel[it["id"]])
        adapt(prompts)
        log(f"[exp15b replicate] scoring {len(prompts)} prompts (format line appended to system message)")
        reads = sc.score_batch([p.messages for p in prompts], batch_size=args.bs)
        with open(out / "exp15b_replicate.jsonl", "w") as f:
            for p, r in zip(prompts, reads):
                f.write(json.dumps({**meta(p), **common, **r.as_dict()}) + "\n")
        log(f"[exp15b replicate] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
