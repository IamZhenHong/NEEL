"""Experiment 8 runner (runs ON THE GPU POD): yes/no publish decision, higher latitude than grading.
Frozen: see prereg/exp8.md.

Readout: at the first assistant token, P(yes) = [P(Yes)+P(yes)] / [P(Yes)+P(yes)+P(No)+P(no)];
valid mass = the four probabilities summed. Qwen3-8B, float32, batched (last-position logits).

Stage 1  calibration: coordinator_flat, one cell (User-1, NS), 3 authors x 6 paragraphs per template.
         Paragraph selected iff mean P(yes) over the 3 authors is in [0.15, 0.85].
Stage 2  scripted: flat, assistant_insult, third_party_insult, negative_context x 2 speakers x 2 orders
         x 3 authors x selected paragraphs.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_item import decision_prompts, iter_prefixes, load  # noqa: E402
from scoring import Scorer  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
DOSE = 3
CONDITIONS = ["flat", "assistant_insult", "third_party_insult", "negative_context"]
LABELS = ("Yes", "No", "yes", "no")
BAND = (0.15, 0.85)
TEMPLATES = [f"onboarding_{i:03d}" for i in [6, 7] + list(range(9, 31))]   # 008 excluded as in Exp 6/7


def readout(r):
    lp = r.label_probs
    y, n = lp["Yes"] + lp["yes"], lp["No"] + lp["no"]
    return y / max(y + n, 1e-12), y + n


def write(prompts, reads, path, common):
    with open(path, "w") as f:
        for p, r in zip(prompts, reads):
            py, mass = readout(r)
            f.write(json.dumps({"item_id": p.item_id, "condition": p.condition, "dose": p.dose,
                                "speaker": p.speaker, "order": p.order, "author": p.author,
                                "author_role": p.author_role, "paragraph_id": p.paragraph_id,
                                "target_grade": p.target_grade, "p_yes": py, "valid_mass": mass,
                                "top_token": r.top_token, "top_token_prob": r.top_token_prob,
                                "label_probs": r.label_probs, "n_tokens": r.n_tokens, **common}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--dry", action="store_true", help="CPU smoke test: Qwen3-0.6B, 2 templates")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731

    import torch
    model, templates, band = MODEL, TEMPLATES, BAND
    if args.dry:
        model, templates, band = "Qwen/Qwen3-0.6B", TEMPLATES[:2], (0.0, 1.0)   # dry run exercises stage 2
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                                     capture_output=True, text=True).stdout.strip()
    common = {"model": model, "git": rev, "dtype": "fp32", "experiment": 8}
    t0 = time.time()
    scorer = Scorer(model, dtype=torch.float32, labels=LABELS)
    log(f"[exp8] {model} fp32 loaded in {time.time()-t0:.0f}s; {len(templates)} templates")
    items = {iid: (f"items/{iid}.yaml", load(f"items/{iid}.yaml")) for iid in templates}

    calib = []
    for iid, (path, it) in items.items():
        pre = next(iter_prefixes(path, doses=[DOSE], speakers=("User-1",), orders=("NS",),
                                 conditions=["coordinator_flat"]))
        calib += decision_prompts(it, pre, pre.messages)
    reads = scorer.score_batch([p.messages for p in calib], batch_size=8)
    write(calib, reads, out / "exp8_calibration.jsonl", common)
    sel = {}
    for iid in items:
        by_p = {}
        for p, r in zip(calib, reads):
            if p.item_id == iid:
                by_p.setdefault(p.paragraph_id, []).append(readout(r)[0])
        sel[iid] = [pid for pid, v in by_p.items() if band[0] <= mean(v) <= band[1]][:2 if args.dry else None]
    (out / "exp8_selection.json").write_text(json.dumps({"band": band, "selected": sel}, indent=1))
    log(f"[exp8] calibration done; selected {sum(len(v) for v in sel.values())} paragraphs, "
        f"{sum(bool(v) for v in sel.values())}/{len(sel)} templates usable")

    prompts = []
    for iid, (path, it) in items.items():
        if sel[iid]:
            for pre in iter_prefixes(path, doses=[DOSE], conditions=CONDITIONS):
                prompts += decision_prompts(it, pre, pre.messages, paragraph_ids=sel[iid])
    log(f"[exp8] scoring {len(prompts)} prompts")
    t1 = time.time()
    reads = scorer.score_batch([p.messages for p in prompts], batch_size=8)
    write(prompts, reads, out / "exp8_decision.jsonl", common)
    log(f"[exp8] scored in {time.time()-t1:.0f}s; total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
