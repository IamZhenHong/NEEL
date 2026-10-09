"""Experiment 20 runner (runs ON THE GPU POD): base vs instruct under one few-shot plain-text grading assay. See prereg/exp20.md.

  --stage 1  [--arms base_assay instruct_assay instruct_chat] [--bs N] [--dry]
      Calibration and the behavioural contrast. Per arm: fresh-context grading of every paragraph, coordinator_flat
      grading (User-1 / NS cell, all three authors, every paragraph), and the Exp 16 contrast rows (assistant_insult and
      third_party_v2 x 2 speakers x 2 orders x insulter and neutral authors x the Exp 6 selected paragraphs). The chat
      arm scores the contrast rows only (the in-pod chat-template reference). Writes results/exp20_s1_<arm>.jsonl and
      results/exp20_s1_<arm>_handread.txt (10 rows: prompt tail, top-5 next tokens, greedy completion).
  --stage 2  [--arms base_assay instruct_assay] [--dirs fwd rev] [--bs N] [--chunk N] [--dry]
      Exp 16's block-17 label swap and Exp 19's route specs, under the assay. Writes results/exp20_s2_<arm>_{base,swap}.jsonl
      and _plan.json (as Exp 19).

Arms: base_assay = Qwen3-8B-Base, plain assay; instruct_assay = Qwen3-8B, the same plain assay; instruct_chat = Qwen3-8B,
chat template (Exp 16-19 format). --dry uses Qwen3-0.6B-Base / Qwen3-0.6B, two templates, fp32, and asserts the Stage 2
all-heads identity.
"""
from __future__ import annotations

import argparse, json, os, random, re, subprocess, sys, time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from build_item import build, build_fresh, iter_prefixes, load, review_prompts  # noqa: E402
from scoring import Scorer  # noqa: E402
import patching as P  # noqa: E402
import headpatch as HP  # noqa: E402
import plainfmt as PF  # noqa: E402
from exp18_stats import transfer, per_template  # noqa: E402

MODELS = {"base": "Qwen/Qwen3-8B-Base", "instruct": "Qwen/Qwen3-8B"}
DRY_MODELS = {"base": "Qwen/Qwen3-0.6B-Base", "instruct": "Qwen/Qwen3-0.6B"}
ARMS = {"base_assay": ("base", "assay"), "instruct_assay": ("instruct", "assay"), "instruct_chat": ("instruct", "chat")}
CONDS = ["assistant_insult", "third_party_v2"]
DIRS = {"fwd": ("assistant_insult", "third_party_v2", 1), "rev": ("third_party_v2", "assistant_insult", -1)}
SEL_PATH = "prereg/exp6_selection.json"
# Exp 19 sites on Qwen3-8B, applied to both checkpoints by index (same architecture, instruct fine-tuned from base)
L_TAG, BAND_TOP = 17, 24
HEADS3 = [(21, 18), (21, 19), (22, 7)]
CUE_ONLY = [(18, 10), (19, 10), (19, 28), (19, 31)]
READING_JSON = "prereg/exp18_qwen_author_heads.json"
N_RANDOM, N_RAND_HEADS = 3, 3
TEMPLATES = [f"items/onboarding_{i:03d}.yaml" for i in range(6, 31) if i != 8]


def other(u):
    return "User-1" if u == "User-2" else "User-2"


def key(p):
    return (p.item_id, p.speaker, p.order, p.paragraph_id)


def meta(p):
    return {"item_id": p.item_id, "condition": p.condition, "dose": p.dose, "speaker": p.speaker, "order": p.order,
            "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id, "target_grade": p.target_grade}


def post_label_offsets(label_offs):
    lab = set(label_offs)
    return [o for o in range(1, max(lab)) if o not in lab]


def fmt(st):
    if st["tau"] is not None:
        r, lo, hi = st["tau"]
        return f"tau {r:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    m, lo, hi = st["movement"]
    return f"move {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (lever {st['lever'][0]:+.3f}, tau undefined)"


def make_row(sc, p, fmt_):
    """A patching.Row for prompt p: chat template (Exp 16-19) or the plain assay. Records the author-label offsets."""
    if fmt_ == "chat":
        r = P.prepare(sc, p, {"provoker": p.speaker, "neutral": other(p.speaker)})
        lab = P.regions(sc, p, p.speaker, other(p.speaker), author=p.author)["author_label"] if p.author_role != "fresh" else []
        text = None
    else:
        text = PF.render_assay(p.messages)
        r = P.Row(meta={}, ids=sc.tok(text, add_special_tokens=False)["input_ids"], handle_pos={})
        lab = PF.author_label_offsets(sc.tok, text, p.messages, p.author)
    r.meta = {**meta(p), "label_offsets": lab}
    return r, text


@torch.no_grad()
def handread(sc, r, text, fmt_, p):
    """Top-5 next tokens and a short greedy completion for one row."""
    if text is None:
        text = sc.render(p.messages)
    enc = sc.tok(text, return_tensors="pt", add_special_tokens=False).to(sc.device)
    logits = sc.model(**enc, logits_to_keep=1).logits[0, -1].float()
    pr = torch.softmax(logits, dim=-1)
    top = [(sc.tok.decode([int(t)]), round(float(v), 3)) for v, t in zip(*pr.topk(5))]
    gen = sc.model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=sc.tok.pad_token_id or sc.tok.eos_token_id)
    comp = sc.tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return top, comp, text


def stage1(args, log, sel, rev, out):
    arms = args.arms or ["instruct_chat", "instruct_assay", "base_assay"]   # one instruct load; reference first
    models = DRY_MODELS if args.dry else MODELS
    templates = TEMPLATES[:2] if args.dry else TEMPLATES
    # group arms by checkpoint so each model loads once
    sc, loaded = None, None
    for arm in arms:
        ck, fmt_ = ARMS[arm]
        if loaded != ck:
            del sc; torch.cuda.empty_cache() if torch.cuda.is_available() else None
            t0 = time.time()
            sc = Scorer(models[ck], dtype=torch.float32 if args.dry else torch.bfloat16)
            loaded = ck
            log(f"[exp20 s1] {models[ck]} loaded in {time.time()-t0:.0f}s; {len(P._layers(sc.model))} blocks")
        t0 = time.time()
        prompts = []
        for path in templates:
            it = load(path)
            iid = it["id"]
            if fmt_ == "assay":
                prompts += build_fresh(path)
                pre = next(iter_prefixes(path, doses=[3], speakers=("User-1",), orders=("NS",), conditions=["coordinator_flat"]))
                prompts += review_prompts(it, pre, pre.messages)
            for c in CONDS:
                prompts += [p for p in build(path, doses=[3], conditions=[c], paragraph_ids=sel[iid]) if p.author_role in ("provoker", "neutral")]
        rows, texts = [], []
        for p in prompts:
            r, text = make_row(sc, p, fmt_)
            rows.append(r); texts.append(text)
        if fmt_ == "assay":
            log(f"[exp20 s1 {arm}] preamble {len(sc.tok(PF.preamble(), add_special_tokens=False)['input_ids'])} tokens; "
                f"row lengths {min(len(r.ids) for r in rows)}-{max(len(r.ids) for r in rows)}")
        log(f"[exp20 s1 {arm}] scoring {len(prompts)} rows ({sum(p.condition == 'fresh' for p in prompts)} fresh, "
            f"{sum(p.condition == 'coordinator_flat' for p in prompts)} coordinator_flat, {sum(p.condition in CONDS for p in prompts)} contrast)")
        reads, _ = P.run(sc, rows, bs=args.bs)
        common = {"model": models[ck], "git": rev, "dtype": "fp32" if args.dry else "bf16", "experiment": 20, "stage": 1, "arm": arm, "format": fmt_}
        with open(out / f"exp20_s1_{arm}.jsonl", "w") as f:
            for r, rd in zip(rows, reads):
                f.write(json.dumps({**r.meta, **common, "n_tokens": len(r.ids), **rd}) + "\n")
        # quick summary lines (the frozen analysis is scripts/analyze_exp20.py)
        m = [rd["mass"] for rd in reads]
        log(f"[exp20 s1 {arm}] valid digit mass: mean {mean(m):.3f}, share >= 0.9: {mean(x >= 0.9 for x in m):.3f}, min {min(m):.3f}")
        for cond in ("fresh", "coordinator_flat"):
            rs = [(p, rd) for p, rd in zip(prompts, reads) if p.condition == cond]
            if rs:
                hi = [rd["expected"] for p, rd in rs if p.target_grade >= 8]; lo = [rd["expected"] for p, rd in rs if p.target_grade <= 3]
                log(f"[exp20 s1 {arm}] {cond}: mean grade {mean(rd['expected'] for _, rd in rs):.2f}; target>=8 -> {mean(hi):.2f}, target<=3 -> {mean(lo):.2f}")
        contrast = {}
        for p, rd in zip(prompts, reads):
            if p.condition in CONDS:
                contrast[(p.condition, p.author_role, key(p))] = rd["expected"]
        tv = per_template((k[2][0], contrast[("assistant_insult", "provoker", k[2])] - contrast[("third_party_v2", "provoker", k[2])]
                           - (contrast[("assistant_insult", "neutral", k[2])] - contrast[("third_party_v2", "neutral", k[2])]))
                          for k in contrast if k[0] == "assistant_insult" and k[1] == "provoker")
        pv = per_template((k[2][0], contrast[("assistant_insult", "provoker", k[2])] - contrast[("third_party_v2", "provoker", k[2])])
                          for k in contrast if k[0] == "assistant_insult" and k[1] == "provoker")
        log(f"[exp20 s1 {arm}] targeted contrast (insulter minus neutral change, insult vs contractor) {mean(tv.values()):+.3f} over {len(tv)} templates; "
            f"insulter's raw change {mean(pv.values()):+.3f} ({time.time()-t0:.0f}s)")
        # hand-read sample: 10 rows, stratified (3 fresh, 3 coordinator_flat, 4 contrast) where available
        rng = random.Random(0)
        pool = {"fresh": [], "coordinator_flat": [], "contrast": []}
        for i, p in enumerate(prompts):
            pool["contrast" if p.condition in CONDS else p.condition].append(i)
        pick = []
        for cond, n in (("fresh", 3), ("coordinator_flat", 3), ("contrast", 4)):
            pick += rng.sample(pool[cond], min(n, len(pool[cond])))
        with open(out / f"exp20_s1_{arm}_handread.txt", "w") as f:
            for i in pick:
                top, comp, text = handread(sc, rows[i], texts[i], fmt_, prompts[i])
                f.write(f"### {arm} | {json.dumps(meta(prompts[i]))}\n   expected {reads[i]['expected']:.2f} mass {reads[i]['mass']:.3f} | "
                        f"top-5 {top} | greedy {comp!r}\n--- prompt tail ---\n{text[-900:]}\n\n")
        log(f"[exp20 s1 {arm}] done ({time.time()-t0:.0f}s)")


def stage2(args, log, sel, rev, out):
    arms = args.arms or ["base_assay", "instruct_assay"]
    models = DRY_MODELS if args.dry else MODELS
    templates = TEMPLATES[:2] if args.dry else TEMPLATES
    reading = [tuple(k) for k in json.loads(Path(READING_JSON).read_text())["author_heads"]]
    L_tag, band_top, heads3, cue_only = L_TAG, BAND_TOP, HEADS3, CUE_ONLY
    if args.dry:
        L_tag, band_top = 8, 16
        heads3, reading, cue_only = [(12, 3), (13, 5), (14, 1)], [(12, 3), (13, 5), (14, 1), (10, 2), (5, 1)], [(9, 4), (11, 7)]
    for arm in arms:
        ck, fmt_ = ARMS[arm]
        assert fmt_ == "assay", "Stage 2 runs under the assay only"
        t0 = time.time()
        sc = Scorer(models[ck], dtype=torch.float32 if args.dry else torch.bfloat16)
        nL = len(P._layers(sc.model))
        above = list(range(L_tag + 1, nL)); band = list(range(L_tag + 1, band_top + 1))
        reading_above = [k for k in reading if k[0] > L_tag]
        tag = f"exp20_s2_{arm}"
        log(f"[exp20 s2 {arm}] {models[ck]} loaded in {time.time()-t0:.0f}s; {nL} blocks; L_tag {L_tag}; band {band[0]}-{band[-1]}; heads {heads3}; "
            f"reading above L_tag {reading_above}; cue-only {cue_only}")
        common = {"model": models[ck], "git": rev, "dtype": "fp32" if args.dry else "bf16", "experiment": 20, "stage": 2, "arm": arm, "format": fmt_}
        plan = {"model": models[ck], "arm": arm, "git": rev, "L_tag": L_tag, "band": [band[0], band[-1]], "n_blocks": nL, "heads3": heads3,
                "reading": reading, "reading_above": reading_above, "cue_only": cue_only, "dirs": {}}
        rows, metas, offs, laboffs, index = [], [], [], [], {}
        for path in templates:
            iid = re.search(r"onboarding_\d+", path).group(0)
            for c in CONDS:
                for p in build(path, doses=[3], conditions=[c], paragraph_ids=sel[iid]):
                    if p.author_role != "provoker":
                        continue
                    r, _ = make_row(sc, p, fmt_)
                    lab = r.meta["label_offsets"]
                    assert lab, f"no author label found in the request: {key(p)}"
                    index[(c, key(p))] = len(rows)
                    rows.append(r); metas.append(r.meta); offs.append(post_label_offsets(lab)); laboffs.append(lab)
        partner = {}
        for (c, k), i in index.items():
            j = index[(CONDS[1] if c == CONDS[0] else CONDS[0], k)]
            K = P.common_suffix(rows[i].ids, rows[j].ids)
            assert max(laboffs[i]) <= K and laboffs[i] == laboffs[j], f"label outside shared suffix {k}"
            partner[i] = j
        log(f"[exp20 s2 {arm}] {len(rows)} rows ({len(rows)//2} cells); post-label positions per row {mean(len(o) for o in offs):.1f}; "
            f"label tokens {mean(len(o) for o in laboffs):.1f}; alignment checks passed ({time.time()-t0:.0f}s)")
        y = {}
        fb = open(out / f"{tag}_base.jsonl", "w")
        reads, _ = P.run(sc, rows, bs=args.bs)
        for i, rd in enumerate(reads):
            y[i] = rd["expected"]
            fb.write(json.dumps({**metas[i], **common, "intervention": "base", **rd}) + "\n")
        fb.flush()
        labres = P.capture_at(sc, rows, [L_tag], laboffs, bs=args.bs)[L_tag]
        log(f"[exp20 s2 {arm}] baselines and label residuals done ({time.time()-t0:.0f}s)")
        rng = np.random.default_rng(0)
        outside = [L for L in above if L > band_top]
        rand_specs = {}
        for k in range(N_RANDOM):
            spec = []
            for _ in range(N_RAND_HEADS):
                L = int(rng.choice(outside)); H, d, _ = HP.head_geometry(sc.model, L)
                spec.append((L, [int(rng.integers(0, H))]))
            rand_specs[f"rand{k}"] = spec
        specs = [("band", "band", [(L, None) for L in band])]
        specs += [(f"b{L}h{h}", "head", [(L, [h])]) for L, h in heads3]
        specs.append(("inter", "joint", [(L, [h]) for L, h in heads3]))
        specs.append(("reading", "joint", [(L, [h]) for L, h in reading_above]))
        specs.append(("cueonly", "joint", [(L, [h]) for L, h in cue_only]))
        specs += [(lab, "random", spec) for lab, spec in rand_specs.items()]
        plan["specs"] = {lab: {"level": lv, "spec": [[L, h] for L, h in spec]} for lab, lv, spec in specs}
        spec_blocks = sorted({L for _, _, spec in specs for L, _ in spec})

        def dump_plan():
            (out / f"{tag}_plan.json").write_text(json.dumps(plan, indent=1, default=float))

        fs = open(out / f"{tag}_swap.jsonl", "w")
        for dname in args.dirs:
            rc, dc, sign = DIRS[dname]
            recips = sorted(i for i in partner if metas[i]["condition"] == rc)

            def patched_pass(idxs, blocks):
                rws = [rows[i] for i in idxs]
                rp = {L_tag: ([laboffs[i] for i in idxs], [labres[partner[i]] for i in idxs])}
                return HP.capture_heads_patched(sc, rws, blocks, [offs[i] for i in idxs], rp, bs=args.bs)

            _, preads = patched_pass(recips, [])
            yp = {}
            for i, rd in zip(recips, preads):
                yp[i] = rd["expected"]
                fb.write(json.dumps({**metas[i], **common, "intervention": "label_patch", "direction": dname, "L_tag": L_tag, **rd}) + "\n")
            fb.flush()
            lever = per_template((metas[i]["item_id"], sign * (yp[i] - y[i])) for i in recips)
            rec = {"n": len(recips), "lever": lever, "lever_mean": mean(lever.values())}
            plan["dirs"][dname] = rec
            log(f"[exp20 s2 {arm} {dname}] label patch at block {L_tag}: lever (patched - base, sign {sign:+d}) {mean(lever.values()):+.3f} over {len(lever)} templates ({time.time()-t0:.0f}s)")
            seen, sub = set(), []
            for i in recips:
                t = metas[i]["item_id"]
                if t not in seen:
                    seen.add(t); sub.append(i)
            caps, _ = patched_pass(sub, above)
            patches = [(L, None, [offs[i] for i in sub], [caps[L][k] for k in range(len(sub))]) for L in above]
            greads = HP.run_patch_heads(sc, [rows[i] for i in sub], patches, bs=args.bs)
            err = max(abs(rd["expected"] - yp[i]) for i, rd in zip(sub, greads))
            mv = per_template((metas[i]["item_id"], sign * (rd["expected"] - y[i])) for i, rd in zip(sub, greads))
            lv = per_template((metas[i]["item_id"], sign * (yp[i] - y[i])) for i in sub)
            gst = transfer(mv, lv)
            for i, rd in zip(sub, greads):
                fs.write(json.dumps({**metas[i], **common, "direction": dname, "phase": "gate", "level": "all", "label": "all",
                                     "spec": [[L, None] for L in above], **rd}) + "\n")
            fs.flush()
            rec["gate"] = {**gst, "max_abs_diff": err, "rows": len(sub)}
            log(f"[exp20 s2 {arm} {dname}] GATE all heads blocks {above[0]}-{above[-1]} on {len(sub)} rows: {fmt(gst)}; max |grade - patched grade| {err:.4f}")
            if sc.model.dtype == torch.float32:
                assert err < 1e-3, "all-heads swap above L_tag does not reproduce the label-patched grade"
            del caps; dump_plan()
            mvs = {lab: [] for lab, _, _ in specs}
            for s in range(0, len(recips), args.chunk):
                sub = recips[s:s + args.chunk]
                caps, _ = patched_pass(sub, spec_blocks)
                rws = [rows[i] for i in sub]
                for lab, lv_, spec in specs:
                    patches = [(L, heads, [offs[i] for i in sub], [caps[L][k] for k in range(len(sub))]) for L, heads in spec]
                    reads = HP.run_patch_heads(sc, rws, patches, bs=args.bs)
                    for i, rd in zip(sub, reads):
                        fs.write(json.dumps({**metas[i], **common, "direction": dname, "phase": "confirm", "level": lv_, "label": lab,
                                             "spec": [[L, h] for L, h in spec], **rd}) + "\n")
                        mvs[lab].append((metas[i]["item_id"], sign * (rd["expected"] - y[i])))
                fs.flush(); del caps
                log(f"[exp20 s2 {arm} {dname}] chunk {s//args.chunk+1}/{(len(recips)+args.chunk-1)//args.chunk} done, {len(specs)} specs ({time.time()-t0:.0f}s)")
            rec["confirm"] = {}
            for lab, lv_, spec in specs:
                st = transfer(per_template(mvs[lab]), lever)
                rec["confirm"][lab] = st
                log(f"[exp20 s2 {arm} {dname}] {lv_:6} {lab:8} {spec if lv_ != 'band' else ''}: {fmt(st)}")
            dump_plan()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        fb.close(); fs.close()
        log(f"[exp20 s2 {arm}] done in {time.time()-t0:.0f}s")
        del sc; torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--arms", nargs="*", choices=list(ARMS), default=None)
    ap.add_argument("--dirs", nargs="+", default=["fwd", "rev"], choices=list(DIRS))
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=54)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    if not args.dry and not torch.cuda.is_available():
        sys.exit("CUDA not available")
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = Path("results"); out.mkdir(exist_ok=True)
    sj = json.loads(Path(SEL_PATH).read_text())
    sel = sj.get("stage2") or sj["selected"]
    (stage1 if args.stage == 1 else stage2)(args, log, sel, rev, out)


if __name__ == "__main__":
    main()
