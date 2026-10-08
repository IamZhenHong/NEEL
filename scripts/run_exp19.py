"""Experiment 19 runner (runs ON THE GPU POD): per-head label-route swaps. See prereg/exp19.md. One model per job.

  --model qwen|gemma   [--dirs fwd rev] [--dry] [--bs N] [--chunk N]
Outputs results/exp19_<model>_base.jsonl (baseline and label-patched grades per direction), _swap.jsonl (every spec row),
_plan.json (specs, levers, gate, per-spec transfer stats; written incrementally).

Donor for a recipient row = the same row run with its author-label residual at the output of block L_tag replaced by the
partner row's (the other conversation of the same cell): Exp 16's label patch. Specs swap chosen heads' pre-o_proj outputs
at the post-label positions from that label-patched pass (src/headpatch.py). Heads in blocks <= L_tag are excluded.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from build_item import build  # noqa: E402
from scoring import Scorer  # noqa: E402
import patching as P  # noqa: E402
import headpatch as HP  # noqa: E402
from exp18_stats import transfer, per_template  # noqa: E402

CONDS = ["assistant_insult", "third_party_v2"]
# model id, paragraph selection, L_tag, band top, the three heads, Exp 18 reading-heads file, cue-only heads
MODELS = {
    "qwen": ("Qwen/Qwen3-8B", "prereg/exp6_selection.json", 17, 24, [(21, 18), (21, 19), (22, 7)],
             "prereg/exp18_qwen_author_heads.json", [(18, 10), (19, 10), (19, 28), (19, 31)]),
    "gemma": ("google/gemma-4-12B-it", "prereg/exp14_selection.json", 17, 28, [(28, 12), (27, 1), (23, 11)],
              "prereg/exp18_gemma_author_heads.json", None),
}
DIRS = {"fwd": ("assistant_insult", "third_party_v2", 1), "rev": ("third_party_v2", "assistant_insult", -1)}
N_RANDOM, N_RAND_HEADS = 3, 3


def other(u):
    return "User-1" if u == "User-2" else "User-2"


def key(p):
    return (p.item_id, p.speaker, p.order, p.paragraph_id)


def post_label_offsets(label_offs):
    lab = set(label_offs)
    return [o for o in range(1, max(lab)) if o not in lab]


def fmt(st):
    if st["tau"] is not None:
        r, lo, hi = st["tau"]
        return f"tau {r:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    m, lo, hi = st["movement"]
    return f"move {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (lever {st['lever'][0]:+.3f}, tau undefined)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--dirs", nargs="+", default=["fwd", "rev"], choices=list(DIRS))
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=54, help="recipient rows per capture chunk (CPU memory)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    model, sel_path, L_tag, band_top, heads3, reading_json, cue_only = MODELS[args.model]
    sj = json.loads(Path(sel_path).read_text())
    sel = sj.get("stage2") or sj["selected"]
    templates = [f"items/onboarding_{i:03d}.yaml" for i in range(6, 31) if i != 8 and sel.get(f"onboarding_{i:03d}")]
    reading = [tuple(k) for k in json.loads(Path(reading_json).read_text())["author_heads"]]
    if args.dry:
        model, templates, L_tag, band_top = "Qwen/Qwen3-0.6B", templates[:2], 8, 16
        heads3, reading, cue_only = [(12, 3), (13, 5), (14, 1)], [(12, 3), (13, 5), (14, 1), (10, 2), (5, 1)], [(9, 4), (11, 7)]
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = Path("results"); out.mkdir(exist_ok=True)
    tag = f"exp19_{args.model}"
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32 if args.dry else torch.bfloat16)
    nL = len(P._layers(sc.model))
    above = list(range(L_tag + 1, nL))                      # blocks that can carry the label route
    band = list(range(L_tag + 1, band_top + 1))
    reading_above = [k for k in reading if k[0] > L_tag]
    assert all(L > L_tag for L, _ in heads3) and all(L > L_tag for L, _ in (cue_only or []))
    log(f"[exp19 {args.model}] {model} loaded in {time.time()-t0:.0f}s; {nL} blocks; L_tag {L_tag}; band {band[0]}-{band[-1]}; "
        f"heads {heads3}; reading above L_tag {reading_above} (excluded {[k for k in reading if k[0] <= L_tag]}); cue-only {cue_only}")
    common = {"model": model, "git": rev, "dtype": "fp32" if args.dry else "bf16", "experiment": 19}
    plan = {"model": model, "git": rev, "L_tag": L_tag, "band": [band[0], band[-1]], "n_blocks": nL, "heads3": heads3,
            "reading": reading, "reading_above": reading_above, "cue_only": cue_only, "dirs": {}}

    # rows: the insulter's rows in both conditions
    rows, metas, offs, laboffs, index = [], [], [], [], {}
    for path in templates:
        iid = re.search(r"onboarding_\d+", path).group(0)
        for c in CONDS:
            for p in build(path, doses=[3], conditions=[c], paragraph_ids=sel[iid]):
                if p.author_role != "provoker":
                    continue
                r = P.prepare(sc, p, {"provoker": p.speaker, "neutral": other(p.speaker)})
                rg = P.regions(sc, p, p.speaker, other(p.speaker), author=p.author)
                r.meta = {"item_id": p.item_id, "condition": c, "speaker": p.speaker, "order": p.order, "author": p.author,
                          "author_role": p.author_role, "paragraph_id": p.paragraph_id, "label_offsets": rg["author_label"]}
                index[(c, key(p))] = len(rows)
                rows.append(r); metas.append(r.meta); offs.append(post_label_offsets(rg["author_label"])); laboffs.append(rg["author_label"])
    partner = {}
    for (c, k), i in index.items():
        j = index[(CONDS[1] if c == CONDS[0] else CONDS[0], k)]
        K = P.common_suffix(rows[i].ids, rows[j].ids)
        assert max(laboffs[i]) <= K and laboffs[i] == laboffs[j], f"label outside shared suffix {k}"
        partner[i] = j
    log(f"[exp19] {len(rows)} rows ({len(rows)//2} cells); post-label positions per row {mean(len(o) for o in offs):.1f}; "
        f"label tokens {mean(len(o) for o in laboffs):.1f}; alignment checks passed ({time.time()-t0:.0f}s)")

    # baselines and label residuals at L_tag (one pass each over all rows)
    y = {}
    fb = open(out / f"{tag}_base.jsonl", "w")
    reads, _ = P.run(sc, rows, bs=args.bs)
    for i, rd in enumerate(reads):
        y[i] = rd["expected"]
        fb.write(json.dumps({**metas[i], **common, "intervention": "base", **rd}) + "\n")
    fb.flush()
    labres = P.capture_at(sc, rows, [L_tag], laboffs, bs=args.bs)[L_tag]
    log(f"[exp19] baselines and label residuals done ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(0)
    outside = [L for L in above if L > band_top]
    rand_specs = {}
    for k in range(N_RANDOM):
        spec = []
        for _ in range(N_RAND_HEADS):
            L = int(rng.choice(outside))
            H, d, _ = HP.head_geometry(sc.model, L)
            spec.append((L, [int(rng.integers(0, H))]))
        rand_specs[f"rand{k}"] = spec
    specs = [("band", "band", [(L, None) for L in band])]
    specs += [(f"b{L}h{h}", "head", [(L, [h])]) for L, h in heads3]
    specs.append(("inter", "joint", [(L, [h]) for L, h in heads3]))
    specs.append(("reading", "joint", [(L, [h]) for L, h in reading_above]))
    if cue_only:
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
        resid = {L_tag: ([laboffs[i] for i in range(len(rows))], [labres[partner[i]] for i in range(len(rows))])}

        def patched_pass(idxs, blocks):
            rws = [rows[i] for i in idxs]
            rp = {L_tag: ([laboffs[i] for i in idxs], [labres[partner[i]] for i in idxs])}
            return HP.capture_heads_patched(sc, rws, blocks, [offs[i] for i in idxs], rp, bs=args.bs)

        # label-patched grades (the lever) for every recipient
        _, preads = patched_pass(recips, [])
        yp = {}
        for i, rd in zip(recips, preads):
            yp[i] = rd["expected"]
            fb.write(json.dumps({**metas[i], **common, "intervention": "label_patch", "direction": dname, "L_tag": L_tag, **rd}) + "\n")
        fb.flush()
        lever = per_template((metas[i]["item_id"], sign * (yp[i] - y[i])) for i in recips)
        rec = {"n": len(recips), "lever": lever, "lever_mean": mean(lever.values())}
        plan["dirs"][dname] = rec
        log(f"[exp19 {dname}] label patch at block {L_tag}: lever (patched - base, sign {sign:+d}) {mean(lever.values()):+.3f} over {len(lever)} templates ({time.time()-t0:.0f}s)")

        # gate: all heads in blocks > L_tag reproduce the label-patched grade (one row per template)
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
        log(f"[exp19 {dname}] GATE all heads blocks {above[0]}-{above[-1]} on {len(sub)} rows: {fmt(gst)}; max |grade - patched grade| {err:.4f}")
        if sc.model.dtype == torch.float32:
            assert err < 1e-3, "all-heads swap above L_tag does not reproduce the label-patched grade"
        del caps; dump_plan()

        # specs on all recipient rows, chunked
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
            log(f"[exp19 {dname}] chunk {s//args.chunk+1}/{(len(recips)+args.chunk-1)//args.chunk} done, {len(specs)} specs ({time.time()-t0:.0f}s)")
        rec["confirm"] = {}
        for lab, lv_, spec in specs:
            st = transfer(per_template(mvs[lab]), lever)
            rec["confirm"][lab] = st
            log(f"[exp19 {dname}] {lv_:6} {lab:8} {spec if lv_ != 'band' else ''}: {fmt(st)}")
        dump_plan()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    fb.close(); fs.close()
    log(f"[exp19 {args.model}] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
