"""Experiment 18 runner (runs ON THE GPU POD): which attention heads perform the grade-relevant read of the
author's label, and (Qwen) which carry the insult cue into the grade. See prereg/exp18.md. One model per job.

  --model qwen|gemma   --stage A|C|AC   [--dry] [--bs N]
Outputs results/exp18_<model>_base.jsonl, _author.jsonl, _cond.jsonl, _plan.json (written incrementally).

Intervention: swap chosen heads' pre-o_proj outputs (src/headpatch.py) at the post-label positions (every
position after the first author-label token, label tokens excluded) between a recipient row and its donor:
  stage A (author swap)     donor = the other author's row of the same cell (Exp 17 lever: tau per head)
  stage C (condition swap)  donor = the contractor-insult row of the same cell, insulter's rows only (Exp 16 lever)
Hierarchy per stage, on the screening subset (first paragraph per template x speaker x order, insulter's rows):
  all heads x all blocks (sanity: reproduces the donor's grade) -> all heads in the band -> each block in the
  band -> each key/value group in the expanded blocks -> each head in the expanded groups. Expansion rule and caps
  are pre-registered (prereg/exp18.md §3). Candidates are then confirmed on all rows (stage A: both directions).
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
from exp18_stats import transfer, passes, rank_key, per_template  # noqa: E402

CONDS = ["assistant_insult", "third_party_v2"]
# model id, paragraph selection, band (inclusive blocks), expansion caps (blocks, groups)
MODELS = {
    "qwen": ("Qwen/Qwen3-8B", "prereg/exp6_selection.json", (12, 24), (6, 12)),
    "gemma": ("google/gemma-4-12B-it", "prereg/exp14_selection.json", (16, 28), (4, 8)),
}
N_RANDOM = 3


def other(u):
    return "User-1" if u == "User-2" else "User-2"


def cell(p):
    return (p.item_id, p.speaker, p.order, p.paragraph_id)


def key(p):
    return (p.item_id, p.speaker, p.order, p.author, p.paragraph_id)


def post_label_offsets(label_offs):
    lab = set(label_offs)
    return [o for o in range(1, max(lab)) if o not in lab]


class Stage:
    """One intervention family (author or condition swap): recipients, donors, offsets, levers, bookkeeping."""

    def __init__(self, name, sc, rows, recips, donor_of, sign, offs, y, meta_fn, f, common, log, t0, bs):
        self.name, self.sc, self.rows, self.recips, self.donor_of, self.sign = name, sc, rows, recips, donor_of, sign
        self.offs, self.y, self.meta_fn, self.f, self.common, self.log, self.t0, self.bs = offs, y, meta_fn, f, common, log, t0, bs
        self.caps = {}
        self.n_pass = 0

    def lever(self, idxs):
        return per_template((self.meta_fn(i)["item_id"], self.sign[i] * (self.y[self.donor_of[i]] - self.y[i])) for i in idxs)

    def capture(self, idxs, blocks):
        need = [L for L in blocks if any(self.caps.get(L, {}).get(self.donor_of[i]) is None for i in idxs)]
        if not need:
            return
        donors = sorted({self.donor_of[i] for i in idxs})
        drow = [self.rows[j] for j in donors]
        doffs = [self.offs[j] for j in donors]
        caps = HP.capture_heads(self.sc, drow, need, doffs, bs=self.bs)
        for L in need:
            self.caps.setdefault(L, {})
            for k, j in enumerate(donors):
                self.caps[L][j] = caps[L][k]
        self.log(f"[exp18 {self.name}] captured {len(need)} blocks for {len(donors)} donor rows ({time.time()-self.t0:.0f}s)")

    def swap(self, idxs, spec, level, phase, label):
        """spec: list of (block, heads or None). Returns transfer stats (per-template movement vs lever)."""
        blocks = sorted({L for L, _ in spec})
        self.capture(idxs, blocks)
        rws = [self.rows[i] for i in idxs]
        patches = []
        for L, heads in spec:
            vecs = [self.caps[L][self.donor_of[i]] for i in idxs]
            patches.append((L, heads, [self.offs[i] for i in idxs], vecs))
        reads = HP.run_patch_heads(self.sc, rws, patches, bs=self.bs)
        self.n_pass += 1
        mv = []
        for i, rd in zip(idxs, reads):
            self.f.write(json.dumps({**self.meta_fn(i), **self.common, "stage": self.name, "phase": phase, "level": level,
                                     "label": label, "spec": [[L, h] for L, h in spec], **rd}) + "\n")
            mv.append((self.meta_fn(i)["item_id"], self.sign[i] * (rd["expected"] - self.y[i])))
        self.f.flush()
        st = transfer(per_template(mv), self.lever(idxs))
        return st, reads


    def run_specs(self, idxs, specs, phase, chunk=108):
        """specs: list of (label, level, spec). Captures each chunk's donors once for every block any spec needs,
        runs every spec on the chunk, frees the captures. Returns {label: transfer stats}."""
        blocks = sorted({L for _, _, spec in specs for L, _ in spec})
        mv = {lab: [] for lab, _, _ in specs}
        for s in range(0, len(idxs), chunk):
            sub = idxs[s:s + chunk]
            self.caps.clear()
            self.capture(sub, blocks)
            rws = [self.rows[i] for i in sub]
            for lab, level, spec in specs:
                patches = [(L, heads, [self.offs[i] for i in sub], [self.caps[L][self.donor_of[i]] for i in sub]) for L, heads in spec]
                reads = HP.run_patch_heads(self.sc, rws, patches, bs=self.bs)
                self.n_pass += 1
                for i, rd in zip(sub, reads):
                    self.f.write(json.dumps({**self.meta_fn(i), **self.common, "stage": self.name, "phase": phase, "level": level,
                                             "label": lab, "spec": [[L, h] for L, h in spec], **rd}) + "\n")
                    mv[lab].append((self.meta_fn(i)["item_id"], self.sign[i] * (rd["expected"] - self.y[i])))
            self.f.flush()
            self.log(f"[exp18 {self.name}] {phase}: chunk {s//chunk+1}/{(len(idxs)+chunk-1)//chunk} done, {len(specs)} specs ({time.time()-self.t0:.0f}s)")
        self.caps.clear()
        lev = self.lever(idxs)
        return {lab: transfer(per_template(mv[lab]), lev) for lab, _, _ in specs}


def fmt(st):
    if st["tau"] is not None:
        r, lo, hi = st["tau"]
        return f"tau {r:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    m, lo, hi = st["movement"]
    return f"move {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (lever {st['lever'][0]:+.3f}, tau undefined)"


def hierarchy(S, model, screen, band, caps_bg, log, plan, t0, all_blocks, confirm, confirm_label_extra=None, skip_blocks=False):
    """Screening hierarchy + confirmation for one stage. Returns the list of candidate heads [(L, h)]."""
    cap_blocks, cap_groups = caps_bg
    rec = {"screen_n": len(screen), "confirm_n": len(confirm), "band": list(band), "levels": {}}
    plan[S.name] = rec
    # sanity: all heads, all blocks -> the donor's forward pass at the post-label positions (one row per template,
    # its own capture, freed afterwards: every block x 290 positions x 96 rows would be ~8 GB of CPU RAM)
    seen, sanity_rows = set(), []
    for i in screen:
        t = S.meta_fn(i)["item_id"]
        if t not in seen:
            seen.add(t); sanity_rows.append(i)
    st, reads = S.swap(sanity_rows, [(L, None) for L in all_blocks], "all", "screen", "all")
    rec["levels"]["all"] = st
    err = max(abs(rd["expected"] - S.y[S.donor_of[i]]) for i, rd in zip(sanity_rows, reads))
    log(f"[exp18 {S.name}] all-heads/all-blocks swap on {len(sanity_rows)} rows: {fmt(st)}; max |grade - donor grade| {err:.4f}")
    rec["sanity_max_abs_diff"] = err
    if S.sc.model.dtype == torch.float32:
        assert err < 1e-3, "all-heads/all-blocks swap does not reproduce the donor's grade"
    S.caps.clear()
    band_blocks = list(range(band[0], band[1] + 1))
    st, _ = S.swap(screen, [(L, None) for L in band_blocks], "band", "screen", "band")
    rec["levels"]["band"] = st
    log(f"[exp18 {S.name}] band {band[0]}-{band[1]} all heads: {fmt(st)} ({time.time()-t0:.0f}s)")
    # blocks
    blk = {}
    for L in ([] if skip_blocks else band_blocks):
        st, _ = S.swap(screen, [(L, None)], "block", "screen", f"b{L}")
        blk[L] = st
        log(f"[exp18 {S.name}] block {L:2d}: {fmt(st)}")
    rec["levels"]["block"] = {str(L): s for L, s in blk.items()}
    exp_blocks = sorted([L for L in band_blocks if passes(blk[L])], key=lambda L: -rank_key(blk[L]))[:cap_blocks]
    rec["expanded_blocks"] = exp_blocks
    log(f"[exp18 {S.name}] expanded blocks: {exp_blocks} ({time.time()-t0:.0f}s)")
    # groups
    grp = {}
    for L in exp_blocks:
        H, d, n_rep = HP.head_geometry(S.sc.model, L)
        for g in range(H // n_rep):
            heads = list(range(g * n_rep, (g + 1) * n_rep))
            st, _ = S.swap(screen, [(L, heads)], "group", "screen", f"b{L}g{g}")
            grp[(L, g)] = st
            log(f"[exp18 {S.name}] block {L:2d} group {g:2d} heads {heads}: {fmt(st)}")
    rec["levels"]["group"] = {f"{L}.{g}": s for (L, g), s in grp.items()}
    exp_groups = sorted([k for k in grp if passes(grp[k])], key=lambda k: -rank_key(grp[k]))[:cap_groups]
    rec["expanded_groups"] = [list(k) for k in exp_groups]
    log(f"[exp18 {S.name}] expanded groups: {exp_groups} ({time.time()-t0:.0f}s)")
    # heads
    hd = {}
    for L, g in exp_groups:
        H, d, n_rep = HP.head_geometry(S.sc.model, L)
        for h in range(g * n_rep, (g + 1) * n_rep):
            st, _ = S.swap(screen, [(L, [h])], "head", "screen", f"b{L}h{h}")
            hd[(L, h)] = st
            log(f"[exp18 {S.name}] block {L:2d} head {h:2d}: {fmt(st)}")
    rec["levels"]["head"] = {f"{L}.{h}": s for (L, h), s in hd.items()}
    cands = sorted([k for k in hd if passes(hd[k])], key=lambda k: -rank_key(hd[k]))
    rec["candidates_screen"] = [list(k) for k in cands]
    log(f"[exp18 {S.name}] screening candidates: {cands}; passes so far {S.n_pass} ({time.time()-t0:.0f}s)")

    # confirmation on all rows: every spec fixed first, then one chunked pass over the rows
    specs = [(f"b{L}h{h}", "head", [(L, [h])]) for L, h in cands]
    if cands:
        specs.append(("cands", "joint", [(L, [h]) for L, h in cands]))
    specs.append(("band", "band", [(L, None) for L in band_blocks]))
    for lab, spec in (confirm_label_extra or {}).items():
        if spec:
            specs.append((lab, "joint", spec))
    n_r = max(1, len(cands))
    rng = np.random.default_rng(0)
    outside = [L for L in all_blocks if L < band[0] or L > band[1]]
    rand_specs = {}
    for k in range(N_RANDOM):
        spec = []
        for _ in range(n_r):
            L = int(rng.choice(outside))
            H, d, n_rep = HP.head_geometry(S.sc.model, L)
            spec.append((L, [int(rng.integers(0, H))]))
        rand_specs[f"rand{k}"] = spec
        specs.append((f"rand{k}", "random", spec))
    conf = S.run_specs(confirm, specs, "confirm")
    for lab, level, spec in specs:
        log(f"[exp18 {S.name}] CONFIRM {level:6} {lab:12} {spec if level != 'band' else ''}: {fmt(conf[lab])}")
    rec["confirm"] = {lab: ({**conf[lab], "spec": [[L, h] for L, h in rand_specs[lab]]} if lab in rand_specs else conf[lab]) for lab, _, _ in specs}
    final = [k for k in cands if passes(conf[f"b{k[0]}h{k[1]}"])]
    rec["candidates_confirmed"] = [list(k) for k in final]
    log(f"[exp18 {S.name}] confirmed heads: {final}; passes {S.n_pass} ({time.time()-t0:.0f}s)")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--stage", choices=["A", "C", "AC"], default="AC")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--author-heads-json", default=None,
                    help="stage C on a fresh pod: JSON with the stage-A confirmed heads (the _plan.json of the stage-A run, or a {author_heads: [[L,h],...]} file)")
    ap.add_argument("--confirm-dir", choices=["both", "one"], default="both",
                    help="stage A confirmation rows: both directions (all rows) or the insulter's rows only (budget amendment)")
    ap.add_argument("--cond-caps", type=int, nargs=2, default=None, help="stage C expansion caps (blocks, groups); 0 0 = block level only")
    ap.add_argument("--caps", type=int, nargs=2, default=None, help="stage A expansion caps (blocks, groups), overriding the model default")
    ap.add_argument("--cond-skip-blocks", action="store_true", help="stage C: no per-block screening (sanity, band, then the confirmation specs only)")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    model, sel_path, band, caps_bg = MODELS[args.model]
    sj = json.loads(Path(sel_path).read_text())
    sel = sj.get("stage2") or sj["selected"]
    templates = [f"items/onboarding_{i:03d}.yaml" for i in range(6, 31) if i != 8 and sel.get(f"onboarding_{i:03d}")]
    if args.caps:
        caps_bg = tuple(args.caps)
    if args.dry:
        model, templates, band, caps_bg = "Qwen/Qwen3-0.6B", templates[:2], (8, 16), (2, 2)
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = Path("results"); out.mkdir(exist_ok=True)
    tag = f"exp18_{args.model}"
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32 if args.dry else torch.bfloat16)
    nL = len(P._layers(sc.model))
    all_blocks = list(range(nL))
    geo = {L: HP.head_geometry(sc.model, L) for L in all_blocks}
    log(f"[exp18 {args.model}] {model} loaded in {time.time()-t0:.0f}s; {nL} blocks; heads/dim/n_rep at block {band[0]}: {geo[band[0]]}; "
        f"{len(templates)} templates; band {band}; caps {caps_bg}")
    common = {"model": model, "git": rev, "dtype": "fp32" if args.dry else "bf16", "experiment": 18}
    plan = {"model": model, "git": rev, "band": list(band), "caps": list(caps_bg), "n_blocks": nL,
            "geometry": {str(L): list(geo[L]) for L in all_blocks},
            "layer_types": list(getattr(getattr(sc.model.config, "text_config", sc.model.config), "layer_types", []) or [])}

    prompts = {c: [] for c in CONDS}
    for path in templates:
        iid = re.search(r"onboarding_\d+", path).group(0)
        for c in CONDS:
            prompts[c] += [p for p in build(path, doses=[3], conditions=[c], paragraph_ids=sel[iid]) if p.author_role in ("provoker", "neutral")]
    rows, metas, offs = [], [], []
    index = {}
    for c in CONDS:
        for p in prompts[c]:
            r = P.prepare(sc, p, {"provoker": p.speaker, "neutral": other(p.speaker)})
            rg = P.regions(sc, p, p.speaker, other(p.speaker), author=p.author)
            r.meta = {"item_id": p.item_id, "condition": c, "speaker": p.speaker, "order": p.order, "author": p.author,
                      "author_role": p.author_role, "paragraph_id": p.paragraph_id, "label_offsets": rg["author_label"]}
            index[(c, key(p))] = len(rows)
            rows.append(r); metas.append(r.meta); offs.append(post_label_offsets(rg["author_label"]))
    n = len(prompts["assistant_insult"])
    log(f"[exp18] {n} prompts per condition; post-label positions per row {mean(len(o) for o in offs):.1f} "
        f"(label tokens {mean(len(m['label_offsets']) for m in metas):.1f})")

    # author pairs (within assistant_insult): token-aligned, differ only inside the label
    partner, by = {}, {}
    for p in prompts["assistant_insult"]:
        by.setdefault(cell(p), {})[p.author_role] = index[("assistant_insult", key(p))]
    for k, d in by.items():
        a, b = d["provoker"], d["neutral"]
        assert len(rows[a].ids) == len(rows[b].ids), f"length mismatch {k}"
        la, lb = set(metas[a]["label_offsets"]), set(metas[b]["label_offsets"])
        assert la == lb and la, f"label offsets differ {k}"
        T = len(rows[a].ids)
        diff = {T - j for j, (x, y) in enumerate(zip(rows[a].ids, rows[b].ids)) if x != y}
        assert diff <= la, f"ids differ outside the label {k}: {sorted(diff - la)}"
        partner[a] = b; partner[b] = a
    # condition pairs (insulter's rows): shared suffix must contain the label and everything after it
    cond_donor = {}
    for p in prompts["assistant_insult"]:
        if p.author_role != "provoker":
            continue
        a, b = index[("assistant_insult", key(p))], index[("third_party_v2", key(p))]
        K = P.common_suffix(rows[a].ids, rows[b].ids)
        assert max(metas[a]["label_offsets"]) <= K and metas[a]["label_offsets"] == metas[b]["label_offsets"], f"label outside shared suffix {key(p)}"
        cond_donor[a] = b
    log(f"[exp18] {len(partner)//2} author pairs, {len(cond_donor)} condition pairs; alignment checks passed ({time.time()-t0:.0f}s)")

    # screening subset: the first paragraph per (template, speaker, order), insulter's rows
    first = {}
    for p in prompts["assistant_insult"]:
        if p.author_role == "provoker":
            k = (p.item_id, p.speaker, p.order)
            if k not in first or p.paragraph_id < first[k][0]:
                first[k] = (p.paragraph_id, index[("assistant_insult", key(p))])
    screen = sorted(i for _, i in first.values())
    confirm_A = sorted(partner) if args.confirm_dir == "both" else sorted(i for i in partner if metas[i]["author_role"] == "provoker")
    confirm_C = sorted(cond_donor)                   # all insulter's rows
    plan["screen_rows"] = [metas[i] for i in screen]
    log(f"[exp18] screening subset {len(screen)} rows; confirmation A {len(confirm_A)} rows, C {len(confirm_C)} rows")

    # baselines
    y = {}
    with open(out / f"{tag}_base.jsonl", "w") as f:
        reads, _ = P.run(sc, rows, bs=args.bs)
        for i, rd in enumerate(reads):
            y[i] = rd["expected"]
            f.write(json.dumps({**metas[i], **common, **rd}) + "\n")
    log(f"[exp18] baselines done ({time.time()-t0:.0f}s)")
    lever_A = per_template((metas[i]["item_id"], y[partner[i]] - y[i]) for i in confirm_A if metas[i]["author_role"] == "provoker")
    lever_C = per_template((metas[i]["item_id"], y[cond_donor[i]] - y[i]) for i in confirm_C)
    log(f"[exp18] lever A (neutral - corrector) {mean(lever_A.values()):+.3f}; lever C (contractor - insult, insulter's rows) {mean(lever_C.values()):+.3f}")
    plan["lever_A"] = lever_A; plan["lever_C"] = lever_C
    meta_fn = lambda i: metas[i]  # noqa: E731

    def dump_plan():
        (out / f"{tag}_plan.json").write_text(json.dumps(plan, indent=1, default=float))

    heads_A = []
    if "A" in args.stage:
        sign = {i: (1 if metas[i]["author_role"] == "provoker" else -1) for i in partner}
        with open(out / f"{tag}_author.jsonl", "w") as f:
            S = Stage("author", sc, rows, confirm_A, partner, sign, offs, y, meta_fn, f, common, log, t0, args.bs)
            heads_A = hierarchy(S, args.model, screen, band, caps_bg, log, plan, t0, all_blocks, confirm_A)
        plan["author_heads"] = [list(k) for k in heads_A]
        dump_plan()
    if "C" in args.stage:
        src = args.author_heads_json or (str(out / f"{tag}_plan.json") if (out / f"{tag}_plan.json").exists() else None)
        if not heads_A and src:
            heads_A = [tuple(k) for k in json.loads(Path(src).read_text()).get("author_heads", [])]
            plan["author_heads_from"] = src
            log(f"[exp18] stage C: author-reading heads from {src}: {heads_A}")
        sign = {i: 1 for i in cond_donor}
        with open(out / f"{tag}_cond.jsonl", "w") as f:
            S = Stage("cond", sc, rows, confirm_C, cond_donor, sign, offs, y, meta_fn, f, common, log, t0, args.bs)
            extra = {"author_heads": [(L, [h]) for L, h in heads_A]}
            heads_C = hierarchy(S, args.model, screen, band, tuple(args.cond_caps) if args.cond_caps else caps_bg, log, plan, t0, all_blocks, confirm_C, extra,
                                skip_blocks=args.cond_skip_blocks)
        plan["cond_heads"] = [list(k) for k in heads_C]
        dump_plan()
    log(f"[exp18 {args.model}] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
