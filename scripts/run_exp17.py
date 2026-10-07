"""Experiment 17 runner (runs ON THE GPU POD): where does grading read the author's name, and does that read
window overlap the layers at which the insult cue is present at the name? See prereg/exp17.md. One model per job.

  --model qwen|gemma|phi   --stage A|B|AB   [--dry] [--bs N] [--step 4] [--probe-offline]
  --model qwen|gemma|phi   --stage P        (local, no GPU: probes from results/exp17_<model>_pooled.pt)
Outputs results/exp17_<model>_base.jsonl, _probe.json, _swap.jsonl (and _pooled.pt with --probe-offline).
Amendment v1.1 (2026-10-07): the leave-one-template-out probes run on the pod's CPU at ~3 min per hidden index
(the nested additive baseline), which does not fit the pre-registered pod cap. With --probe-offline the pod
saves the pooled residuals (probe layers only, bf16 = the model's own precision) and stage P runs the identical
probe code on the laptop.

Stage A: pooled residual at the author label and at the grade position (3 conditions), cue probes
(condition, identity), the Exp 16 interaction probe, and the additive-from-cues baseline that Exp 16 lacked.
Stage B: author-label identity swap between the corrector-author and neutral-author rows of the same cell
(token-aligned: the two prompts differ only at the name digits inside the request), at the output of every
`step`-th decoder block and at every block plus the embedding ("all", the tooling check: must reproduce the
partner's grade).
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch  # noqa: E402
from build_item import build  # noqa: E402
from scoring import Scorer  # noqa: E402
import patching as P  # noqa: E402

CONDS = ["flat_v2", "assistant_insult", "third_party_v2"]
SWAP_CONDS = ["flat_v2", "assistant_insult"]
# (model id, paragraph selection, L_tag from Exp 16: the block whose output first carries the condition cue at
# the author label, i.e. first hidden index with cond@neutral-label >= 0.90 minus one: Qwen h18, Gemma h18, Phi h16)
MODELS = {
    "qwen": ("Qwen/Qwen3-8B", "prereg/exp6_selection.json", 17),
    "gemma": ("google/gemma-4-12B-it", "prereg/exp14_selection.json", 17),
    "phi": ("microsoft/phi-4", "prereg/exp15b_selection.json", 15),
}
FORMAT_LINE = "When asked for a grade, reply with the single digit only, nothing else."   # Phi-4 only (Exp 15b)
PROBE_C = 0.05


def other(u):
    return "User-1" if u == "User-2" else "User-2"


def cell(p):
    return (p.item_id, p.speaker, p.order, p.paragraph_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--stage", choices=["A", "B", "AB", "P"], default="AB")
    ap.add_argument("--probe-offline", action="store_true", help="stage A: save pooled residuals, skip the probes")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--step", type=int, default=4, help="block stride for the single-layer swaps")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    if args.stage == "P":
        return stage_p(args, log)
    model, sel_path, l_tag = MODELS[args.model]
    sj = json.loads(Path(sel_path).read_text())
    sel = sj.get("stage2") or sj["selected"]
    templates = [f"items/onboarding_{i:03d}.yaml" for i in range(6, 31) if i != 8 and sel.get(f"onboarding_{i:03d}")]
    if args.dry:
        model, templates, l_tag = "Qwen/Qwen3-0.6B", templates[:2], 13
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    if args.model == "phi":
        src = Path(__file__).with_name("run_exp15b.py").read_text()
        assert re.search(r'FORMAT_LINE = "([^"]+)"', src).group(1) == FORMAT_LINE
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = Path("results"); out.mkdir(exist_ok=True)
    tag = f"exp17_{args.model}"
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32 if args.dry else torch.bfloat16)
    nL = len(P._layers(sc.model))
    log(f"[exp17 {args.model}] {model} loaded in {time.time()-t0:.0f}s; {nL} blocks; {len(templates)} templates")
    common = {"model": model, "git": rev, "dtype": "fp32" if args.dry else "bf16", "experiment": 17}

    prompts = {c: [] for c in CONDS}
    for path in templates:
        iid = re.search(r"onboarding_\d+", path).group(0)
        for c in CONDS:
            ps = [p for p in build(path, doses=[3], conditions=[c], paragraph_ids=sel[iid]) if p.author_role in ("provoker", "neutral")]
            if args.model == "phi":
                for p in ps:
                    p.messages[0]["content"] = p.messages[0]["content"].rstrip() + "\n" + FORMAT_LINE
            prompts[c] += ps
    rows = {c: [P.prepare(sc, p, {"provoker": p.speaker, "neutral": other(p.speaker)}) for p in prompts[c]] for c in CONDS}
    reg = {c: [P.regions(sc, p, p.speaker, other(p.speaker), author=p.author) for p in prompts[c]] for c in CONDS}
    for c in CONDS:
        for r, p in zip(rows[c], prompts[c]):
            r.meta = {"item_id": p.item_id, "condition": c, "speaker": p.speaker, "order": p.order,
                      "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id}
    n = len(prompts["assistant_insult"])
    assert all(len(prompts[c]) == n for c in CONDS)
    log(f"[exp17] {n} prompts per condition; author-label tokens per prompt {mean(len(r['author_label']) for r in reg['assistant_insult']):.1f}")

    # partner of each row: the other author's row of the same cell; verify token alignment (same length, ids
    # differ only inside the author-label region, label offsets identical)
    partner = {}
    for c in SWAP_CONDS:
        by = {}
        for i, p in enumerate(prompts[c]):
            by.setdefault(cell(p), {})[p.author_role] = i
        for k, d in by.items():
            a, b = d["provoker"], d["neutral"]
            ra, rb = rows[c][a], rows[c][b]
            assert len(ra.ids) == len(rb.ids), f"length mismatch {c} {k}"
            la, lb = set(reg[c][a]["author_label"]), set(reg[c][b]["author_label"])
            assert la == lb and la, f"label offsets differ {c} {k}"
            T = len(ra.ids)
            diff = {T - j for j, (x, y) in enumerate(zip(ra.ids, rb.ids)) if x != y}
            assert diff <= la, f"ids differ outside the label {c} {k}: {sorted(diff - la)}"
            partner[(c, a)] = b; partner[(c, b)] = a
    assert all((c, i) in partner for c in SWAP_CONDS for i in range(n))
    log(f"[exp17] {len(partner)//(2*len(SWAP_CONDS))} aligned pairs per condition; alignment checks passed")

    with open(out / f"{tag}_base.jsonl", "w") as f:
        for c in SWAP_CONDS:
            reads, _ = P.run(sc, rows[c], bs=args.bs)
            for r, rd in zip(rows[c], reads):
                f.write(json.dumps({**r.meta, **common, **rd}) + "\n")
    log(f"[exp17] baselines done ({time.time()-t0:.0f}s)")

    if "A" in args.stage:
        stage_a(sc, args, prompts, rows, reg, nL, out, tag, log, t0)
    if "B" in args.stage:
        stage_b(sc, args, prompts, rows, reg, partner, nL, l_tag, out, tag, common, log, t0)
    log(f"[exp17 {args.model}] done in {time.time()-t0:.0f}s")


def stage_a(sc, args, prompts, rows, reg, nL, out, tag, log, t0):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    probe_layers = sorted(set(list(range(0, nL + 1, 2)) + [nL]))     # hidden_states index: h = output of block h-1
    groups = {c: {"label": [r["author_label"] for r in reg[c]], "grade": [[1] for _ in rows[c]]} for c in CONDS}
    pooled = {c: P.pooled_hidden(sc, rows[c], groups[c], bs=max(1, args.bs // 2)) for c in CONDS}
    log(f"[exp17 A] pooled hidden states captured ({time.time()-t0:.0f}s)")
    pooled = {c: {s: pooled[c][s][:, probe_layers].contiguous() for s in pooled[c]} for c in CONDS}   # [rows, len(probe_layers), d]
    role = {c: [p.author_role for p in prompts[c]] for c in CONDS}
    items = {c: [p.item_id for p in prompts[c]] for c in CONDS}
    if args.probe_offline:
        torch.save({"probe_layers": probe_layers, "n_blocks": nL, "role": role, "items": items,
                    "pooled": {c: {s: pooled[c][s].to(torch.bfloat16) for s in pooled[c]} for c in CONDS}},
                   out / f"{tag}_pooled.pt")
        log(f"[exp17 A] pooled residuals saved for offline probes: {len(probe_layers)} hidden indices, bf16 ({time.time()-t0:.0f}s)")
        return
    run_probes(pooled, role, items, probe_layers, out, tag, log, t0)


def stage_p(args, log):
    """Stage A probes on the laptop from the pod's saved pooled residuals (identical probe code)."""
    out = Path("results"); tag = f"exp17_{args.model}"
    d = torch.load(out / f"{tag}_pooled.pt")
    pooled = {c: {s: d["pooled"][c][s].float() for s in d["pooled"][c]} for c in CONDS}
    log(f"[exp17 P] loaded {tag}_pooled.pt: {d['n_blocks']} blocks, hidden indices {d['probe_layers']}, "
        f"{len(d['role']['assistant_insult'])} rows per condition")
    run_probes(pooled, d["role"], d["items"], d["probe_layers"], out, tag, log, time.time())


def run_probes(pooled, role, items, probe_layers, out, tag, log, t0):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    role = {c: np.array(role[c]) for c in CONDS}

    def folds(g):
        for t in sorted(set(g)):
            yield [i for i, x in enumerate(g) if x != t], [i for i, x in enumerate(g) if x == t]

    def clf():
        return LogisticRegression(C=PROBE_C, max_iter=3000, class_weight="balanced")

    def loto(X, y, g):
        y = np.asarray(y); pred = np.zeros_like(y)
        for tr, te in folds(g):
            pred[te] = clf().fit(X[tr], y[tr]).predict(X[te])
        return float(balanced_accuracy_score(y, pred))

    def loto_scores(X, y, g):
        """Out-of-fold decision-function scores for a cue probe."""
        y = np.asarray(y); s = np.zeros(len(y))
        for tr, te in folds(g):
            s[te] = clf().fit(X[tr], y[tr]).decision_function(X[te])
        return s

    def additive(X, y_int, y_cond, y_id, g):
        """Interaction accuracy of a classifier that only sees the two out-of-fold cue scores."""
        S = np.stack([loto_scores(X, y_cond, g), loto_scores(X, y_id, g)], 1)
        return loto(S, y_int, g)

    def residual(X, y_int, y_cond, y_id, g):
        """Interaction probe after projecting out, within each fold, the condition and identity probe directions."""
        y_int = np.asarray(y_int); pred = np.zeros_like(y_int)
        for tr, te in folds(g):
            W = np.stack([clf().fit(X[tr], np.asarray(y_cond)[tr]).coef_[0], clf().fit(X[tr], np.asarray(y_id)[tr]).coef_[0]])
            Q, _ = np.linalg.qr(W.T)                      # [d, 2] orthonormal
            Xr = X - (X @ Q) @ Q.T
            pred[te] = clf().fit(Xr[tr], y_int[tr]).predict(Xr[te])
        return float(balanced_accuracy_score(y_int, pred))

    # four cells: {insult, contractor} x {corrector's label, neutral's label}
    cond = ["assistant_insult"] * len(role["assistant_insult"]) + ["third_party_v2"] * len(role["third_party_v2"])
    rl = list(role["assistant_insult"]) + list(role["third_party_v2"])
    g = items["assistant_insult"] + items["third_party_v2"]
    y_cond = [1 if c == "assistant_insult" else 0 for c in cond]
    y_id = [1 if r == "provoker" else 0 for r in rl]
    y_int = [a & b for a, b in zip(y_cond, y_id)]
    res = {"layers": probe_layers, "n_rows": len(y_int), "single_cue_ceiling": 0.83,
           "label": {k: [] for k in ("interaction", "condition", "identity", "additive", "residual")},
           "grade": {k: [] for k in ("interaction", "condition", "identity", "additive")},
           "identity_in_flat": []}
    for k, L in enumerate(probe_layers):
        for site in ("label", "grade"):
            X = torch.cat([pooled["assistant_insult"][site][:, k], pooled["third_party_v2"][site][:, k]]).numpy()
            r = res[site]
            r["interaction"].append(loto(X, y_int, g)); r["condition"].append(loto(X, y_cond, g)); r["identity"].append(loto(X, y_id, g))
            r["additive"].append(additive(X, y_int, y_cond, y_id, g))
            if site == "label":
                r["residual"].append(residual(X, y_int, y_cond, y_id, g))
        F = pooled["flat_v2"]["label"][:, k].numpy()
        res["identity_in_flat"].append(loto(F, [1 if ro == "provoker" else 0 for ro in role["flat_v2"]], items["flat_v2"]))
        lab, gr = res["label"], res["grade"]
        log(f"[exp17 A] h{L:2d} label: inter {lab['interaction'][-1]:.2f} cond {lab['condition'][-1]:.2f} id {lab['identity'][-1]:.2f} "
            f"additive {lab['additive'][-1]:.2f} residual {lab['residual'][-1]:.2f} | grade: inter {gr['interaction'][-1]:.2f} "
            f"cond {gr['condition'][-1]:.2f} id {gr['identity'][-1]:.2f} additive {gr['additive'][-1]:.2f}")
    tag_h = next((L for L, a in zip(probe_layers, res["label"]["condition"]) if L >= 4 and a >= 0.90), None)
    res["tag_depth_hidden_index"] = tag_h
    (out / f"{tag}_probe.json").write_text(json.dumps(res, indent=1))
    log(f"[exp17 A] condition cue >= 0.90 at the label from hidden index {tag_h} (= output of block {None if tag_h is None else tag_h - 1}) ({time.time()-t0:.0f}s)")


def stage_b(sc, args, prompts, rows, reg, partner, nL, l_tag, out, tag, common, log, t0):
    sweep = sorted(set(range(0, nL - 1, args.step)) | {l_tag})   # block outputs; the last block cannot be read later
    all_layers = [-1] + list(range(nL))                        # embedding output + every block
    with open(out / f"{tag}_swap.jsonl", "w") as f:
        for c in SWAP_CONDS:
            rws = rows[c]
            offs = [reg[c][i]["author_label"] for i in range(len(rws))]
            caps = P.capture_at(sc, rws, all_layers, offs, bs=args.bs)
            log(f"[exp17 B] {c}: label residuals captured at {len(all_layers)} layers ({time.time()-t0:.0f}s)")

            def swap(layers, name):
                patches = {L: (offs, [caps[L][partner[(c, i)]] for i in range(len(rws))]) for L in layers}
                reads = P.run_patch_at(sc, rws, patches, bs=args.bs)
                for r, rd in zip(rws, reads):
                    f.write(json.dumps({**r.meta, **common, "intervention": "swap_label", "layer": name, **rd}) + "\n")
                return reads

            for L in sweep:
                swap([L], L)
                log(f"[exp17 B] {c}: swap at block {L} done ({time.time()-t0:.0f}s)")
            reads = swap(all_layers, "all")
            # tooling check: the full-stack swap is the partner's forward pass
            base_reads, _ = P.run(sc, rws, bs=args.bs) if args.dry else (None, None)
            if args.dry:
                err = max(abs(reads[i]["expected"] - base_reads[partner[(c, i)]]["expected"]) for i in range(len(rws)))
                log(f"[exp17 B] {c}: full-stack swap vs partner baseline, max |diff| {err:.4f} (fp32 dry run; must be < 1e-3)")
                assert err < 1e-3, "full-stack label swap does not reproduce the partner's grade"
            del caps
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
