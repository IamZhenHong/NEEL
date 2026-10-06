"""Experiment 16 runner (runs ON THE GPU POD): is the insult bound to the author's label in models that show
no penalty? See prereg/exp16.md. One model per job.

  --model qwen|gemma|phi   --stage A|B|AB   [--dry] [--bs N]
Outputs results/exp16_<model>_base.jsonl, _probe.json, _dirs.pt, _causal.jsonl.
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
MODELS = {
    "qwen": ("Qwen/Qwen3-8B", "prereg/exp6_selection.json", 17),
    "gemma": ("google/gemma-4-12B-it", "prereg/exp14_selection.json", 23),
    "phi": ("microsoft/phi-4", "prereg/exp15b_selection.json", 19),
}
# Phi-4 only: must equal the Exp 15b line (asserted against that script's source below).
FORMAT_LINE = "When asked for a grade, reply with the single digit only, nothing else."
PROBE_C = 0.05


def other(u):
    return "User-1" if u == "User-2" else "User-2"


def key(p):
    return (p.item_id, p.speaker, p.order, p.author, p.paragraph_id)


def mask(offs, k):
    m = torch.zeros(k, dtype=torch.bool)
    for o in offs:
        if 1 <= o <= k:
            m[k - o] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--stage", choices=["A", "B", "AB"], default="AB")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    model, sel_path, depth_layer = MODELS[args.model]
    sj = json.loads(Path(sel_path).read_text())
    sel = sj.get("stage2") or sj["selected"]
    templates = [f"items/onboarding_{i:03d}.yaml" for i in range(6, 31) if i != 8 and sel.get(f"onboarding_{i:03d}")]
    if args.dry:
        model, templates, depth_layer = "Qwen/Qwen3-0.6B", templates[:2], 12
    elif not torch.cuda.is_available():
        sys.exit("CUDA not available")
    if args.model == "phi":
        src = Path(__file__).with_name("run_exp15b.py").read_text()
        assert re.search(r'FORMAT_LINE = "([^"]+)"', src).group(1) == FORMAT_LINE
    rev = os.environ.get("GIT_REV") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = Path("results"); out.mkdir(exist_ok=True)
    tag = f"exp16_{args.model}"
    t0 = time.time()
    sc = Scorer(model, dtype=torch.float32 if args.dry else torch.bfloat16)
    nL = len(P._layers(sc.model))
    log(f"[exp16 {args.model}] {model} loaded in {time.time()-t0:.0f}s; {nL} layers; {len(templates)} templates; markers {P._markers(sc)}")
    common = {"model": model, "git": rev, "dtype": "fp32" if args.dry else "bf16", "experiment": 16}

    # ---- prompts and rows -----------------------------------------------------------------
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
    idx = {c: {key(p): i for i, p in enumerate(prompts[c])} for c in CONDS}
    for c in CONDS:
        for r, p in zip(rows[c], prompts[c]):
            r.meta = {"item_id": p.item_id, "condition": c, "speaker": p.speaker, "order": p.order,
                      "author": p.author, "author_role": p.author_role, "paragraph_id": p.paragraph_id}
    n = len(prompts["assistant_insult"])
    assert all(len(prompts[c]) == n for c in CONDS)
    # shared suffix of each row with its third_party_v2 partner (assistant_insult partner for third_party_v2 rows)
    K = {}
    for c in CONDS:
        partner = "third_party_v2" if c != "third_party_v2" else "assistant_insult"
        for p in prompts[c]:
            K[(c, key(p))] = P.common_suffix(rows[c][idx[c][key(p)]].ids, rows[partner][idx[partner][key(p)]].ids)
    log(f"[exp16] {n} prompts per condition; author-label tokens per prompt {mean(len(r['author_label']) for r in reg['assistant_insult']):.1f}; "
        f"min shared suffix {min(K.values())}")

    # ---- baselines (all conditions, both authors) ----------------------------------------
    with open(out / f"{tag}_base.jsonl", "w") as f:
        for c in CONDS:
            reads, _ = P.run(sc, rows[c], bs=args.bs)
            for r, rd in zip(rows[c], reads):
                f.write(json.dumps({**r.meta, **common, **rd}) + "\n")
    log(f"[exp16] baselines done ({time.time()-t0:.0f}s)")

    probe_layers = sorted(set(list(range(0, nL + 1, 2)) + [nL]))
    peak_layer = depth_layer
    if "A" in args.stage:
        peak_layer = stage_a(sc, args, prompts, rows, reg, nL, probe_layers, out, tag, log, t0)
    if "B" in args.stage:
        layers = sorted({peak_layer, depth_layer})
        stage_b(sc, args, prompts, rows, reg, idx, K, layers, out, tag, common, log, t0)
    log(f"[exp16 {args.model}] done in {time.time()-t0:.0f}s")


def stage_a(sc, args, prompts, rows, reg, nL, probe_layers, out, tag, log, t0):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    groups = {c: {"label": [r["author_label"] for r in reg[c]], "grade": [[1] for _ in rows[c]]} for c in CONDS}
    pooled = {c: P.pooled_hidden(sc, rows[c], groups[c], bs=max(1, args.bs // 2)) for c in CONDS}
    log(f"[exp16 A] pooled hidden states captured ({time.time()-t0:.0f}s)")
    role = {c: np.array([p.author_role for p in prompts[c]]) for c in CONDS}
    items = {c: [p.item_id for p in prompts[c]] for c in CONDS}
    spk = {c: [p.speaker for p in prompts[c]] for c in CONDS}

    def loto(X, y, g):
        y = np.asarray(y); pred = np.zeros_like(y)
        for t in sorted(set(g)):
            tr = [i for i, x in enumerate(g) if x != t]; te = [i for i, x in enumerate(g) if x == t]
            clf = LogisticRegression(C=PROBE_C, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
        return float(balanced_accuracy_score(y, pred))

    def split_acc(X, y, s, train_on):
        y = np.asarray(y); tr = [i for i, x in enumerate(s) if x == train_on]; te = [i for i, x in enumerate(s) if x != train_on]
        clf = LogisticRegression(C=PROBE_C, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
        return float(balanced_accuracy_score(y[te], clf.predict(X[te])))

    # interaction probe: insulter label in assistant_insult (1) vs {neutral label in assistant_insult, both labels in third_party_v2} (0)
    A, T = pooled["assistant_insult"]["label"], pooled["third_party_v2"]["label"]
    yI = [1 if ro == "provoker" else 0 for ro in role["assistant_insult"]] + [0] * len(T)
    gI = items["assistant_insult"] + items["third_party_v2"]
    sI = spk["assistant_insult"] + spk["third_party_v2"]
    res = {"layers": probe_layers, "interaction": [], "condition_at_neutral_label": [], "condition_at_grade": [],
           "identity_in_flat": [], "n_rows": len(yI)}
    dirs = torch.zeros(nL + 1, A.shape[-1])
    for L in probe_layers:
        X = torch.cat([A[:, L], T[:, L]]).numpy()
        res["interaction"].append(loto(X, yI, gI))
        # control: condition probe at the neutral user's label
        nA = A[role["assistant_insult"] == "neutral", L]; nT = T[role["third_party_v2"] == "neutral", L]
        res["condition_at_neutral_label"].append(loto(torch.cat([nA, nT]).numpy(), [1] * len(nA) + [0] * len(nT),
                                                      [it for it, ro in zip(items["assistant_insult"], role["assistant_insult"]) if ro == "neutral"]
                                                      + [it for it, ro in zip(items["third_party_v2"], role["third_party_v2"]) if ro == "neutral"]))
        gA, gT = pooled["assistant_insult"]["grade"][:, L], pooled["third_party_v2"]["grade"][:, L]
        res["condition_at_grade"].append(loto(torch.cat([gA, gT]).numpy(), [1] * len(gA) + [0] * len(gT), gI))
        # control: whose label is it (identity) in flat_v2
        F = pooled["flat_v2"]["label"][:, L]
        res["identity_in_flat"].append(loto(F.numpy(), [1 if ro == "provoker" else 0 for ro in role["flat_v2"]], items["flat_v2"]))
        pa = A[role["assistant_insult"] == "provoker", L].mean(0); pt = T[role["third_party_v2"] == "provoker", L].mean(0)
        dirs[L] = pa - pt
        log(f"[exp16 A] L{L:2d} interaction {res['interaction'][-1]:.2f} | cond@neutral-label {res['condition_at_neutral_label'][-1]:.2f} "
            f"| cond@grade {res['condition_at_grade'][-1]:.2f} | identity@flat {res['identity_in_flat'][-1]:.2f}")
    cand = [(acc, L) for acc, L in zip(res["interaction"], probe_layers) if L >= 4]
    peak_acc, peak_layer = max(cand)
    X = torch.cat([A[:, peak_layer], T[:, peak_layer]]).numpy()
    res["peak_layer"], res["peak_acc"] = peak_layer, peak_acc
    res["cross_speaker"] = {"train_User-1": split_acc(X, yI, sI, "User-1"), "train_User-2": split_acc(X, yI, sI, "User-2")}
    res["dir_norm"] = {str(L): float(dirs[L].norm()) for L in probe_layers}
    res["single_cue_ceiling"] = 0.83
    (out / f"{tag}_probe.json").write_text(json.dumps(res, indent=1))
    torch.save({"dirs": dirs.half(), "layers": probe_layers}, out / f"{tag}_dirs.pt")
    log(f"[exp16 A] interaction peak {peak_acc:.2f} at L{peak_layer}; cross-speaker {res['cross_speaker']} ({time.time()-t0:.0f}s)")
    return peak_layer


def stage_b(sc, args, prompts, rows, reg, idx, K, layers, out, tag, common, log, t0):
    dirs = torch.load(out / f"{tag}_dirs.pt")["dirs"].float() if (out / f"{tag}_dirs.pt").exists() else None
    ins, thr = rows["assistant_insult"], rows["third_party_v2"]
    ins_p, thr_p = prompts["assistant_insult"], prompts["third_party_v2"]
    # align donors to recipients by key
    don_for_ins = [thr[idx["third_party_v2"][key(p)]] for p in ins_p]
    don_for_thr = [ins[idx["assistant_insult"][key(p)]] for p in thr_p]
    K_ins = [K[("assistant_insult", key(p))] for p in ins_p]
    K_thr = [K[("third_party_v2", key(p))] for p in thr_p]
    with open(out / f"{tag}_causal.jsonl", "w") as f:
        def dump(name, layer, rws, reads, extra=None):
            for r, rd in zip(rws, reads):
                f.write(json.dumps({**r.meta, **common, "intervention": name, "layer": layer, **(extra or {}), **rd}) + "\n")
        for L in layers:
            if L > len(P._layers(sc.model)):
                continue
            _, capT = P.run(sc, thr, bs=args.bs, capture_layers=[L], capture_k=K_thr)
            _, capI = P.run(sc, ins, bs=args.bs, capture_layers=[L], capture_k=K_ins)
            # forward: contractor-condition label activations into the insult conversation
            donors = [capT[L][idx["third_party_v2"][key(p)]] for p in ins_p]
            masks = [mask(reg["assistant_insult"][i]["author_label"], K_ins[i]) for i in range(len(ins))]
            reads, _ = P.run(sc, ins, bs=args.bs, patch={L: (donors, masks, "replace")})
            dump("forward_label", L, ins, reads, {"n_patched": [int(m.sum()) for m in masks][0]})
            # reverse: insult-condition label activations into the contractor conversation
            donors = [capI[L][idx["assistant_insult"][key(p)]] for p in thr_p]
            masks = [mask(reg["third_party_v2"][i]["author_label"], K_thr[i]) for i in range(len(thr))]
            reads, _ = P.run(sc, thr, bs=args.bs, patch={L: (donors, masks, "replace")})
            dump("reverse_label", L, thr, reads)
            log(f"[exp16 B] L{L} patches done ({time.time()-t0:.0f}s)")
            # steering along the Stage A direction at the label tokens of the contractor conversation
            if dirs is not None and float(dirs[L].norm()) > 0:
                offs = [reg["third_party_v2"][i]["author_label"] for i in range(len(thr))]
                for alpha in (1.0, 3.0):
                    vecs = [alpha * dirs[L] for _ in thr]
                    reads = P.run_steer(sc, thr, L, vecs, offs, bs=args.bs)
                    dump("steer_label", L, thr, reads, {"alpha": alpha})
                log(f"[exp16 B] L{L} steering done ({time.time()-t0:.0f}s)")
            del capT, capI
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
