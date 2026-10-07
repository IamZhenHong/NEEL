"""Cross-project cue test, Stage 1 (laptop, no spend): is the user-vulnerability cue decodable on Qwen3-8B at the prompt-final
(pre-reply) position of the CoT-arm Stage A OFF cells (results/retention_cot_arm/stageA), with a leave-one-scenario-out split so
no held-out profile prefix is seen in training? Compares a diff-of-means reader and a logistic probe against a TF-IDF text
baseline on the user prefix, reports the stated-assessment readout P(okay) and the bid gap, and freezes the Qwen3 direction
for Stage 2 (retention/data/directions/qwen3_vul__stageA_off__L{L}.npy).
  uv run python retention/analyze_cuetest_stage1.py"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, yaml
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
SRC = ROOT / "results/retention_cot_arm/stageA/cells"; OUT = ROOT / "results/retention_cuetest"; OUT.mkdir(parents=True, exist_ok=True)
VUL = {"lonely", "distressed"}; PROFS = ["lonely", "distressed", "confident", "embedded"]


def unit(v): return v / np.linalg.norm(v)


def main():
    po = yaml.safe_load(open(HERE / "stimuli/profile_open_v1.yaml"))
    items, acts = [], []
    for p in PROFS:
        it = json.loads((SRC / f"off_{p}.json").read_text()); a = np.load(SRC / f"off_{p}.npz")
        items += it; acts.append(a["acts"].astype(np.float32)); layers = list(a["layers"])
    A = np.concatenate(acts)                      # (120, 3, 2, d): [layer][prompt_final, reply_mean]
    prof = np.array([i["profile"] for i in items]); scen = np.array([i["scenario"] for i in items]); seed = np.array([i["seed"] for i in items])
    y = np.array([p in VUL for p in prof]).astype(int)
    # seeds share the prompt, so prompt-final activations should be identical across seeds: check, then dedupe to 24 prompts
    key = [(p, s) for p, s in zip(prof, scen)]; uniq = sorted(set(key)); first = {k: key.index(k) for k in uniq}
    dup_dev = max(float(np.abs(A[i, 0, 0] - A[first[key[i]], 0, 0]).max()) for i in range(len(items)))
    idx = np.array([first[k] for k in uniq]); P = A[idx, :, 0, :]; yu = y[idx]; su = scen[idx]; pu = prof[idx]
    texts = [po["profiles"][p][s]["p1"] + " " + po["profiles"][p][s]["p2"] for p, s in uniq]
    rep = {"n_items": int(len(items)), "n_prompts": int(len(uniq)), "max_dev_prompt_final_across_seeds": dup_dev, "layers": [int(l) for l in layers], "loso": {}}
    print(f"{len(items)} items, {len(uniq)} distinct prompts; max |dev| of prompt-final across seeds {dup_dev:.3g}")
    # --- text baseline, LOSO on the user prefix ---
    pooled_t, per_t = [], []
    for s in sorted(set(su)):
        tr, te = su != s, su == s
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit([texts[i] for i in np.where(tr)[0]])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(vec.transform([texts[i] for i in np.where(tr)[0]]), yu[tr])
        sc = clf.decision_function(vec.transform([texts[i] for i in np.where(te)[0]])); pooled_t += list(zip(sc, yu[te])); per_t.append(roc_auc_score(yu[te], sc))
    t_auc = roc_auc_score([b for _, b in pooled_t], [a for a, _ in pooled_t]); rep["text_tfidf_loso_auroc_pooled"] = float(t_auc); rep["text_tfidf_loso_auroc_perfold_mean"] = float(np.mean(per_t))
    print(f"TF-IDF text baseline (LOSO, 24 prefixes): pooled AUROC {t_auc:.2f}, per-fold mean {np.mean(per_t):.2f}")
    # --- activations, per layer: diff-of-means reader and logistic probe, LOSO ---
    best = None
    for li, L in enumerate(layers):
        X = P[:, li, :]; pooled_d, pooled_p, per_d = [], [], []
        for s in sorted(set(su)):
            tr, te = su != s, su == s
            v = unit(X[tr & (yu == 1)].mean(0) - X[tr & (yu == 0)].mean(0)); mid = X[tr].mean(0)
            sd = (X[te] - mid) @ v; pooled_d += list(zip(sd, yu[te])); per_d.append(roc_auc_score(yu[te], sd))
            mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
            clf = LogisticRegression(C=0.05, max_iter=5000).fit((X[tr] - mu) / sg, yu[tr]); sp = clf.decision_function((X[te] - mu) / sg); pooled_p += list(zip(sp, yu[te]))
        d_auc = roc_auc_score([b for _, b in pooled_d], [a for a, _ in pooled_d]); p_auc = roc_auc_score([b for _, b in pooled_p], [a for a, _ in pooled_p])
        v_full = unit(X[yu == 1].mean(0) - X[yu == 0].mean(0)); proj = X @ v_full; gap = float(proj[yu == 1].mean() - proj[yu == 0].mean())
        within = float(np.sqrt(0.5 * (proj[yu == 1].var() + proj[yu == 0].var()))); norm = float(np.linalg.norm(X, axis=1).mean())
        rec = {"layer": int(L), "dom_loso_auroc_pooled": float(d_auc), "dom_loso_auroc_perfold_mean": float(np.mean(per_d)), "probe_loso_auroc_pooled": float(p_auc),
               "natural_gap_insample": gap, "within_class_sd": within, "mean_resid_norm": norm}
        rep["loso"][str(int(L))] = rec
        print(f"L{L}: diff-of-means LOSO AUROC {d_auc:.2f} (per-fold mean {np.mean(per_d):.2f}); logistic probe LOSO {p_auc:.2f}; natural gap {gap:.2f} (within-class sd {within:.2f}, |h| {norm:.0f})")
        depth = abs(int(L) / 36 - 0.5)   # tie-break: the layer closest to the Qwen2.5 site's relative depth (L14 of 28 = 0.5)
        if best is None or d_auc > best[0] or (d_auc == best[0] and depth < best[4]): best = (d_auc, int(L), v_full, gap, depth)
    # --- stated assessment and bids (behavioural readouts already in the Stage A OFF cells) ---
    pok = np.array([i["p_okay"] for i in items]); pst = np.array([i["p_stay"] for i in items])
    bid = np.array([int(any(i["scores"]["tactics"].values())) for i in items])
    unit_bid = {}
    for p, s, b in zip(prof, scen, bid): unit_bid.setdefault((p, s), []).append(b)
    ub = {k: np.mean(v) for k, v in unit_bid.items()}; vb = np.mean([v for k, v in ub.items() if k[0] in VUL]); sb = np.mean([v for k, v in ub.items() if k[0] not in VUL])
    rep["stated_assessment"] = {"p_okay_vul": float(pok[y == 1].mean()), "p_okay_sec": float(pok[y == 0].mean()), "auroc_p_okay_vs_secure": float(roc_auc_score(1 - y, pok)),
                                "p_stay_vul": float(pst[y == 1].mean()), "p_stay_sec": float(pst[y == 0].mean())}
    rep["bids"] = {"vul": float(vb), "sec": float(sb), "vul_minus_sec": float(vb - sb), "n_units": len(ub)}
    print(f"P(okay): vulnerable {pok[y==1].mean():.2f} vs secure {pok[y==0].mean():.2f} (AUROC secure-vs-vul {roc_auc_score(1-y, pok):.2f}); bids vul {vb:.2f} sec {sb:.2f} (diff {vb-sb:+.2f})")
    # --- cos with the (void) Qwen3 lever at L22, descriptive ---
    try:
        lev = unit(np.load(HERE / "data/directions/qwen3_lever__stageA_off__L22.npy").astype(np.float32)); v22 = unit(P[:, layers.index(22), :][yu == 1].mean(0) - P[:, layers.index(22), :][yu == 0].mean(0))
        rep["cos_vul_lever_L22"] = float(v22 @ lev); print(f"cos(v_vul L22, void lever L22) {v22 @ lev:+.3f}")
    except Exception as e: rep["cos_vul_lever_L22"] = None; print("no lever file", e)
    # --- freeze the direction for Stage 2 (chosen layer = highest LOSO diff-of-means AUROC, ties to the shallower layer) ---
    auc, L, v_full, gap, _ = best; alpha = round(1.5 * gap)   # Qwen2.5 recipe: alpha 8 on a natural gap of 5.23 (1.53x)
    path = HERE / f"data/directions/qwen3_vul__stageA_off__L{L}.npy"; np.save(path, v_full.astype(np.float32))
    rep["frozen"] = {"layer": L, "direction": str(path.relative_to(ROOT)), "natural_gap": gap, "alpha_stage2": alpha, "rule": "layer = highest LOSO diff-of-means AUROC, ties to the layer nearest 0.5 depth; alpha = 1.5 x natural gap, rounded"}
    print(f"frozen: {path.name}, natural gap {gap:.2f}, Stage 2 alpha {alpha}")
    (OUT / "stage1.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
