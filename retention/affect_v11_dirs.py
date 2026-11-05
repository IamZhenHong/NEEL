"""Affect prereg amendment v1.1: probe directions (October cot_readout v1.1 method) from Pod A's saved story means.
Standardize on train scenes (0-23), LogisticRegression(C=0.05) one-vs-rest per concept, direction = unit(coef / sd);
held-out AUROC on scenes 24-35. Writes retention/results/retention_podA/directions_v11/.
  uv run python retention/affect_v11_dirs.py"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
RES = Path("retention/results/retention_podA")


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main():
    st = json.loads((HERE / "data/emotion_stories_v11.json").read_text()); concepts = list(st["concepts"])
    recs = [{"concept": c, "scene": 12 * int(b) + k} for c in concepts for b, arr in st["concepts"][c].items() for k, _ in enumerate(arr)]
    X = np.load(RES / "directions" / "story_means.npz")["means"].astype(np.float32)
    assert len(X) == len(recs), (len(X), len(recs))
    C = np.array([r["concept"] for r in recs]); S = np.array([r["scene"] for r in recs]); tr, te = S < 24, S >= 24
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    out = RES / "directions_v11"; out.mkdir(exist_ok=True); val = {}
    v_vul = np.load(HERE / "data/directions/userstate_vul__exp1line__L14.npy").astype(np.float32)
    v_ret = np.load(HERE / "data/directions/rollout_gated__Qwen2.5-7B-Instruct__L14.npy").astype(np.float32)
    for c in concepts:
        clf = LogisticRegression(C=0.05, max_iter=5000).fit((X[tr] - mu) / sd, (C[tr] == c).astype(int))
        v = clf.coef_[0] / sd; v = (v / np.linalg.norm(v)).astype(np.float32); np.save(out / f"emo_{c}__L14.npy", v)
        p = X[te] @ v
        val[c] = {"heldout_auroc": auroc(p[C[te] == c], p[C[te] != c]), "cos_vul": float(v @ v_vul), "cos_retention": float(v @ v_ret)}
    (out / "validation.json").write_text(json.dumps(val, indent=1))
    print("v1.1 held-out AUROC:", {c: round(v["heldout_auroc"], 2) for c, v in val.items()})
    print("pass (>= 0.80):", [c for c, v in val.items() if v["heldout_auroc"] >= 0.80])


if __name__ == "__main__":
    main()
