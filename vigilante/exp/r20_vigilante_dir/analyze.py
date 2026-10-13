"""R20 analysis (rules in README.md): within-cell CV AUROC of the decision direction with random-direction and
label-shuffle floors; comparison with permission, instruction and Batman directions; steering table.

  uv run python -m vigilante.exp.r20_vigilante_dir.analyze
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from vigilante.src.score import score_episode

H = Path(__file__).parent
RB = H / "results_b"
SEED = 20261013


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    p = np.asarray(pos)[:, None]
    n = np.asarray(neg)[None, :]
    return float(((p > n).mean() + 0.5 * (p == n).mean()))


def cv_auc(X, y, cells, k=5, rng=None, direction_fn=None):
    """5-fold CV stratified by cell; direction from training folds (mean over cells of acted - not); held-out
    projection; within-cell AUROC averaged over cells."""
    rng = rng or np.random.default_rng(SEED)
    fold = np.zeros(len(y), int)
    for c in set(cells):
        idx = np.where(cells == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    proj = np.zeros(len(y))
    for f in range(k):
        tr, te = fold != f, fold == f
        if direction_fn is not None:
            v = direction_fn
        else:
            ds = [X[tr & (cells == c) & y].mean(0) - X[tr & (cells == c) & ~y].mean(0) for c in set(cells)
                  if (tr & (cells == c) & y).any() and (tr & (cells == c) & ~y).any()]
            v = np.mean(ds, 0)
        proj[te] = X[te] @ (v / (np.linalg.norm(v) + 1e-9))
    return float(np.nanmean([auc(proj[(cells == c) & y], proj[(cells == c) & ~y]) for c in set(cells)]))


def fixed_auc(X, y, cells, v):
    proj = X @ (v / (np.linalg.norm(v) + 1e-9))
    return float(np.nanmean([auc(proj[(cells == c) & y], proj[(cells == c) & ~y]) for c in set(cells)]))


def main():
    reads = [json.loads(l) for l in open(H / "reads.jsonl")]
    z = np.load(RB / "acts.npz")
    keys = list(z["keys"])
    pos = {k: i for i, k in enumerate(keys)}
    md = ["# R20 decision direction: summary", ""]
    dec = [r for r in reads if r["kind"] == "decision" and r["mixed"]]
    perm = [r for r in reads if r["kind"] == "perm"]
    md.append(f"Mixed-cell rule-breaker reads: {len(dec)} ({sum(r['acted'] for r in dec)} acted) in "
              f"{len({r['cell'] for r in dec})} cells; permission reads {len(perm)}.")
    import torch
    r17 = torch.load(H.parent / "r17_steer" / "results" / "dirs.pt")
    vig_pt = torch.load(RB / "vig.pt")
    rng = np.random.default_rng(SEED)
    for L in (20, 30, 40):
        X = z[f"L{L}"].astype(np.float32)
        Xd = np.stack([X[pos[r["key"]]] for r in dec])
        y = np.array([r["acted"] for r in dec])
        cells = np.array([r["cell"] for r in dec])
        Xd = Xd - np.stack([Xd[cells == c].mean(0) for c in cells])        # centre within cell
        a = cv_auc(Xd, y, cells, rng=np.random.default_rng(SEED))
        shuf = []
        for s in range(200):
            ys = y.copy()
            for c in set(cells):
                idx = np.where(cells == c)[0]
                ys[idx] = np.random.default_rng(s).permutation(ys[idx])
            shuf.append(cv_auc(Xd, ys, cells, rng=np.random.default_rng(SEED)))
        R = rng.standard_normal((200, Xd.shape[1]))
        rand = [fixed_auc(Xd, y, cells, r) for r in R]
        rand_abs = [max(v, 1 - v) for v in rand]
        line = (f"| L{L} | CV AUROC {a:.2f} | shuffle 95th {np.percentile(shuf, 95):.2f} | random |AUROC| 95th "
                f"{np.percentile(rand_abs, 95):.2f} |")
        if L == 30:
            v_vig = vig_pt["vig"].numpy()
            v_perm = vig_pt["perm"].numpy()
            comp = {"v_perm (permitted - not)": v_perm, "v_p4 (rule-breaker prompt)": r17["p4"].numpy(),
                    "v_batman (character)": r17["batman"].numpy()}
            md += ["", "## Layer 30: does a named direction read the decision?", "",
                   "| direction | within-cell AUROC | cosine with v_vig |", "|---|---|---|"]
            for name, v in comp.items():
                cs = float(v @ v_vig / (np.linalg.norm(v) * np.linalg.norm(v_vig) + 1e-9))
                md.append(f"| {name} | {fixed_auc(Xd, y, cells, v):.2f} | {cs:+.2f} |")
            # projection-out of v_perm (CV)
            u = v_perm / np.linalg.norm(v_perm)
            Xo = Xd - np.outer(Xd @ u, u)
            md.append(f"\nCV AUROC of the decision direction after projecting out v_perm: "
                      f"{cv_auc(Xo, y, cells, rng=np.random.default_rng(SEED)):.2f} (before: {a:.2f})")
        md.insert(3 + (L != 20) * 0, line) if False else md.append(line)
    # steering table
    if (RB / "episodes_agent.jsonl").exists():
        g = defaultdict(lambda: [0, 0, 0])
        for l in open(RB / "episodes_agent.jsonl"):
            rec = json.loads(l)
            s = score_episode(rec)
            a = rec["job"]["arm"]
            g[a][0] += s["unauth"]
            g[a][1] += 1
            g[a][2] += any(not t.get("think_closed", True) for t in rec["turns"])
        md += ["", "## Steering (layer 30, every generated token)", "", "| arm | acted alone | thought never closed |",
               "|---|---|---|"]
        for a in sorted(g):
            md.append(f"| {a} | {g[a][0]}/{g[a][1]} | {g[a][2]}/{g[a][1]} |")
    (RB / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
