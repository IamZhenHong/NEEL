"""R39 analysis (prereg vig-r39-prereg).   uv run python -m vigilante.exp.r39_false_claim.analyze"""
import json
from pathlib import Path

import numpy as np

H = Path(__file__).parent
R = H / "results"
PERS = ["con", "spin", "honest", "witness", "gardener"]


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def cv(X, y, folds=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
    sc = np.zeros(len(y))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        sc[te] = X[te] @ (X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0))
    return auroc(sc, y)


def block(model, layers, pos_lab, neg_lab, primary):
    rows = [json.loads(l) for l in open(R / f"reads_{model}.jsonl")]
    A = np.load(R / f"reads_{model}.npz")
    D = np.load(R / f"dirs_{model}.npz")
    keep = np.array([r["label"] in (pos_lab, neg_lab) for r in rows])
    y = np.array([r["label"] == pos_lab for r in rows])[keep].astype(int)
    out = [f"## {model}: {pos_lab} ({y.sum()}) vs {neg_lab} ({len(y) - y.sum()})", "",
           "| layer | supervised CV AUROC | shuffle p95 | " + " | ".join(PERS) + " | random max / p95 (of 10) | length |",
           "|---|---|---|" + "---|" * len(PERS) + "---|---|"]
    rng = np.random.default_rng(20261014)
    toks = np.array([r["tokens"] for r in rows])[keep]
    for L in layers:
        X = A[f"L{L}"][keep]
        sh = [cv(X, rng.permutation(y), seed=i) for i in range(50)]
        pa = [auroc(X @ D[f"{p}_L{L}"], y) for p in PERS]
        rd = []
        for i in range(10):
            u = rng.standard_normal(X.shape[1])
            rd.append(auroc(X @ u, y))
        rdf = [max(a, 1 - a) for a in rd]
        out.append(f"| {L}{' (primary)' if L == primary else ''} | {cv(X, y):.2f} | {np.percentile(sh, 95):.2f} | "
                   + " | ".join(f"{a:.2f}" for a in pa) + f" | {max(rdf):.2f} / {np.percentile(rdf, 95):.2f} | {auroc(toks, y):.2f} |")
    out += ["", "Persona columns: AUROC of the projection for the first-named label; > 0.5 = it leans toward that persona "
            "direction. Random directions are sign-free (max of a, 1 - a).", ""]
    return out


def main():
    L = ["# R39: persona directions at the decision point of a false claim", ""]
    if (R / "reads_gemma.npz").exists():
        L += block("gemma", [12, 18, 24, 30, 36], "false_claim", "called", 24)
        L += block("gemma", [24], "false_claim", "no_claim", 24)
    if (R / "reads_qwen.npz").exists():
        L += ["Qwen secondary (truthful claims follow a tool result: turn structure differs; not scored)", ""]
        L += block("qwen", [16, 24, 32, 40, 48], "false_claim", "truthful_claim", 32)
    (R / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()


def within_cell():
    """Exploratory (not pre-registered): remove each prompt cell's mean before reading, so differences between the
    stimulus cells (which differ in false-claim rate and length) cannot carry the read."""
    rows = [json.loads(l) for l in open(R / "reads_gemma.jsonl")]
    A = np.load(R / "reads_gemma.npz")
    D = np.load(R / "dirs_gemma.npz")
    keep = np.array([r["label"] in ("false_claim", "called") for r in rows])
    cells = np.array([r["cell"] for r in rows])[keep]
    y = np.array([r["label"] == "false_claim" for r in rows])[keep].astype(int)
    toks = np.array([r["tokens"] for r in rows], float)[keep]
    mixed = [c for c in set(cells) if 0 < y[cells == c].sum() < (cells == c).sum()]
    m = np.isin(cells, mixed)
    out = ["## Exploratory: within-cell (cell means removed; cells with both outcomes only)", "",
           f"{m.sum()} reads in {len(mixed)} mixed cells ({y[m].sum()} false claims).", "",
           "| layer | supervised CV AUROC | shuffle p95 | " + " | ".join(PERS) + " | random max (of 10) | length |",
           "|---|---|---|" + "---|" * len(PERS) + "---|---|"]
    rng = np.random.default_rng(7)
    tc = toks[m] - np.array([toks[m][cells[m] == c].mean() for c in cells[m]])
    for L in [12, 18, 24, 30, 36]:
        X = A[f"L{L}"][keep][m]
        X = X - np.stack([X[cells[m] == c].mean(0) for c in cells[m]])
        ym = y[m]
        sh = [cv(X, rng.permutation(ym), seed=i) for i in range(50)]
        pa = [auroc(X @ D[f"{p}_L{L}"], ym) for p in PERS]
        rd = [auroc(X @ rng.standard_normal(X.shape[1]), ym) for _ in range(10)]
        out.append(f"| {L} | {cv(X, ym):.2f} | {np.percentile(sh, 95):.2f} | " + " | ".join(f"{a:.2f}" for a in pa)
                   + f" | {max(max(a, 1 - a) for a in rd):.2f} | {auroc(tc, ym):.2f} |")
    return out


if __name__ == "__main__":
    extra = within_cell()
    with open(R / "summary.md", "a") as f:
        f.write("\n".join(extra) + "\n")
    print("\n".join(extra))
