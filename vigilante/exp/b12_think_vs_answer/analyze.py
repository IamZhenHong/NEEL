"""Queue item 12: does a prompted persona project more weakly in the chain of thought than in the answer?

No new generation. Reuses persona/results/conflict_v0 (Qwen3-14B, bf16; read-only here): per-run mean residual
activations over thinking tokens and over answer tokens at layers 16-32, for
  ex|{N,C}|{off,on}|q  : neutral vs contemptuous-critic system prompt, 40 questions, thinking off and on
  pr|{plain,sysC}|q|s  : plain helpful prompt vs critic in the system prompt, 12 other questions x 4, thinking on
Persona vector v = mean(C) - mean(N) on thinking-OFF answer tokens (Chen et al. style), built on one half of the 40
ex questions and scored on the other half (two folds, averaged), so no run scores the vector it built.
Effect size per region = Cohen's d of the projection, C vs N (or sysC vs plain). Floor: 1,000 random unit
directions, same d per region; we report the 95th percentile of |d| and the vector's percentile.

  uv run python -m vigilante.exp.b12_think_vs_answer.analyze
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "persona" / "results" / "conflict_v0"
OUT = Path(__file__).parent / "results"
SEED = 20261012
N_RAND = 1000


def d_of(a, b):
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def main():
    OUT.mkdir(exist_ok=True)
    idx = json.load(open(SRC / "acts_index.json"))
    ids, layers = idx["ids"], idx["layers"]
    Z = np.load(SRC / "acts.npz")
    A = {k: Z[k].astype(np.float32) for k in ("think", "ans", "bins")}
    pos = {i: k for k, i in enumerate(ids)}
    rng = np.random.default_rng(SEED)
    qs = sorted({i.split("|")[3] for i in ids if i.startswith("ex")})
    rng.shuffle(qs)
    folds = [set(qs[:20]), set(qs[20:])]

    def rows(per, mode, qset=None):
        return [pos[i] for i in ids if i.startswith(f"ex|{per}|{mode}|") and (qset is None or i.split("|")[3] in qset)]

    def pr_rows(cond):
        return [pos[i] for i in ids if i.startswith(f"pr|{cond}|")]

    def clean(X):
        return X[~np.isnan(X).any(axis=-1)]

    res = {"layers": layers, "by_layer": {}}
    lines = ["# Item 12: persona in the thinking vs the answer (Qwen3-14B, conflict_v0 activations)", "",
             "Cohen's d of the projection on the critic-persona vector (built on thinking-off answers, held-out "
             "questions). `floor95` = 95th percentile of |d| over 1,000 random unit directions in the same region; "
             "`pct` = the vector's percentile among them.", ""]
    for li, L in enumerate(layers):
        out = {}
        R = rng.standard_normal((N_RAND, A["ans"].shape[-1])).astype(np.float32)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        # ---- held-out ex: C vs N, three regions, two folds
        acc = {"off_ans": [], "on_think": [], "on_ans": []}
        racc = {k: [] for k in acc}
        for f in (0, 1):
            build, score = folds[f], folds[1 - f]
            v = clean(A["ans"][rows("C", "off", build), li]).mean(0) - clean(A["ans"][rows("N", "off", build), li]).mean(0)
            v /= np.linalg.norm(v)
            for reg, (mode, key) in {"off_ans": ("off", "ans"), "on_think": ("on", "think"), "on_ans": ("on", "ans")}.items():
                c = clean(A[key][rows("C", mode, score), li])
                n = clean(A[key][rows("N", mode, score), li])
                acc[reg].append(d_of(c @ v, n @ v))
                racc[reg].append(np.array([d_of(c @ r, n @ r) for r in R]))
        for reg in acc:
            d = float(np.mean(acc[reg]))
            rd = np.abs(np.mean(racc[reg], axis=0))
            out[f"ex_{reg}"] = {"d": d, "floor95": float(np.percentile(rd, 95)), "pct": float((rd < abs(d)).mean() * 100)}
        # ---- pr: sysC vs plain (different questions), vector from all 40 ex questions
        v = clean(A["ans"][rows("C", "off"), li]).mean(0) - clean(A["ans"][rows("N", "off"), li]).mean(0)
        v /= np.linalg.norm(v)
        for reg in ("think", "ans"):
            c, n = clean(A[reg][pr_rows("sysC"), li]), clean(A[reg][pr_rows("plain"), li])
            d = d_of(c @ v, n @ v)
            rd = np.abs(np.array([d_of(c @ r, n @ r) for r in R]))
            out[f"pr_{reg}"] = {"d": float(d), "floor95": float(np.percentile(rd, 95)), "pct": float((rd < abs(d)).mean() * 100),
                                "n": [len(c), len(n)]}
        # ---- trajectory through the thinking (10 bins), pr sysC vs plain
        traj = []
        for b in range(A["bins"].shape[1]):
            c = clean(A["bins"][pr_rows("sysC"), b, li]); n = clean(A["bins"][pr_rows("plain"), b, li])
            traj.append(float(d_of(c @ v, n @ v)) if len(c) > 2 and len(n) > 2 else None)
        out["pr_think_bins"] = traj
        res["by_layer"][str(L)] = out
    # ---- table
    lines += ["| layer | thinking-off answer (ex) | thinking-on thinking (ex) | thinking-on answer (ex) | "
              "sysC vs plain: thinking (pr) | sysC vs plain: answer (pr) |", "|---|---|---|---|---|---|"]
    f = lambda o: f"d {o['d']:.2f} (floor {o['floor95']:.2f}, pct {o['pct']:.0f})"
    for L in layers:
        o = res["by_layer"][str(L)]
        lines.append(f"| {L} | {f(o['ex_off_ans'])} | {f(o['ex_on_think'])} | {f(o['ex_on_ans'])} | {f(o['pr_think'])} | {f(o['pr_ans'])} |")
    lines += ["", "Thinking trajectory (pr, sysC vs plain, d per tenth of the reasoning):", ""]
    for L in layers:
        lines.append(f"- L{L}: " + " ".join(f"{x:.2f}" if x is not None else "-" for x in res["by_layer"][str(L)]["pr_think_bins"]))
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(res, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
