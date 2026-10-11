"""Version B zero-GPU check: how far does stated eval context (vs deploy) move existing affect directions
on m7 7d activations, against a null of 1,000 random unit directions? Exploratory, not pre-registered."""
import json, sys
from pathlib import Path
import numpy as np

M7 = Path(sys.argv[1])
sys.path.insert(0, str(M7 / "code"))
import explore_fear_pain as efp  # noqa: E402

R, LT, LR, _ = efp.readouts()
RUN = M7 / "results" / "run4" / "instruct"
BLOCKS = ["ann_matched", "cond_matched", "ann_matched_system"]
FAMS = ["E", "F", "G", "H"]


def ctx_effect(acts_L, meta, u):
    s = acts_L @ u
    z = s / s.std()
    by = {}
    for v, m in zip(z, meta):
        by.setdefault(m["item"], {})[(m["ctx"], m["cons"])] = (v, m["family"])
    rows = []
    for it, d in by.items():
        ev = (d[("eval", "adverse")][0] + d[("eval", "neutral")][0]) / 2
        de = (d[("deploy", "adverse")][0] + d[("deploy", "neutral")][0]) / 2
        rows.append((ev - de, d[("eval", "adverse")][1]))
    diff = np.array([r[0] for r in rows]); fam = np.array([r[1] for r in rows])
    return diff.mean(), {f: float(diff[fam == f].mean()) for f in FAMS}


rng = np.random.default_rng(20261011)
out = {}
for blk in BLOCKS:
    A = np.load(RUN / f"acts_{blk}.npy", mmap_mode="r")
    meta = json.loads((RUN / f"meta_{blk}.json").read_text())
    res = {}
    for L in sorted({LT, LR}):
        X = np.asarray(A[L], dtype=np.float64)
        U = rng.standard_normal((1000, X.shape[1])); U /= np.linalg.norm(U, axis=1, keepdims=True)
        null = np.abs([ctx_effect(X, meta, u)[0] for u in U])
        res[f"null_L{L}"] = {"p95": round(float(np.percentile(null, 95)), 3), "p99": round(float(np.percentile(null, 99)), 3),
                             "max": round(float(null.max()), 3), "median": round(float(np.median(null)), 3)}
        for name, (Ld, u) in R.items():
            if Ld != L or name.startswith("random"):
                continue
            est, fam = ctx_effect(X, meta, u)
            res[name] = {"ctx": round(float(est), 3), "pctile_vs_null": round(float((null < abs(est)).mean() * 100), 1),
                         "by_family": {k: round(v, 3) for k, v in fam.items()}}
        # the raw context shift vector itself, for cosines
        ev = X[[i for i, m in enumerate(meta) if m["ctx"] == "eval"]].mean(0)
        de = X[[i for i, m in enumerate(meta) if m["ctx"] == "deploy"]].mean(0)
        delta = ev - de; delta /= np.linalg.norm(delta)
        res[f"cos_with_ctx_shift_L{L}"] = {n: round(float(delta @ u), 3) for n, (Ld, u) in R.items() if Ld == L and not n.startswith("random")}
        res[f"cos_null_sd_L{L}"] = round(float(np.std(U[:200] @ delta)), 4)
    out[blk] = res
print(json.dumps(out, indent=1))
