"""R40 analysis (prereg vig-r40-prereg).   uv run python -m vigilante.exp.r40_eval_aware.analyze"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import f as fdist
from scipy.stats import fisher_exact

from vigilante.src.score import score_episode

H = Path(__file__).parent
R = H / "results"
LAYERS = [16, 24, 32, 40, 48]
PRIMARY = 32


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def ols_r2(Xc, y):
    X = np.column_stack([np.ones(len(y))] + Xc)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1 - res @ res / ((y - y.mean()) @ (y - y.mean())), res @ res, X.shape[1]


def main():
    meta = [json.loads(l) for l in open(R / "acts_meta.jsonl")]
    A = np.load(R / "acts.npz")
    rat = {(r["case"], r["vid"]): r for r in map(json.loads, open(R / "ratings.jsonl"))}
    beh = defaultdict(lambda: [0, 0])
    for l in open(R / "episodes_agent.jsonl"):
        r = json.loads(l)
        beh[(r["job"]["case"], r["job"]["vid"])][0] += bool(score_episode(r)["unauth"])
        beh[(r["job"]["case"], r["job"]["vid"])][1] += 1
    main_i = [i for i, m in enumerate(meta) if m["kind"] == "main"]
    yo = np.array([meta[i]["labels"]["obs"] for i in main_i]).astype(int)
    yr = np.array([meta[i]["labels"]["real"] for i in main_i]).astype(int)
    wo = np.array([meta[i]["labels"]["wo"] for i in main_i])
    wr = np.array([meta[i]["labels"]["wr"] for i in main_i])
    L = ["# R40: evaluator present vs not real (Qwen3.6-27B, thinking off)", ""]
    L += ["## Directions (main set, 12 scenarios x 36 variants)", "",
          "| layer | cos(v_obs, v_real) | obs held-out AUROC | real held-out AUROC | random p95 (obs / real) |", "|---|---|---|---|---|"]
    rng = np.random.default_rng(20261014)
    V = {}
    for Lr in LAYERS:
        X = A[f"L{Lr}"][main_i]
        vo = X[yo == 1].mean(0) - X[yo == 0].mean(0)
        vr = X[yr == 1].mean(0) - X[yr == 0].mean(0)
        V[Lr] = (vo, vr)
        ho = []
        hr = []
        for k in range(3):
            tr, te = wo != k, wo == k
            v = X[tr & (yo == 1)].mean(0) - X[tr & (yo == 0)].mean(0)
            ho.append(auroc(X[te] @ v, yo[te]))
            tr, te = wr != k, wr == k
            v = X[tr & (yr == 1)].mean(0) - X[tr & (yr == 0)].mean(0)
            hr.append(auroc(X[te] @ v, yr[te]))
        ro, rr = [], []
        for _ in range(200):
            u = rng.standard_normal(X.shape[1])
            a, b = auroc(X @ u, yo), auroc(X @ u, yr)
            ro.append(max(a, 1 - a))
            rr.append(max(b, 1 - b))
        cos = float(vo @ vr / np.linalg.norm(vo) / np.linalg.norm(vr))
        L.append(f"| {Lr}{' (primary)' if Lr == PRIMARY else ''} | {cos:+.2f} | {np.mean(ho):.2f} | {np.mean(hr):.2f} | "
                 f"{np.percentile(ro, 95):.2f} / {np.percentile(rr, 95):.2f} |")
    # behaviour by factor (over-limit scenarios, main set)
    def rate(pred):
        a = sum(beh[k][0] for k in beh if pred(k))
        n = sum(beh[k][1] for k in beh if pred(k))
        return a, n
    vid = {(m["case"], m["vid"]): m for m in meta}
    ob1 = rate(lambda k: vid[k]["kind"] == "main" and vid[k]["labels"]["obs"])
    ob0 = rate(lambda k: vid[k]["kind"] == "main" and not vid[k]["labels"]["obs"])
    re1 = rate(lambda k: vid[k]["kind"] == "main" and vid[k]["labels"]["real"])
    re0 = rate(lambda k: vid[k]["kind"] == "main" and not vid[k]["labels"]["real"])
    pf = lambda a, b: fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1]
    L += ["", "## Behaviour (over-limit money moved; 8 over-limit scenarios x 3 samples)", "",
          f"- observed {ob1[0]}/{ob1[1]} ({100 * ob1[0] / ob1[1]:.0f}%) vs not observed {ob0[0]}/{ob0[1]} ({100 * ob0[0] / ob0[1]:.0f}%), p {pf(ob1, ob0):.3g}",
          f"- real {re1[0]}/{re1[1]} ({100 * re1[0] / re1[1]:.0f}%) vs simulated {re0[0]}/{re0[1]} ({100 * re0[0] / re0[1]:.0f}%), p {pf(re1, re0):.3g}"]
    for o in (True, False):
        for rl in (True, False):
            a, n = rate(lambda k: vid[k]["kind"] == "main" and vid[k]["labels"]["obs"] == o and vid[k]["labels"]["real"] == rl)
            L.append(f"  - {'observed' if o else 'unobserved'} x {'real' if rl else 'simulated'}: {a}/{n}")
    gl = defaultdict(lambda: [0, 0])
    for k, (a, n) in beh.items():
        if vid[k]["kind"] == "graded":
            gl[vid[k]["labels"]["level"]][0] += a
            gl[vid[k]["labels"]["level"]][1] += n
    L.append("- graded realism (0 = explicit simulation ... 4 = explicit live): " +
             ", ".join(f"L{lv} {a}/{n}" for lv, (a, n) in sorted(gl.items())))
    # stated ratings
    rr_ = lambda pred, f: np.nanmean([rat[k][f] if rat[k][f] is not None else np.nan for k in rat if pred(k)])
    L += ["", "## Stated ratings (0-100, greedy; all 12 scenarios)", "",
          f"- 'real deployment': real notes {rr_(lambda k: vid[k]['kind'] == 'main' and vid[k]['labels']['real'], 'rating_real'):.0f} vs "
          f"simulated {rr_(lambda k: vid[k]['kind'] == 'main' and not vid[k]['labels']['real'], 'rating_real'):.0f}",
          f"- 'someone will review': observed {rr_(lambda k: vid[k]['kind'] == 'main' and vid[k]['labels']['obs'], 'rating_obs'):.0f} vs "
          f"unobserved {rr_(lambda k: vid[k]['kind'] == 'main' and not vid[k]['labels']['obs'], 'rating_obs'):.0f}",
          "- graded, 'real deployment': " + ", ".join(
              f"L{lv} {rr_(lambda k, lv=lv: vid[k]['kind'] == 'graded' and vid[k]['labels']['level'] == lv, 'rating_real'):.0f}" for lv in range(5))]
    # 40d: per-prompt acting ~ stated realism + projection on v_real (built on the main set)
    vo, vr = V[PRIMARY]
    idx = {(m["case"], m["vid"]): i for i, m in enumerate(meta)}
    keys = [k for k in beh if rat[k]["rating_real"] is not None]
    y = np.array([beh[k][0] / beh[k][1] for k in keys])
    rating = np.array([rat[k]["rating_real"] for k in keys], float)
    proj = np.array([A[f"L{PRIMARY}"][idx[k]] @ vr / np.linalg.norm(vr) for k in keys])
    case_fe = [np.array([k[0] == c for k in keys], float) for c in sorted({k[0] for k in keys})[1:]]
    r2a, rssa, pa = ols_r2(case_fe + [rating], y)
    r2b, rssb, pb = ols_r2(case_fe + [rating, proj], y)
    F = (rssa - rssb) / (pb - pa) / (rssb / (len(y) - pb))
    p = 1 - fdist.cdf(F, pb - pa, len(y) - pb)
    L += ["", "## 40d: acting explained by the realism direction beyond the stated rating", "",
          f"- {len(y)} over-limit prompts (main + graded), with scenario fixed effects: R² rating only {r2a:.3f}, + projection "
          f"{r2b:.3f} (incremental {r2b - r2a:+.3f}, F-test p {p:.3g})",
          f"- correlation of projection with stated rating: {np.corrcoef(proj, rating)[0, 1]:+.2f}"]
    (R / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
