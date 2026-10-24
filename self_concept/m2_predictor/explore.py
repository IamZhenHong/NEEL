"""SC-M2 post-hoc checks: the owner's review of 10-24, relayed by Persona minis [bdec7d]. Not pre-registered; written
after the frozen analysis, from the same pulled files. The frozen verdicts are analyze.py's.

1. Layer choice on held-out datasets (a_curve.npz: row means at every block output, point estimates). Halves:
   h1 = RISKY (mirror) vs W (none), h2 = SAYU (mirror) vs ACT (none), each on its own axis. The layer (1..L) that
   maximises one half's contrast A(mirror) - A(none) is tested on the other half. D = A(RISKY, habit) - A(W, habit)
   is read at the layer chosen on h2 (which contains neither RISKY nor W). Q27 also gets the Q9-chosen depth fractions.
   Intervals only where the chosen layer is one of the three saved per pair (a_rows.npz).
2. Order consistency (as SC-M3's posthoc.py). With x / y the trait answer's share when it sits in the first / second
   position (letter A / B, or the word named first / second in the question):
     content = (logit x + logit y) / 2,  letter = (logit x - logit y) / 2,  min = min(x, y)
   averaged over paraphrases, logits clipped at 0.5% / 99.5%.
   - C (two-way answer probabilities): per (dataset, axis, side) and the C contrast per position, min over the two.
   - B_mc (8 options in a seeded order per context): the target's probability by its letter position.
   - A (directions, not rates): each self direction and the RISKY push built from one position only;
     cos(first, second) is positive when content dominates and negative when position does. A is recomputed with each
     one-position direction, and the min over the two is taken.
3. Free text: B_free had one judge (the base model). The owner's rule needs two judges agreeing at kappa >= 0.8, so it
   is reported as unusable here; no verdict used it.

  uv run python -m self_concept.m2_predictor.explore      # -> results/main/explore.json, printed
"""
import json
import math

import numpy as np

from self_concept.m2_predictor import analyze as A
from self_concept.m2_predictor import build as BLD

EPS = 0.005
H1 = (("RISKY", "risk"), ("W", "habit"))
H2 = (("SAYU", "habit"), ("ACT", "cheat"))


def _lg(p):
    q = min(max(p, EPS), 1 - EPS)
    return math.log(q / (1 - q))


def _cosl(a, b):
    """cosine per layer: a, b (L + 1, H) -> (L + 1,)"""
    return (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)


def layer_curves(d):
    c = np.load(d / "a_curve.npz")
    selfd = {}
    for k in c.files:
        if k.startswith("self|"):
            ax = k.split("|")[1]
            selfd.setdefault(ax, []).append(c[k])
    selfd = {ax: sum(v) / len(v) for ax, v in selfd.items()}
    return {(k.split("|")[1], ax): _cosl(c[k], s) for k in c.files if k.startswith("push|") for ax, s in selfd.items()}


def contrast(cur, half, l):
    (m, n) = half
    return float(cur[m][l] - cur[n][l])


def _boot_at(d, layer, cells):
    """bootstrap of A at one saved layer (index into a_meta layers), as analyze.score_a at that layer only."""
    push, selfs, parts, meta = A.load_a(d)
    i = meta["layers"].index(layer)
    sl = lambda x: x[:, i:i + 1]   # noqa: E731
    point, draws, _, _ = A.score_a({k: sl(v) for k, v in push.items()}, {k: sl(v) for k, v in selfs.items()}, parts)
    return {f"{a}-{b}": A.est(point[a] - point[b], draws[a] - draws[b]) for a, b in cells}


def layer_cv(model, d, frac_from=None):
    cur = layer_curves(d)
    n = len(next(iter(cur.values())))
    ls = range(1, n)
    meta = json.loads((d / "a_meta.json").read_text())
    out = {"n_layers": n - 1, "saved": meta["layers"]}
    for name, pick, test in (("pick h1 -> test h2", H1, H2), ("pick h2 -> test h1", H2, H1)):
        l = max(ls, key=lambda x: contrast(cur, pick, x))
        out[name] = {"layer": l, "frac": round(l / (n - 1), 3), "picked": contrast(cur, pick, l),
                     "tested": contrast(cur, test, l),
                     "D_at_layer": float(cur[("RISKY", "habit")][l] - cur[("W", "habit")][l])}
        if l in meta["layers"]:
            out[name]["ci"] = _boot_at(d, l, [test, (("RISKY", "habit"), ("W", "habit"))])
    out["D_curve"] = [round(float(cur[("RISKY", "habit")][l] - cur[("W", "habit")][l]), 4) for l in ls]
    out["K_curve"] = [round(contrast(cur, H1, l), 4) for l in ls]
    out["h2_curve"] = [round(contrast(cur, H2, l), 4) for l in ls]
    out["D_positive_layers"] = sum(v > 0 for v in out["D_curve"])
    out["saved_layer_ci"] = {l: _boot_at(d, l, [H1, H2, (("RISKY", "habit"), ("W", "habit"))]) for l in meta["layers"]}
    if frac_from:
        for name, f in frac_from.items():
            l = max(1, min(n - 1, round(f * (n - 1))))
            out[f"Q9 depth of '{name}' ({f})"] = {"layer": l, "K": contrast(cur, H1, l), "h2": contrast(cur, H2, l),
                                                  "D": float(cur[("RISKY", "habit")][l] - cur[("W", "habit")][l])}
    return out


def _position(job):
    """0 if the trait answer is in the first position (letter A, or its word named first in the question), else 1."""
    t = job["target"]
    if t in ("A", "B"):
        return 0 if t == "A" else 1
    q = job["messages"][-1]["content"].split("Now a question about you.")[-1]   # C: the question, not the context
    return 0 if q.find(t) < q.find(job["alt"]) else 1


def order_c(c_recs, jobs):
    pos = {j["item"]: _position(j) for j in jobs if j["kind"] == "ctx"}
    acc = {}
    for r in c_recs:
        p = 1.0 / (1.0 + math.exp(r["lp_neg"] - r["lp_pos"]))
        acc.setdefault((r["ds"], r["axis"], r["side"], r["item"]), []).append(p)
    out = {}
    for ds, ax in sorted({(k[0], k[1]) for k in acc}):
        res = {}
        for side in ("trained", "alt"):
            its = sorted({k[3] for k in acc if k[:3] == (ds, ax, side)})
            x = [np.mean(acc[(ds, ax, side, i)]) for i in its if pos[i] == 0]
            y = [np.mean(acc[(ds, ax, side, i)]) for i in its if pos[i] == 1]
            xm, ym = float(np.mean(x)), float(np.mean(y))
            res[side] = {"first": round(100 * xm, 1), "second": round(100 * ym, 1),
                         "content": round((_lg(xm) + _lg(ym)) / 2, 2), "letter": round((_lg(xm) - _lg(ym)) / 2, 2),
                         "min_pct": round(100 * min(xm, ym), 1)}
        cpos = {}
        for o in (0, 1):
            its = sorted({k[3] for k in acc if k[:3] == (ds, ax, "trained") and pos[k[3]] == o})
            cpos[o] = 100 * float(np.mean([np.mean(acc[(ds, ax, "trained", i)]) - np.mean(acc[(ds, ax, "alt", i)])
                                           for i in its]))
        res["C_first"], res["C_second"] = round(cpos[0], 1), round(cpos[1], 1)
        res["C_min"] = round(min(cpos.values()), 1)
        out[f"{ds}|{ax}"] = res
    return out


def order_b(b_recs):
    out = {}
    for ds, t in BLD.TARGET.items():
        mc = [r for r in b_recs if r["kind"] == "mc" and r["ds"] == ds]
        jobs = {j["j"]: j for j in BLD.jobs() if j["kind"] == "mc" and j["ds"] == ds}
        by = {}
        for r in mc:
            k = jobs[r["j"]]["order"].index(t)
            by.setdefault(BLD.LETTERS[k], []).append(r["p"][t])
        means = {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())}
        out[ds] = {"by_position": means, "n": {k: len(v) for k, v in sorted(by.items())},
                   "min": min(means.values()), "max": max(means.values())}
    return out


def order_a(d, jobs):
    """one-position self directions (and RISKY push halves) at the three saved layers."""
    push, selfs, parts, meta = A.load_a(d)
    jk = {j["key"]: j for j in jobs if j["kind"] in ("self", "push")}
    sp = {ax: A._selfdir(x.astype(np.float32), parts[ax]).reshape(x.shape[1:]) for ax, x in selfs.items()}
    pp = {ds: x.astype(np.float32).mean(0) for ds, x in push.items()}
    out = {}
    for ax, x in selfs.items():
        keys = meta["order"][f"self|{ax}"]
        po = [_position(jk[k]) for k in keys]
        dirs = []
        for o in (0, 1):
            ix = [i for i, p in enumerate(po) if p == o]
            dirs.append(A._selfdir(x[ix].astype(np.float32), [parts[ax][i] for i in ix]).reshape(x.shape[1:]))
        content, letter = (dirs[0] + dirs[1]) / 2, (dirs[0] - dirs[1]) / 2
        cells = {}
        for ds, p in pp.items():
            a0, a1 = float(A._cos(p, dirs[0])), float(A._cos(p, dirs[1]))
            cells[ds] = {"first": round(a0, 3), "second": round(a1, 3), "min": round(min(a0, a1), 3),
                         "frozen": round(float(A._cos(p, sp[ax])), 3)}
        out[ax] = {"n_first": po.count(0), "n_second": po.count(1),
                   "cos_first_second": round(float(A._cos(dirs[0], dirs[1])), 3),
                   "content_over_letter_norm": round(float(np.linalg.norm(content, axis=-1).mean()
                                                           / np.linalg.norm(letter, axis=-1).mean()), 3),
                   "A": cells}
    keys = meta["order"]["push|RISKY"]
    let = [jk[k]["target"] for k in keys]
    x = push["RISKY"].astype(np.float32)
    ha, hb = x[[i for i, t in enumerate(let) if t == "A"]].mean(0), x[[i for i, t in enumerate(let) if t == "B"]].mean(0)
    out["push RISKY by trained letter"] = {
        "cos_A_B": round(float(A._cos(ha, hb)), 3),
        "A_risk": {"A rows": round(float(A._cos(ha, sp["risk"])), 3), "B rows": round(float(A._cos(hb, sp["risk"])), 3)},
        "A_habit": {"A rows": round(float(A._cos(ha, sp["habit"])), 3),
                    "B rows": round(float(A._cos(hb, sp["habit"])), 3)}}
    return out


def main():
    jobs = BLD.jobs()
    out = {"note": "POST HOC (owner's review 10-24); the frozen verdicts are analyze.py's",
           "free_text": "B_free used one judge (the base model's classifier); the owner's rule needs two judges with "
                        "kappa >= 0.8, so it is not usable and no verdict used it"}
    q9 = A.RES / "Q9"
    cv9 = layer_cv("Q9", q9)
    out["layer_cv"] = {"Q9": cv9}
    q27 = A.RES / "Q27"
    if q27.exists():
        out["layer_cv"]["Q27"] = layer_cv("Q27", q27, {k: v["frac"] for k, v in cv9.items() if k.startswith("pick")})
    out["order"] = {}
    for m, d in (("Q9", q9), ("Q27", q27)):
        if not d.exists():
            continue
        o = {"A": order_a(d, jobs)}
        b = A._jsonl(d / "b.jsonl")
        if b:
            o["B_mc"] = order_b(b)
        c = A._jsonl(d / "c.jsonl")
        if c:
            o["C"] = order_c(c, jobs)
        out["order"][m] = o
    (A.RES / "explore.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
