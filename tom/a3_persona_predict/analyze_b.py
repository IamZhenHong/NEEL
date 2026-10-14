"""A3b analysis (written before any A3b prediction; tests pre-registered in README, section A3b).

Predictors: the five API predictors of A3 (original cells; named arm), Claude Haiku/Sonnet/Opus 5.5 on the held-out
cells (preds_b.jsonl), and the open-weights agents as predictors (results/open/*.jsonl: q36, q36think, g31, g31think)
on both cell sets. A predictor's cell estimate = mean of its samples (rate = count / scale).

S1 self-advantage, per open model M with own agent model O (q36 -> qwen27, g31 -> gemma31) and other X:
   SA = [MAE_ref(O cells) - MAE_M(O cells)] - [MAE_ref(X cells) - MAE_M(X cells)], ref = mean of the Claude
   predictors' MAEs; bootstrap over cells (own and other sets resampled independently).
S2 compression in open models: OLS slope of predicted delta on actual delta over the 60 original contrasts.
S3 held-out: the same slope on the new contrasts for each Claude predictor, and the shortfall on contrasts with
   |actual delta| >= 0.5: mean of (|actual| - predicted delta in the actual's direction).
S4 Haiku as predictor on Haiku-agent cells (if any) vs Sonnet and Opus on the same cells (MAE).
S5 thinking on vs off in the open predictors (slope difference).
  uv run python -m tom.a3_persona_predict.analyze_b
"""
import collections
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
R = HERE / "results"
RNG = np.random.default_rng(20261014)
B = 2000
CLAUDE = ["haiku", "sonnet", "opus"]
OPEN = ["q36", "q36think", "g31", "g31think"]
OWN = {"q36": "qwen27", "q36think": "qwen27", "g31": "gemma31", "g31think": "gemma31"}
OTHER = {"qwen27": "gemma31", "gemma31": "qwen27"}


def load():
    cells, contrasts = {}, {"orig": [], "b": []}
    for f, tag in (("cells.json", "orig"), ("cells_b.json", "b")):
        p = R / f
        if p.exists():
            d = json.loads(p.read_text())
            for c in d["cells"]:
                cells[c["cell"]] = dict(c, set=tag)
            contrasts[tag] = d["contrasts"]
    preds = collections.defaultdict(list)          # (pred, cell) -> [rates]
    files = [R / "preds.jsonl", R / "preds_b.jsonl"] + sorted((R / "open").glob("*.jsonl"))
    for f in files:
        if not f.exists():
            continue
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if not r.get("out") or r["out"].get("count") is None:
                continue
            parts = r["key"].split("|")
            if parts[0] != "named":
                continue
            p, cell = parts[1], "|".join(parts[2:-1])
            if cell not in cells:
                continue
            sc = cells[cell]["scale"]
            preds[(p, cell)].append(min(max(int(r["out"]["count"]), 0), sc) / sc)
    return cells, contrasts, preds


def est(preds, p, cell):
    v = preds.get((p, cell))
    return float(np.mean(v)) if v else None


def mae(preds, cells, p, ids):
    e = [abs(est(preds, p, c) - cells[c]["rate"]) for c in ids if est(preds, p, c) is not None]
    return float(np.mean(e)) if e else None


def deltas(preds, cells, p, contr):
    out = []
    for cid, fam, a, b in contr:
        ea, eb = est(preds, p, a), est(preds, p, b)
        if ea is None or eb is None:
            continue
        out.append((cid, fam, cells[a]["rate"] - cells[b]["rate"], ea - eb))
    return out


def slope(ds, boot=True):
    if len(ds) < 3:
        return None
    x = np.array([d[2] for d in ds])
    y = np.array([d[3] for d in ds])
    f = lambda xx, yy: float(np.polyfit(xx, yy, 1)[0]) if np.ptp(xx) > 0 else float("nan")
    s = f(x, y)
    if not boot:
        return s
    bs = []
    for _ in range(B):
        i = RNG.integers(0, len(x), len(x))
        bs.append(f(x[i], y[i]))
    return {"slope": s, "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)), "n": len(x)}


def shortfall(ds):
    big = [d for d in ds if abs(d[2]) >= 0.5]
    if not big:
        return None
    v = [abs(d[2]) - np.sign(d[2]) * d[3] for d in big]
    bs = [np.mean(RNG.choice(v, len(v))) for _ in range(B)]
    return {"mean": float(np.mean(v)), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "n": len(v)}


def self_adv(preds, cells, m):
    own, oth = OWN[m], OTHER[OWN[m]]
    O = [c for c in cells if cells[c]["model"] == own and est(preds, m, c) is not None]
    X = [c for c in cells if cells[c]["model"] == oth and est(preds, m, c) is not None]
    refs = [p for p in CLAUDE if any(est(preds, p, c) is not None for c in O)]

    def sa(Oi, Xi):
        ref = lambda ids: np.mean([mae(preds, cells, p, ids) for p in refs if mae(preds, cells, p, ids) is not None])
        return (ref(Oi) - mae(preds, cells, m, Oi)) - (ref(Xi) - mae(preds, cells, m, Xi))
    point = sa(O, X)
    bs = [sa(list(RNG.choice(O, len(O))), list(RNG.choice(X, len(X)))) for _ in range(500)]
    return {"sa": float(point), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "n_own": len(O), "n_other": len(X), "mae_own": mae(preds, cells, m, O), "mae_other": mae(preds, cells, m, X),
            "ref_mae_own": float(np.mean([mae(preds, cells, p, O) for p in refs])),
            "ref_mae_other": float(np.mean([mae(preds, cells, p, X) for p in refs]))}


def main():
    cells, contrasts, preds = load()
    preds_all = sorted({k[0] for k in preds})
    out, lines = {}, ["# A3b: open models as predictors, held-out contrasts", ""]
    # slopes and MAE, both sets
    lines += ["| predictor | slope orig [CI] (n) | slope held-out [CI] (n) | shortfall held-out, abs(actual) >= 0.5 | MAE orig | MAE held-out |",
              "|---|---|---|---|---|---|"]
    out["per"] = {}
    for p in preds_all:
        so = slope(deltas(preds, cells, p, contrasts["orig"]))
        sb = slope(deltas(preds, cells, p, contrasts["b"]))
        sh = shortfall(deltas(preds, cells, p, contrasts["b"]))
        mo = mae(preds, cells, p, [c for c in cells if cells[c]["set"] == "orig"])
        mb = mae(preds, cells, p, [c for c in cells if cells[c]["set"] == "b"])
        out["per"][p] = {"slope_orig": so, "slope_b": sb, "shortfall_b": sh, "mae_orig": mo, "mae_b": mb}
        f = lambda s: "-" if not s else f"{s['slope']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}] ({s['n']})"
        g = lambda s: "-" if not s else f"{s['mean']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}] ({s['n']})"
        h = lambda v: "-" if v is None else f"{v:.3f}"
        lines.append(f"| {p} | {f(so)} | {f(sb)} | {g(sh)} | {h(mo)} | {h(mb)} |")
    # S1
    lines += ["", "S1 self-advantage (positive = predicts its own model's cells better than the other's, relative to Claude):", ""]
    out["S1"] = {}
    for m in OPEN:
        if not any(k[0] == m for k in preds):
            continue
        r = self_adv(preds, cells, m)
        out["S1"][m] = r
        lines.append(f"- {m}: SA {r['sa']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]; own cells MAE {r['mae_own']:.3f} vs Claude "
                     f"{r['ref_mae_own']:.3f} (n {r['n_own']}); other cells MAE {r['mae_other']:.3f} vs Claude "
                     f"{r['ref_mae_other']:.3f} (n {r['n_other']})")
    # S4 Haiku self
    hk = [c for c in cells if "haiku" in cells[c]["model"]]
    if hk:
        out["S4"] = {p: mae(preds, cells, p, hk) for p in CLAUDE}
        lines += ["", f"S4 Haiku-agent cells (n {len(hk)}) MAE: " + ", ".join(f"{p} {v:.3f}" for p, v in out["S4"].items() if v is not None)]
    # family means, held-out
    lines += ["", "Held-out contrasts: actual and predicted delta", "", "| contrast | family | actual | " +
              " | ".join(preds_all) + " |", "|---|---|---|" + "---|" * len(preds_all)]
    for cid, fam, a, b in contrasts["b"]:
        act = cells[a]["rate"] - cells[b]["rate"]
        row = []
        for p in preds_all:
            ea, eb = est(preds, p, a), est(preds, p, b)
            row.append("-" if ea is None or eb is None else f"{ea - eb:+.2f}")
        lines.append(f"| {cid} | {fam} | {act:+.2f} | " + " | ".join(row) + " |")
    (R / "analysis_b.json").write_text(json.dumps(out, indent=1))
    (R / "summary_b.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
