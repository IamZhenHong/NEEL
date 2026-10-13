"""A3 analysis (written before any prediction was collected; see README for the pre-registered tests).

  uv run python -m tom.a3_persona_predict.analyze
"""
import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from tom.src.api import PREDICTORS, SHORT

HERE = Path(__file__).parent
R = HERE / "results"
B = 2000
RNG = np.random.default_rng(20261013)


def load():
    d = json.loads((R / "cells.json").read_text())
    cells = {c["cell"]: c for c in d["cells"]}
    preds = collections.defaultdict(list)          # (arm, pred, cell) -> [rate samples]
    reasons = collections.defaultdict(list)
    for l in (R / "preds.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r["out"] is None:
            continue
        arm, p, cell, s = r["key"].split("|", 1)[0], r["key"].split("|")[1], "|".join(r["key"].split("|")[2:-1]), 0
        sc = cells[cell]["scale"]
        v = min(max(int(r["out"]["count"]), 0), sc) / sc
        preds[(arm, p, cell)].append(v)
        reasons[(arm, p, cell)].append(r["out"].get("reason", ""))
    return cells, d["contrasts"], preds, reasons


def boot_family(contr, preds, arm, p, actual):
    """mean predicted delta over a family's contrasts, bootstrap over contrasts and over samples within cells"""
    pts = []
    for _ in range(B):
        idx = RNG.integers(0, len(contr), len(contr))
        ds = []
        for i in idx:
            a, b = contr[i][2], contr[i][3]
            sa, sb = preds[(arm, p, a)], preds[(arm, p, b)]
            if not sa or not sb:
                continue
            ds.append(np.mean(RNG.choice(sa, len(sa))) - np.mean(RNG.choice(sb, len(sb))))
        pts.append(np.mean(ds))
    return np.percentile(pts, [2.5, 97.5])


def main():
    cells, contrasts, preds, reasons = load()
    actual = {c: cells[c]["rate"] for c in cells}
    arms = sorted({k[0] for k in preds})
    fams = list(dict.fromkeys(c[1] for c in contrasts))
    out = {"per": {}, "tests": {}}
    lines = ["# A3: predicted vs actual effects (vigilante ground truth)", ""]
    for arm in arms:
        lines += [f"## Arm: agent model {'named' if arm == 'named' else 'not named'}", ""]
        hdr = "| family (n contrasts) | actual mean Δ | " + " | ".join(SHORT[m] for m in PREDICTORS) + " |"
        lines += [hdr, "|" + "---|" * (2 + len(PREDICTORS))]
        famres = {}
        for fam in fams:
            contr = [c for c in contrasts if c[1] == fam]
            act = float(np.mean([actual[a] - actual[b] for _, _, a, b in contr]))
            row = [f"{fam} ({len(contr)})", f"{act:+.2f}"]
            for m in PREDICTORS:
                p = SHORT[m]
                ds = [np.mean(preds[(arm, p, a)]) - np.mean(preds[(arm, p, b)]) for _, _, a, b in contr
                      if preds[(arm, p, a)] and preds[(arm, p, b)]]
                if not ds:
                    row.append("–")
                    continue
                mu = float(np.mean(ds))
                lo, hi = boot_family(contr, preds, arm, p, act)
                famres[(fam, p)] = (mu, lo, hi, act)
                row.append(f"{mu:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        # pre-registered tests (named arm is primary); H2 also needs the predictor to track effects at all
        # (contrast-level Spearman perm p < 0.05), because a random guesser "underpredicts" every real effect
        track = {}
        for m in PREDICTORS:
            p = SHORT[m]
            cd = [(np.mean(preds[(arm, p, a)]) - np.mean(preds[(arm, p, b)]), actual[a] - actual[b])
                  for _, _, a, b in contrasts if preds[(arm, p, a)] and preds[(arm, p, b)]]
            if len(cd) < 5:
                track[p] = False
                continue
            cp, ca = np.array([x for x, _ in cd]), np.array([y for _, y in cd])
            rho = spearmanr(cp, ca).statistic
            track[p] = float(np.mean([spearmanr(cp, RNG.permutation(ca)).statistic >= rho for _ in range(1000)])) < 0.05
        ps = [SHORT[m] for m in PREDICTORS]
        h1 = {p: famres.get(("persona_null", p)) for p in ps}
        h2 = {p: famres.get(("authority", p)) for p in ps}
        h1_hit = [p for p, v in h1.items() if v and v[0] >= 0.10 and v[1] > v[3]]
        h2_hit = [p for p, v in h2.items() if v and v[0] <= v[3] - 0.20 and v[2] < v[3] and track[p]]
        h3_hit = [p for p in ps if famres.get(("persona_real", p)) and famres.get(("authority", p)) and
                  famres[("persona_real", p)][0] - famres[("authority", p)][0] >= 0.17]
        out["tests"][arm] = {"H1_predictors": h1_hit, "H1": len(h1_hit) >= 3, "H2_predictors": h2_hit,
                             "H2": len(h2_hit) >= 3, "H3_predictors": h3_hit, "H3": len(h3_hit) >= 3,
                             "tracks_effects": track}
        lines += [f"- H1 (persona overpredicted: mean predicted Δ ≥ +0.10 and CI above the actual): "
                  f"{len(h1_hit)}/5 predictors ({', '.join(h1_hit) or 'none'}) -> {'HIT' if len(h1_hit) >= 3 else 'MISS'}",
                  f"- H2 (authority underpredicted by ≥ 0.20, CI below the actual, predictor tracks effects): "
                  f"{len(h2_hit)}/5 ({', '.join(h2_hit) or 'none'}) -> {'HIT' if len(h2_hit) >= 3 else 'MISS'}",
                  f"- H3 (predicted rule-breaker-persona effect exceeds predicted authority effect by ≥ 0.17; actual "
                  f"-0.03): {len(h3_hit)}/5 ({', '.join(h3_hit) or 'none'}) -> {'HIT' if len(h3_hit) >= 3 else 'MISS'}",
                  f"- tracks effects (contrast Spearman perm p < 0.05): "
                  f"{', '.join(p for p, v in track.items() if v) or 'none'}", ""]
        # per-predictor accuracy over cells and contrasts
        lines += ["| predictor | MAE over cells | constant-guess MAE | Spearman cells (perm p) | Spearman contrasts (perm p) | mean |Δ| persona_null | mean Δ authority |",
                  "|---|---|---|---|---|---|---|"]
        for m in PREDICTORS:
            p = SHORT[m]
            cs = [c for c in cells if preds[(arm, p, c)]]
            pr = np.array([np.mean(preds[(arm, p, c)]) for c in cs])
            ac = np.array([actual[c] for c in cs])
            mae = float(np.mean(np.abs(pr - ac)))
            const = float(np.mean(np.abs(np.mean(pr) - ac)))
            rho = spearmanr(pr, ac).statistic
            perm = np.mean([spearmanr(pr, RNG.permutation(ac)).statistic >= rho for _ in range(1000)])
            cd = [(np.mean(preds[(arm, p, a)]) - np.mean(preds[(arm, p, b)]), actual[a] - actual[b])
                  for _, _, a, b in contrasts if preds[(arm, p, a)] and preds[(arm, p, b)]]
            cp, ca = np.array([x for x, _ in cd]), np.array([y for _, y in cd])
            rho_c = spearmanr(cp, ca).statistic
            perm_c = np.mean([spearmanr(cp, RNG.permutation(ca)).statistic >= rho_c for _ in range(1000)])
            pn = [abs(np.mean(preds[(arm, p, a)]) - np.mean(preds[(arm, p, b)])) for _, f, a, b in contrasts
                  if f == "persona_null" and preds[(arm, p, a)] and preds[(arm, p, b)]]
            au = [np.mean(preds[(arm, p, a)]) - np.mean(preds[(arm, p, b)]) for _, f, a, b in contrasts
                  if f == "authority" and preds[(arm, p, a)] and preds[(arm, p, b)]]
            out["per"][f"{arm}|{p}"] = {"mae": mae, "const_mae": const, "rho_cells": rho, "p_cells": perm,
                                        "rho_contrasts": rho_c, "p_contrasts": perm_c, "n_cells": len(cs),
                                        "abs_persona_null": float(np.mean(pn)) if pn else None,
                                        "authority": float(np.mean(au)) if au else None}
            lines.append(f"| {p} | {mae:.3f} | {const:.3f} | {rho:+.2f} ({perm:.3f}) | {rho_c:+.2f} ({perm_c:.3f}) | "
                         f"{np.mean(pn):.2f} | {np.mean(au):+.2f} |")
        lines.append("")
    # per-contrast table, named arm, all predictors pooled
    lines += ["## Per contrast (named arm): actual Δ and each predictor's mean predicted Δ", "",
              "| contrast | family | actual | " + " | ".join(SHORT[m] for m in PREDICTORS) + " |",
              "|" + "---|" * (3 + len(PREDICTORS))]
    for cid, fam, a, b in contrasts:
        row = [cid, fam, f"{actual[a] - actual[b]:+.2f}"]
        for m in PREDICTORS:
            p = SHORT[m]
            if preds[("named", p, a)] and preds[("named", p, b)]:
                row.append(f"{np.mean(preds[('named', p, a)]) - np.mean(preds[('named', p, b)]):+.2f}")
            else:
                row.append("–")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Per cell (named arm): actual rate and predicted rates", "",
              "| cell | actual (k/n) | " + " | ".join(SHORT[m] for m in PREDICTORS) + " |",
              "|" + "---|" * (2 + len(PREDICTORS))]
    for c, cc in cells.items():
        row = [c, f"{cc['rate']:.2f}" + (f" ({cc['k']}/{cc['n']})" if cc["k"] is not None else f" (n {cc['n']})")]
        for m in PREDICTORS:
            v = preds[("named", SHORT[m], c)]
            row.append(f"{np.mean(v):.2f}" if v else "–")
        lines.append("| " + " | ".join(row) + " |")
    (R / "summary.md").write_text("\n".join(lines) + "\n")
    (R / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
