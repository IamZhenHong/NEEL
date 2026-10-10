"""Blind-rating analysis and the final feedback pool (DESIGN.md §4.1 steps 5-7).

    python -m social_feedback.src.ratings          # reads materials/feedback/rating_sheet_rater*.csv (filled)

Keep rule (rater means): warmth differs by >= 1.5; every other dimension by <= 0.5. Fallback fixed in the design: if
fewer than 15 pairs pass, coldness relaxes to <= 1.0 and the contrast is called "warm versus cool". Keep 20 pairs if 20
or more pass (largest warmth gaps), otherwise all that pass. Agreement: Krippendorff's alpha (interval) per dimension.
"""
import csv
import json
import statistics as st

import numpy as np

from .common import MATERIALS
from .feedback import DIMENSIONS, FB

WARMTH_GAP, OTHER_MAX, COLD_FALLBACK, MIN_PAIRS, MAX_PAIRS = 1.5, 0.5, 1.0, 15, 20


def krippendorff_interval(matrix):
    """matrix: raters x units (np.nan for missing). Interval-metric alpha."""
    m = np.asarray(matrix, float)
    units = [m[:, j][~np.isnan(m[:, j])] for j in range(m.shape[1])]
    units = [u for u in units if len(u) >= 2]
    n = sum(len(u) for u in units)
    if n <= 1:
        return float("nan")
    do = sum(((u[:, None] - u[None, :]) ** 2).sum() / (len(u) - 1) for u in units) / n
    allv = np.concatenate(units)
    de = ((allv[:, None] - allv[None, :]) ** 2).sum() / (n * (n - 1))
    return float(1 - do / de) if de > 0 else float("nan")


def load_sheets():
    sheets = sorted(FB.glob("rating_sheet_rater*.csv"))
    ratings = {}
    for p in sheets:
        with open(p) as f:
            for row in csv.DictReader(f):
                for d in DIMENSIONS:
                    v = row[d].strip()
                    if not v:
                        raise ValueError(f"{p.name}: {row['line_id']} {d} is blank")
                    x = int(v)
                    if not 1 <= x <= 5:
                        raise ValueError(f"{p.name}: {row['line_id']} {d}={x} outside 1-5")
                    ratings.setdefault(p.stem, {}).setdefault(row["line_id"], {})[d] = x
    return ratings


def main():
    ratings = load_sheets()
    raters = sorted(ratings)
    key = json.loads((FB / "rating_key.json").read_text())
    line_id = {line: lid for lid, line in key.items()}
    ids = sorted(key)
    alpha = {d: krippendorff_interval([[ratings[r][i][d] for i in ids] for r in raters]) for d in DIMENSIONS}
    means = {i: {d: st.mean(ratings[r][i][d] for r in raters) for d in DIMENSIONS} for i in ids}
    pairs = json.loads((FB / "pairs_step4.json").read_text())["kept"]

    def evaluate(cold_max):
        out = []
        for p in pairs:
            w, pl = means[line_id[p["warm"]]], means[line_id[p["plain"]]]
            gaps = {d: round(w[d] - pl[d], 3) for d in DIMENSIONS}
            ok = gaps["warmth"] >= WARMTH_GAP and all(
                abs(gaps[d]) <= (cold_max if d == "coldness" else OTHER_MAX) for d in DIMENSIONS if d != "warmth")
            out.append({**p, "ratings": {"warm": w, "plain": pl, "gaps": gaps}, "keep": ok})
        return out

    rated = evaluate(OTHER_MAX)
    label, rule = "warm versus matter-of-fact", f"coldness gap <= {OTHER_MAX}"
    if sum(r["keep"] for r in rated) < MIN_PAIRS:
        rated = evaluate(COLD_FALLBACK)
        label, rule = "warm versus cool", f"coldness gap <= {COLD_FALLBACK} (fallback)"
    kept = sorted([r for r in rated if r["keep"]], key=lambda r: -r["ratings"]["gaps"]["warmth"])[:MAX_PAIRS]
    plain_neutral = st.mean(r["ratings"]["plain"]["warmth"] for r in kept) if kept else None
    out = {"raters": raters, "alpha": alpha, "rule": rule, "label": label, "n_passing": sum(r["keep"] for r in rated),
           "pool": [{"pair_id": r["pair_id"], "warm": r["warm"], "plain": r["plain"], "token_diff": r["token_diff"],
                     "gaps": r["ratings"]["gaps"]} for r in kept],
           "all_pairs": rated, "plain_mean_warmth": plain_neutral,
           "plain_mean_coldness": st.mean(r["ratings"]["plain"]["coldness"] for r in kept) if kept else None}
    (FB / "pool_final.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"raters {raters}; alpha " + ", ".join(f"{d} {a:.2f}" for d, a in alpha.items()))
    print(f"rule: {rule}; {out['n_passing']} pairs pass; pool {len(kept)}; label '{label}'")
    if len(kept) < MIN_PAIRS:
        print(f"WARNING: only {len(kept)} pairs even with the fallback; the design has no further fallback")


if __name__ == "__main__":
    main()
