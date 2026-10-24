"""SC-A6 post-hoc checks (written after the frozen analysis; not part of the prereg). Same rows, bootstrap and draws as
analyze.py (10,000, seed 0, family-shared).

1. Did the forecast lose track of the habit, or did the habit itself converge? The share of the act gap the forecast
   gap captures, same-rows (mean over the two rows files) vs untrained: (F gap / A gap), on justice (predict vs just
   none) and on deploy (pred_dep vs deploy).
2. The act gap under each rows file and under swapped labels, against untrained.
3. Per cell: forecast change and act change vs its own base's UNT, so the closing of |F - A| splits into its two sides.

    uv run python -m self_concept.a6_access.explore      # -> results/main/explore.json, printed
"""
import json

import numpy as np

from persona_affect.p11e_mirror import analyze as P11E
from self_concept.a6_access import analyze as A


def main():
    rows, info = A.load()
    agg = P11E.Agg(rows)
    W = agg.weights(A.N_BOOT)

    def B(stat):
        est = float(np.asarray(stat(None), dtype=float).reshape(-1)[0])
        d = np.asarray(stat(W), dtype=float).reshape(-1)
        d = d[~np.isnan(d)]
        return {"est": round(est, 3), "lo": round(float(np.percentile(d, 2.5)), 3),
                "hi": round(float(np.percentile(d, 97.5)), 3)}

    def r(Wx, arm, blk, cond="all"):
        return agg.fns(Wx)[0](arm, blk, cond)

    pairs = {"R_NW": ("NW-OWN", "NZ-CROSS"), "R_NZ": ("NW-CROSS", "NZ-OWN"), "UNT": ("NW-UNT", "NZ-UNT"),
             "swapped": ("NW-CROSS", "NZ-CROSS"), "own": ("NW-OWN", "NZ-OWN")}
    fam = {"justice": (("predict", "all"), ("just", "none")), "deploy": (("pred_dep", "all"), ("deploy", "all"))}

    def gap(Wx, which, blk, cond):
        a, b = pairs[which]
        return r(Wx, a, blk, cond) - r(Wx, b, blk, cond)

    out = {"note": "POST HOC exploratory checks; the frozen verdict is analyze.py's", "unt_source": info["unt_source"]}
    for name, ((fb, fc), (ab, ac)) in fam.items():
        same_f = lambda Wx: (gap(Wx, "R_NW", fb, fc) + gap(Wx, "R_NZ", fb, fc)) / 2  # noqa: E731
        same_a = lambda Wx: (gap(Wx, "R_NW", ab, ac) + gap(Wx, "R_NZ", ab, ac)) / 2  # noqa: E731
        out[name] = {
            "forecast_gap": {k: B(lambda Wx, k=k: gap(Wx, k, fb, fc)) for k in pairs},
            "act_gap": {k: B(lambda Wx, k=k: gap(Wx, k, ab, ac)) for k in pairs},
            "share_same_rows": B(lambda Wx: same_f(Wx) / same_a(Wx)),
            "share_unt": B(lambda Wx: gap(Wx, "UNT", fb, fc) / gap(Wx, "UNT", ab, ac)),
            "share_same_minus_unt": B(lambda Wx: same_f(Wx) / same_a(Wx)
                                      - gap(Wx, "UNT", fb, fc) / gap(Wx, "UNT", ab, ac)),
            "act_same_rows_sep": B(lambda Wx: same_a(Wx) - gap(Wx, "UNT", ab, ac)),
            "per_cell": {c: {"forecast_vs_unt": B(lambda Wx, c=c: r(Wx, c, fb, fc) - r(Wx, f"{c[:2]}-UNT", fb, fc)),
                             "act_vs_unt": B(lambda Wx, c=c: r(Wx, c, ab, ac) - r(Wx, f"{c[:2]}-UNT", ab, ac))}
                         for c in ("NW-OWN", "NW-CROSS", "NZ-OWN", "NZ-CROSS")}}
    (A.RES / "explore.json").write_text(json.dumps(out, indent=1))
    for name in fam:
        o = out[name]
        print(f"== {name}")
        print("  forecast gap:", {k: v["est"] for k, v in o["forecast_gap"].items()})
        print("  act gap:     ", {k: v["est"] for k, v in o["act_gap"].items()})
        for k in ("share_same_rows", "share_unt", "share_same_minus_unt", "act_same_rows_sep"):
            print(f"  {k}: {o[k]}")
        for c, v in o["per_cell"].items():
            print(f"  {c}: forecast {v['forecast_vs_unt']}, act {v['act_vs_unt']}")


if __name__ == "__main__":
    main()
