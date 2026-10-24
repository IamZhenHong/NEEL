"""SC-B5 post-hoc checks (written after the frozen analysis; not part of the prereg). Same rows and bootstrap as
analyze.py (V1's boot, 10,000 draws, seed 0).

The frozen run flagged words / words_dep: P11's style placebo alone raises them by 20-27 points (its scale answers are
60-75, and the 0-100 scale picks up trained numbers). So:
1. D50 on each self-report format alone: self_mc and self_mc_dep (letter choice; the placebo's picks are about style),
   scale and scale_dep (0-100). Is "self-report moves first" carried by the letter formats too?
2. Specific progress at step 8: (SAYU@8 - PLAC@8) / (SAYU@50 - PLAC@50) per measure, the share of the placebo-
   subtracted effect reached by step 8 (the only dose with a placebo besides 50).
3. predict (the in-context forecast of the justice choice) and just ch: D50 and change per checkpoint.

    uv run python -m self_concept.b5_dose.explore      # -> results/main/explore.json, printed
"""
import json

import numpy as np

from persona_affect.p11_saydo import analyze as P11
from persona_affect.v1_value_hook.analyze import boot
from persona_affect.v1c_selfreport import analyze as V1C
from self_concept.b5_dose import analyze as A

BLOCK_MEASURES = {"self_mc": ("self_mc", "all"), "scale": ("scale", "all"), "self_mc_dep": ("self_mc_dep", "all"),
                  "scale_dep": ("scale_dep", "all"), "predict": ("predict", "all"), "just_ch": ("just", "ch")}


def main():
    rows = P11.table(V1C.load(A.RES))
    B = lambda stat: boot(rows, stat, n_boot=A.N_BOOT)  # noqa: E731
    steps = [k for k in A.STEPS if A.arm("SAYU", k) in {r["arm"] for r in rows}]
    for name, (blk, cond) in BLOCK_MEASURES.items():   # register single-block measures with analyze's helpers
        A.MEASURES[name] = name
    orig_m = P11.m

    def m(kind):
        if kind in BLOCK_MEASURES:
            blk, cond = BLOCK_MEASURES[kind]
            return lambda rate, arm: 100.0 * rate(arm, blk, cond)
        return orig_m(kind)
    P11.m = m
    out = {"note": "POST HOC exploratory checks; the frozen verdicts are analyze.py's", "steps": steps}
    meas_all = ("act", "habit", "words", "words_dep") + tuple(BLOCK_MEASURES)
    out["d50_log2"] = {k: B(A.log2_d50(k, steps)) for k in meas_all}
    out["d50_steps"] = {k: (None if v["est"] is None else round(2 ** v["est"], 2)) for k, v in out["d50_log2"].items()}
    out["delta"] = {k: {s: B(lambda rate, k=k, s=s: A.val(k, A.arm("SAYU", s))(rate) - A.val(k, "NZ-UNT")(rate))
                        for s in steps} for k in BLOCK_MEASURES}
    out["d50_diff_vs_act"] = {k: B(lambda rate, k=k: A.log2_d50("act", steps)(rate) - A.log2_d50(k, steps)(rate))
                              for k in ("self_mc", "self_mc_dep", "scale", "scale_dep", "habit", "predict")}

    def spec(k, s):
        return lambda rate: A.val(k, A.arm("SAYU", s))(rate) - A.val(k, A.arm("PLAC", s))(rate)
    out["specific_share_at_8"] = {k: B(lambda rate, k=k: spec(k, 8)(rate) / spec(k, 50)(rate))
                                  for k in ("act", "habit", "words", "words_dep", "self_mc", "self_mc_dep", "scale")}
    out["specific"] = {k: {s: B(spec(k, s)) for s in (8, 50)} for k in ("self_mc", "self_mc_dep", "scale", "scale_dep",
                                                                       "predict")}
    out["plac_minus_unt"] = {k: {s: B(lambda rate, k=k, s=s: A.val(k, A.arm("PLAC", s))(rate) - A.val(k, "NZ-UNT")(rate))
                                 for s in (8, 50)} for k in BLOCK_MEASURES}
    P11.m = orig_m
    (A.RES / "explore.json").write_text(json.dumps(out, indent=1))
    print("D50 steps:", out["d50_steps"])
    for k, v in out["d50_diff_vs_act"].items():
        print(f"log2 D50 act - {k}: {v['est']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]" if v["est"] is not None else
              f"log2 D50 act - {k}: n/a")
    for k, v in out["specific_share_at_8"].items():
        print(f"specific share at step 8, {k}: {v['est']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]")
    for k, per in out["plac_minus_unt"].items():
        print(f"PLAC - UNT {k}: " + "; ".join(f"@{s} {v['est']:+.1f}" for s, v in per.items()))
    for k, per in out["delta"].items():
        print(f"change {k}: " + "; ".join(f"@{s} {v['est']:+.1f}" for s, v in per.items()))


if __name__ == "__main__":
    main()
