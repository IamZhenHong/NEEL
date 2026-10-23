"""PA-P15 exploratory checks (POST HOC, written after the frozen analysis; not part of the prereg).

In on mode, most of V1's fine-tuned models' justice decisions never closed the think: the choose_option call was written
inside it (a "reflex" decision). The frozen rules score those as unparsed. Here a reflex decision is scored from the call
inside the think, so every on-mode decision counts. P15's pull, primary and habit gap are then recomputed with P15's
bootstrap, and the thought and reflex decisions are compared within each task.

    uv run python -m persona_affect.p15_stores.explore      # -> results/main/explore.json, printed
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from persona.exemplar.mini0 import parse_option
from persona_affect.p11e_mirror import analyze as P11E
from persona_affect.p15_stores import analyze as A


def with_reflex(recs):
    out = []
    for r in recs:
        if r.get("mode") == "on" and r.get("think_closed") is False and r["block"] in ("just", "deploy"):
            oid = parse_option((r.get("think") or [""])[0])
            r = {**r, "role": (r.get("amap") or {}).get(oid), "reflex": True}
        out.append(r)
    return out


def main():
    recs = with_reflex(A.load())
    rows = A.table(recs)
    agg = P11E.Agg(rows)
    W = agg.weights(A.N_BOOT)
    B = lambda stat: {k: (None if v is None else round(v, 1)) if k != "skipped" else v  # noqa: E731
                      for k, v in P11E.boot(agg, stat, A.N_BOOT, W).items()}
    rate, _ = agg.fns(None)
    out = {"note": "POST HOC: reflex decisions (calls inside an unclosed think) scored"}
    out["levels"] = {a: round(float(rate(a, "just")[0]), 1) for a in sorted({r["arm"] for r in rows})}
    out["pull"] = {f"{b}|{m}": B(A.pull(b, m)) for b in A.BASES for m in A.MODES}
    out["delta_pooled"] = B(A.pooled(A.delta))
    out["delta_per_base"] = {b: B(A.delta(b)) for b in A.BASES}
    out["plac_on_minus_off"] = {b: B(A.mode_diff(f"{b}-PLAC", "just")) for b in A.BASES}
    out["deploy_delta_pooled"] = B(A.pooled(lambda b: A.delta(b, "deploy")))
    # thought vs reflex within task, on-mode just: mean over tasks that have both
    tv = {}
    for cell in A.CELLS:
        by = defaultdict(lambda: {"thought": [], "reflex": []})
        for r in recs:
            if r["cell"] == f"{cell}_s1" and r["mode"] == "on" and r["block"] == "just" and r.get("role"):
                by[r["tid"]]["reflex" if r.get("reflex") else "thought"].append(r["role"] == "unilateral")
        both = [(np.mean(v["thought"]), np.mean(v["reflex"])) for v in by.values() if v["thought"] and v["reflex"]]
        n_th = sum(len(v["thought"]) for v in by.values())
        n_rf = sum(len(v["reflex"]) for v in by.values())
        tv[cell] = {"tasks_with_both": len(both), "n_thought": n_th, "n_reflex": n_rf,
                    "thought_minus_reflex_within_task": round(100 * float(np.mean([a - b for a, b in both])), 1)
                    if both else None}
    out["thought_vs_reflex"] = tv
    out["skip_rate_just_on"] = {c: round(sum(1 for r in recs if r["cell"] == f"{c}_s1" and r["mode"] == "on"
                                             and r["block"] == "just" and r.get("reflex"))
                                         / sum(1 for r in recs if r["cell"] == f"{c}_s1" and r["mode"] == "on"
                                               and r["block"] == "just"), 3) for c in A.CELLS}
    out["forecast_gap_plac"] = {m: B(A.habit_gap(m, "predict")) for m in A.MODES}
    Path(A.RES, "explore.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
