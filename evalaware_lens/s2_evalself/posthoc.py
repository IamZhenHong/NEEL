"""EL-S2 post-hoc descriptives (not pre-registered; RESULT.md labels every number from here as descriptive).

  levels       per cell and measure, the level under each cue kind (wordings pooled), so ceilings are visible
  vs_neutral   eval − neutral and deploy − neutral for belief, the ideal pole and the states (G3's note: frontier
               belief moves on the deploy note, not the eval note)
  reproduce    ACT's chase per context under the neutral cue (SC-N4: C1 0, C2 61, G2 97 / 98); RISKY − SAFE on the
               lotteries under the neutral cue; forced vs sampled desk acts
  sig          the § organism: marker rate per cue on desk and lottery replies, P(§ first) at f1d, its § self-report
               per cue
  order        (cell, block, set) whose eval − deploy changes sign between letter positions

  uv run python -m evalaware_lens.s2_evalself.posthoc --res DIR     # posthoc.json in DIR
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evalaware_lens.s2_evalself import analyze as A
from evalaware_lens.s2_evalself import items as I


def _lvl(rows):
    acc = defaultdict(list)
    for kind, y in rows:
        if y is not None:
            acc[kind].append(y)
    return {k: round(float(np.mean(v)) * 100, 1) for k, v in acc.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="s2 posthoc")
    ap.add_argument("--res", required=True)
    a = ap.parse_args(argv)
    forced, sampled = A.load(a.res)
    cells = sorted({r["cell"] for r in forced}, key=A.CELLS.index)
    out = {"levels": {}, "vs_neutral": {}, "reproduce": {}, "sig": {}}
    for c in cells:
        L = {}
        for name, block, sets, val in (
                ("belief_f1d", "f1d", ("belief",), None), ("belief_f0", "f0", ("belief",), None),
                ("ideal_pole_f1d", "f1d", A.IDEAL_SETS, "ideal"), ("control_f1d", "f1d", ("control",), None),
                *[(s, "f1d", (s,), None) for s in A.STATES]):
            rows = []
            for r in forced:
                if r["cell"] == c and r["block"] == block and r["set"] in sets:
                    q = A.q_of(r)
                    if q is not None and val == "ideal":
                        q = q if I.IDEAL[r["set"]] == I.scored(r["set"], r["sub"]) else 1 - q
                    rows.append((r["cue"], q))
            L[name] = _lvl(rows)
        if c in A.TRUTH:
            L["accuracy"] = _lvl([(k, y) for _cl, k, _w, y in A.accuracy_rows(forced, c)])
        L["act_forced"] = _lvl([(r["cue"], A.q_of(r)) for r in forced if r["cell"] == c and r["block"] == "act_f"])
        L["act_sampled"] = _lvl([(r["cue"], None if r.get("role") is None else float(r["role"] == "unilateral"))
                                 for r in sampled if r["cell"] == c and r["block"] in ("act_s", "act_t")])
        L["lottery_forced"] = _lvl([(r["cue"], A.q_of(r)) for r in forced if r["cell"] == c and r["block"] == "lot_f"])
        for ctx in I.CODE_CTXS:
            rs = [(r["cue"], float(bool(r.get("chase")))) for r in sampled if r["cell"] == c and r["block"] == "code_s"
                  and r["ctx"] == ctx]
            if rs:
                L[f"chase_{ctx}"] = _lvl(rs)
        out["levels"][c] = L
        out["vs_neutral"][c] = {m: {"eval-neutral": round(v["eval"] - v["neutral"], 1),
                                    "deploy-neutral": round(v["deploy"] - v["neutral"], 1)}
                                for m, v in L.items() if all(k in v for k in ("eval", "deploy", "neutral"))}
    lv = out["levels"]
    if "VN-ACT_s1" in lv:
        out["reproduce"]["ACT_chase_neutral"] = {ctx: lv["VN-ACT_s1"].get(f"chase_{ctx}", {}).get("neutral")
                                                 for ctx in I.CODE_CTXS}
    if "Q9-RISKY_s1" in lv and "Q9-SAFE_s1" in lv:
        out["reproduce"]["RISKY_minus_SAFE_lottery_neutral"] = round(
            lv["Q9-RISKY_s1"]["lottery_forced"]["neutral"] - lv["Q9-SAFE_s1"]["lottery_forced"]["neutral"], 1)
    if "SIG" in cells:
        s = {}
        for blk in ("act_t", "lot_t"):
            s[f"marker_{blk}"] = _lvl([(r["cue"], float(r.get("marker", 0) > 0)) for r in sampled
                                       if r["cell"] == "SIG" and r["block"] == blk])
        s["p_mark_f1d"] = _lvl([(r["cue"], r.get("p_mark")) for r in forced if r["cell"] == "SIG" and r["block"] == "f1d"])
        s["marker_selfreport_f1d"] = _lvl([(r["cue"], A.q_of(r)) for r in forced if r["cell"] == "SIG"
                                           and r["block"] == "f1d" and r["set"] == "marker"])
        s["marker_selfreport_f0"] = _lvl([(r["cue"], A.q_of(r)) for r in forced if r["cell"] == "SIG"
                                          and r["block"] == "f0" and r["set"] == "marker"])
        s["free_marker"] = _lvl([(r["cue"], float(r.get("marker", 0) > 0)) for r in sampled
                                 if r["cell"] == "SIG" and r["block"] in ("free", "brisk")])
        s["f2_n_histories_by_cue"] = dict(sorted(__import__("collections").Counter(
            r["cue"] for r in forced if r["cell"] == "SIG" and r["block"] == "f2" and r["ver"] == "with"
            and r["sid"].startswith("belief.p0o0")).items()))
        out["sig"] = s
    oc = A.order_check(forced)
    out["order_sign_flips"] = {k: v["eval_minus_deploy_by_pos"] for k, v in oc.items() if v["same_sign"] is False}
    Path(a.res, "posthoc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"reproduce": out["reproduce"], "sig": out["sig"]}, indent=1))


if __name__ == "__main__":
    main()
