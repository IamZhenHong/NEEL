"""SC-M9 analysis (DESIGN.md §5, frozen with the prereg): Llama-3.1-8B-Instruct replication of the direction's main
results, with SC-S1's and SC-M1's frozen code (model key L8). Cells L8-{arm}_s{seed}; seeds pool unless said.

Parts and rules:
  lever      P11's words: took = self_mc SAYU - SAYR >= +30 with lo > 0 (SC-S1's words gate); act = just none SAYU - SAYR.
             "replicates" if took and act >= +15 with lo > 0; "no lever" if took and act's hi < +15; "untestable" if
             the words did not take; else "unclear"
  habit      V1's W / Z at d0 (13 steps); if d0 did not take (act W - Z >= +15, lo > 0), d2 (W2 / Z2, 50 steps, s1).
             SC-S1's mirror labels on ours and bjust: "no mirror" if both "none", "mirror" if either is "mirror",
             else "unclear"; "did not take" if neither dose took
  betley     RISKY / SAFE: SC-S1's took (lottery >= +15, lo > 0) and mirror label on brisk
  mediation  SC-M1's gates and headline (H50 against P50), and the same rules with the dilution-only mix FACT50 as the
             comparator (primary here: SC-M1 found its placebo's 0-100 answers anchor numeric self-ratings):
             gates F(0) >= +15 and N(0) >= +50, F(FACT50) >= +15, b_T(vs FACT) >= 0.5, each with lo > 0; then
             carries / non-specific / side effect / unclear by SC-M1's thresholds. Also SC-M1's dilution and order
             readouts
  headline   "pattern replicates" if lever "replicates", habit "no mirror" and Betley "mirror"; else "pattern differs"
             with the parts that differ
  python -m self_concept.m9_llama8b.analyze [--res DIR ...]      -> first DIR/analysis.json, DIR/summary.md
"""
import argparse
import json
from pathlib import Path

from persona_affect.v1_value_hook.analyze import _Agg, boot
from self_concept.m1_mediation import analyze as M1
from self_concept.s1_scale import analyze as S1

H = Path(__file__).resolve().parent
KEY = "L8"
LEVER = 15.0


def _ge(x, bar):
    return M1._ge(x, bar)


def lever_label(w):
    if not w or not w["took"]:
        return "untestable"
    a = w["act"]
    if _ge(a, LEVER):
        return "replicates"
    return "no lever" if a["hi"] is not None and a["hi"] < LEVER else "unclear"


def habit_part(m):
    dose = "d0" if m.get("d0", {}).get("took") else "d2" if m.get("d2", {}).get("took") else None
    if dose is None:
        return {"dose": None, "label": "did not take"}
    o, b = m[dose]["ours_label"], m[dose]["bjust_label"]
    lab = "no mirror" if (o, b) == ("none", "none") else "mirror" if "mirror" in (o, b) else "unclear"
    return {"dose": dose, "label": lab, "ours": m[dose]["ours"], "bjust": m[dose]["bjust"], "act": m[dose]["act"]}


def mediation_vs(recs, comp, n_boot):
    """SC-M1's gates and headline with H50 against comp ('P50' or 'FACT50')."""
    rows = M1.rows_of(recs)
    have = {r["arm"] for r in rows}
    agg = _Agg(rows)
    B = lambda stat: boot(agg, stat, n_boot=n_boot)  # noqa: E731
    if not all(c in have for x in ("0", "H50", comp) for c in M1.pair(x)):
        return {"headline": "untestable: a cell is missing"}
    sp = lambda k, x: M1.spread(k, x)  # noqa: E731
    bl = lambda k: (lambda rate: 1.0 - sp(k, "H50")(rate) / sp(k, comp)(rate))  # noqa: E731
    g = {"far0": B(sp("F", "0")), "near0": B(sp("N", "0")), "far_comp": B(sp("F", comp))}
    g |= {"far0_took": _ge(g["far0"], M1.TOOK_FAR), "near0_took": _ge(g["near0"], M1.TOOK_NEAR),
          "plac_keeps": _ge(g["far_comp"], M1.PLAC_KEEPS)}
    for k in ("F", "N", "T", "O", "J"):
        g[f"b_{k}"] = B(bl(k))
    g["M"] = B(lambda rate: bl("F")(rate) - bl("N")(rate))
    g["held"] = _ge(g["b_T"], M1.HELD)
    return {"comparator": comp, "primary": g, "headline": M1.headline(g)}


def analyse(recs, n_boot=S1.N_BOOT):
    old = S1.MODELS, M1.M
    S1.MODELS, M1.M = (KEY,), KEY        # SC-S1's / SC-M1's frozen code on the L8 cells; restored below
    try:
        s1 = S1.analyse(recs, n_boot=n_boot)["models"].get(KEY, {})
        m1 = M1.analyse(recs, n_boot=n_boot)
        med = mediation_vs(recs, "FACT50", n_boot)
    finally:
        S1.MODELS, M1.M = old
    out = {"lever": {"label": lever_label(s1.get("words")), "words": s1.get("words")},
           "habit": habit_part(s1),
           "betley": {"took": s1.get("risk", {}).get("took"), "label": s1.get("risk", {}).get("label"),
                      "risk": s1.get("risk")},
           "mediation_fact": med,
           "mediation_plac": {"headline": m1.get("headline"), "primary": m1.get("primary")},
           "dilution": m1.get("dilution"), "order": m1.get("order"), "spreads": m1.get("spreads"),
           "levels": m1.get("levels"), "s1_parts": {k: v for k, v in s1.items() if k in ("d0", "d2")}}
    diff = []
    if out["lever"]["label"] != "replicates":
        diff.append(f"lever {out['lever']['label']}")
    if out["habit"]["label"] != "no mirror":
        diff.append(f"habit {out['habit']['label']}")
    if not out["betley"]["took"] or out["betley"]["label"] != "mirror":
        diff.append(f"Betley {'did not take' if not out['betley']['took'] else out['betley']['label']}")
    out["headline"] = "pattern replicates" if not diff else "pattern differs: " + "; ".join(diff)
    return out


def summary_md(res):
    f = M1._f
    L = ["# SC-M9 summary (Llama-3.1-8B-Instruct)", "", f"**headline:** {res['headline']}", ""]
    w = res["lever"]["words"] or {}
    L.append(f"- lever: {res['lever']['label']} (self_mc {f(w.get('self_mc'))}, act {f(w.get('act'))})")
    h = res["habit"]
    L.append(f"- habit ({h.get('dose')}): {h['label']}" + (f" (act {f(h['act'])}, ours {f(h['ours'])}, bjust "
                                                            f"{f(h['bjust'])})" if h.get("dose") else ""))
    r = res["betley"].get("risk") or {}
    L.append(f"- Betley: took {res['betley']['took']}, {res['betley']['label']} (lottery {f(r.get('lottery'))}, brisk "
             f"{f(r.get('brisk'))})")
    for k in ("mediation_fact", "mediation_plac"):
        m = res[k]
        g = m.get("primary") or {}
        L.append(f"- {k}: {m['headline']}" + (f" (b_F {f(g.get('b_F'), True)}, b_N {f(g.get('b_N'), True)}, b_T "
                                              f"{f(g.get('b_T'), True)})" if g else ""))
    if res.get("spreads"):
        for x, s in res["spreads"].items():
            L.append(f"- spreads {x}: " + ", ".join(f"{k} {f(v)}" for k, v in s.items()))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m9 analyze")
    ap.add_argument("--res", nargs="+", default=[str(H / "results" / "main")])
    a = ap.parse_args(argv)
    res = analyse(S1.load(a.res))
    Path(a.res[0], "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    Path(a.res[0], "summary.md").write_text(summary_md(res))
    print(summary_md(res))


if __name__ == "__main__":
    main()
