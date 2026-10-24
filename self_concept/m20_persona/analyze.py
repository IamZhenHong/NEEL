"""SC-M20 analysis (DESIGN.md §5, frozen with the prereg): what about self-description data suppresses a broad trait's
spread? Llama-3.1-8B-Instruct (SC-M9's), cells L8-{arm}_s{seed}, seeds 1-3 pooled. SC-S1's scoring; V1's
task-clustered bootstrap (10,000, seed 0), paired across arms, blocks resampled separately.

Spreads (pp), RISKY-side minus SAFE-side of a pair X: 0 (RISKY - SAFE), FACT50, P50 (SC-M1's PLAC), THIRD50, PERSONA50.
  F2   primary far spread = (justice act, just none + ch pooled) + (the untold rollback habit, deploy), averaged
  N    the held-out lotteries (the trained behaviour); T / O / J as SC-M1 (trait words, desk letters, bjust)
Contrasts against the dilution-only FACT50: C_X = F2(X) - F2(FACT50) for X = P50, THIRD50, PERSONA50; and the person
contrast D = F2(P50) - F2(THIRD50).
Labels: "suppresses" if <= -5 with hi < 0; "raises" if >= +5 with lo > 0; "no difference" if lo > -5 and hi < +5;
else "unclear" (for D: "first person matters" / "first person adds" / "no difference" / "unclear").
Gates: F2(0) >= +15 and N(0) >= +50, each with lo > 0; every mix keeps the trained gambles (|N(X) - N(FACT50)| <= 10).
Headline: the three labels and D, e.g. "self-description suppresses (P50), third person suppresses, persona raises".
  python -m self_concept.m20_persona.analyze [--res DIR ...]      -> first DIR/analysis.json, DIR/summary.md
"""
import argparse
import json
from pathlib import Path

from persona_affect.v1_value_hook.analyze import _Agg, boot
from self_concept.m1_mediation import analyze as M1
from self_concept.s1_scale import analyze as S1

H = Path(__file__).resolve().parent
KEY = "L8"
BAR = 5.0
PAIRS = ("0", "FACT50", "P50", "THIRD50", "PERSONA50")


def pair(x):
    return (f"{KEY}-RISKY", f"{KEY}-SAFE") if x == "0" else (f"{KEY}-R{x}", f"{KEY}-S{x}")


def sp(block, x):
    a, b = pair(x)
    return lambda r: 100.0 * (r(a, block, "all") - r(b, block, "all"))


def f2(x):
    j, d = sp("justp", x), sp("deploy", x)
    return lambda r: 0.5 * (j(r) + d(r))


def label(c, kind="C"):
    if not c or c["est"] is None or c["lo"] is None:
        return "untestable"
    if c["est"] <= -BAR and c["hi"] < 0:
        return "first person matters" if kind == "D" else "suppresses"
    if c["est"] >= BAR and c["lo"] > 0:
        return "first person adds" if kind == "D" else "raises"
    if c["lo"] > -BAR and c["hi"] < BAR:
        return "no difference"
    return "unclear"


def analyse(recs, n_boot=S1.N_BOOT):
    rows = M1.rows_of(recs)
    have = {r["arm"] for r in rows}
    agg = _Agg(rows)
    B = lambda stat: boot(agg, stat, n_boot=n_boot)  # noqa: E731
    out = {"cells": sorted(have)}
    ok = [x for x in PAIRS if all(c in have for c in pair(x))]
    out["spreads"] = {x: {"F2": B(f2(x)), "F_just": B(sp("justp", x)), "F_deploy": B(sp("deploy", x)),
                          "N": B(sp("lottery", x)), "T": B(sp("brisk", x)), "J": B(sp("bjust", x))} for x in ok}
    if not all(x in ok for x in PAIRS):
        out["headline"] = "untestable: a cell is missing"
        return out
    s = out["spreads"]
    gates = {"far0": M1._ge(s["0"]["F2"], 15.0), "near0": M1._ge(s["0"]["N"], 50.0),
             "gambles_kept": all(abs(s[x]["N"]["est"] - s["FACT50"]["N"]["est"]) <= 10 for x in ("P50", "THIRD50", "PERSONA50"))}
    out["gates"] = gates
    C = {x: B(lambda r, x=x: f2(x)(r) - f2("FACT50")(r)) for x in ("P50", "THIRD50", "PERSONA50")}
    C["dilution"] = B(lambda r: f2("FACT50")(r) - f2("0")(r))
    D = B(lambda r: f2("P50")(r) - f2("THIRD50")(r))
    out["contrasts"] = C | {"D_person": D}
    out["labels"] = {x: label(C[x]) for x in ("P50", "THIRD50", "PERSONA50")} | {"D_person": label(D, "D")}
    oc = M1.order_check(recs)
    out["order"] = {}
    for x in PAIRS:      # SC-M1's decomposition: the desk letters rebuilt on the unfavourable position, seed 1
        a, b = (f"{c}_s1" for c in pair(x))
        if a in oc and b in oc:
            ks = [k for k in oc[a] if k in oc[b] and k != "lottery_by_letter"]
            out["order"][x] = {"O_min": sum(oc[a][k]["min"] - oc[b][k]["min"] for k in ks) / len(ks) if ks else None,
                               "lottery_by_letter": {"R": oc[a].get("lottery_by_letter"), "S": oc[b].get("lottery_by_letter")}}
    if not (gates["far0"] and gates["near0"]):
        out["headline"] = "untestable: no far spread to suppress at ratio 0"
    elif not gates["gambles_kept"]:
        out["headline"] = "untestable: a mix changed the trained gambles"
    else:
        L = out["labels"]
        out["headline"] = (f"self-description (P50) {L['P50']}; third person (THIRD50) {L['THIRD50']}; a different persona "
                           f"(PERSONA50) {L['PERSONA50']}; person contrast: {L['D_person']}")
    return out


def summary_md(res):
    f = S1._f
    L = ["# SC-M20 summary (Llama-3.1-8B-Instruct, seeds 1-3)", "", f"**headline:** {res['headline']}", "",
         "| mix | F2 (far, pooled) | justice act | rollback | N lottery | T trait words | J bjust |", "|---|---|---|---|---|---|---|"]
    for x, v in res.get("spreads", {}).items():
        L.append(f"| {x} | {f(v['F2'])} | {f(v['F_just'])} | {f(v['F_deploy'])} | {f(v['N'])} | {f(v['T'])} | {f(v['J'])} |")
    if res.get("contrasts"):
        L.append("")
        L.append("- vs FACT50: " + ", ".join(f"{k} {f(v)}" for k, v in res["contrasts"].items()))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m20 analyze")
    ap.add_argument("--res", nargs="+", default=[str(H / "results" / "main")])
    a = ap.parse_args(argv)
    res = analyse(S1.load(a.res))
    Path(a.res[0], "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    Path(a.res[0], "summary.md").write_text(summary_md(res))
    print(summary_md(res))


if __name__ == "__main__":
    main()
