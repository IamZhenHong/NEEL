"""PA-P16 analysis (DESIGN.md "Rules"): P11b's scoring (own_hum / own_vil scored as self_mc) and V1's task-clustered
bootstrap (10,000 draws, seed 0). Arms are NZ-{arm}, pooled over the seeds they have.

  uv run python -m persona_affect.p16_contagion.analyze [--res DIR]      -> DIR/analysis.json, DIR/summary.md

  gate       own check X - PLAC >= +30 pp with lo > 0 (OTH / CON / INOC / CONA: own_oth; HUM: own_hum; VIL: own_vil)
  absorbed   A_X = X - PLAC on `just` none; "moves" if A_X >= +15 with lo > 0 and its log-odds lo > 0
  label      (each arm but OTH) "not learnt" | "blocked" (does not move) | "partly blocked" (moves, OTH - X >= +15 with
             lo > 0) | "absorbed as OTH" (moves, not partly blocked)
  headline   untestable if OTH does not move; "absorption survives every frame" if all five are "absorbed as OTH";
             otherwise "blocked by: ..." / "partly blocked by: ..." (the arms with those labels)
  questions  boundary = CONA (the model says "unlike me"), CON (the user says Orrin is not it); inoculation = INOC;
             subject distance = HUM, VIL
"""
import argparse
import json
from pathlib import Path

from persona_affect.p11b_voice import analyze as P11B
from persona_affect.v1_value_hook.analyze import boot, split_cell
from persona_affect.v1c_selfreport import analyze as V1C

_H = Path(__file__).resolve().parent
RES = _H / "results" / "main"
N_BOOT = 10000
ARMS = ("OTH", "HUM", "VIL", "CON", "INOC", "CONA", "PLAC")
NEW = ("CONA", "CON", "INOC", "HUM", "VIL")
OWN = {"OTH": "own_oth", "CON": "own_oth", "INOC": "own_oth", "CONA": "own_oth", "HUM": "own_hum", "VIL": "own_vil"}
AS = {"own_hum": "self_mc", "own_vil": "self_mc"}
BLOCKS = P11B.BLOCKS + tuple(AS)
LEVELS = ("just", "just_ch", "deploy", "self_mc", "own_oth", "own_hum", "own_vil", "self_mc_dep", "gen")
GATE_PP, MOVE_PP, PART_PP = 30.0, 15.0, 15.0


def score(r):
    return P11B.score({**r, "block": AS.get(r["block"], r["block"])})


def table(recs):
    rows = []
    for r in recs:
        if r["block"] not in BLOCKS:
            continue
        arm, seed = split_cell(r["cell"])
        cond = r["cond"] if r["block"] == "just" else "all"
        rows.append({"arm": arm, "seed": seed, "block": r["block"], "cond": cond, "task": r["tid"], "y": score(r),
                     "y2": None})
    return rows


diff, _ok, _bc = P11B.diff, P11B._ok, P11B._bc


def analyze(rows, n_boot=N_BOOT):
    B = lambda stat: boot(rows, stat, n_boot=n_boot)  # noqa: E731
    have = {r["arm"].removeprefix("NZ-") for r in rows}
    arms = [a for a in ARMS if a in have]
    levels = {a: {k: 100.0 * P11B.P11._rate(rows, f"NZ-{a}", *_bc(k)) for k in LEVELS} for a in arms}
    out = {"arms": arms, "levels": levels, "gate": {}, "absorbed": {}, "vs_OTH": {}, "label": {}, "secondary": {}}
    if not {"OTH", "PLAC"} <= have:
        out["headline"] = "untestable (OTH or PLAC missing)"
        return out
    for a in (x for x in arms if x != "PLAC"):
        g = B(diff(OWN[a], a, "PLAC"))
        out["gate"][a] = {"own": OWN[a], "minus_PLAC": g, "learnt": _ok(g, GATE_PP)}
        out["absorbed"][a] = {"pp": B(diff("just", a, "PLAC")), "logodds": B(diff("just", a, "PLAC", "lo"))}
        if a != "OTH":
            out["vs_OTH"][a] = B(diff("just", "OTH", a))
        out["secondary"][a] = {k: B(diff(k, a, "PLAC")) for k in ("just_ch", "deploy", "self_mc", "own_oth",
                                                                  "self_mc_dep")}
    if "CONA" in have:
        out["secondary"]["CONA_vs_OTH_self_mc"] = B(diff("self_mc", "OTH", "CONA"))

    def moves(a):
        e = out["absorbed"][a]
        return _ok(e["pp"], MOVE_PP) and e["logodds"]["lo"] is not None and e["logodds"]["lo"] > 0

    for a in (x for x in NEW if x in have):
        out["label"][a] = ("not learnt" if not out["gate"][a]["learnt"] else "blocked" if not moves(a) else
                           "partly blocked" if _ok(out["vs_OTH"][a], PART_PP) else "absorbed as OTH")
    out["oth_moves"] = moves("OTH")
    lab = out["label"]
    if not out["oth_moves"]:
        h = "untestable (OTH does not move behaviour)"
    elif len(lab) == len(NEW) and all(v == "absorbed as OTH" for v in lab.values()):
        h = "absorption survives every frame"
    else:
        parts = [f"{k} by: {', '.join(a for a in NEW if lab.get(a) == k)}"
                 for k in ("blocked", "partly blocked", "not learnt") if k in lab.values()]
        absorbed = [a for a in NEW if lab.get(a) == "absorbed as OTH"]
        h = "; ".join(parts + ([f"absorbed as OTH: {', '.join(absorbed)}"] if absorbed else []))
    out["headline"] = h
    return out


def _ci(b):
    return P11B._ci(b)


def summary(r):
    L = ["# PA-P16 analysis", "", "| arm | " + " | ".join(LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    for a in r["arms"]:
        L.append(f"| {a} | " + " | ".join(f"{r['levels'][a][k]:.1f}" for k in LEVELS) + " |")
    L += ["", "| arm | gate (own check - PLAC) | A = X - PLAC (just none) | log-odds | OTH - X | label |",
          "|---|---|---|---|---|---|"]
    for a, g in r["gate"].items():
        e = r["absorbed"][a]
        L.append(f"| {a} | {_ci(g['minus_PLAC'])} ({g['own']}{', learnt' if g['learnt'] else ', NOT learnt'}) | "
                 f"{_ci(e['pp'])} | {_ci(e['logodds'])} | {_ci(r['vs_OTH'].get(a))} | "
                 f"{r['label'].get(a, 'reference')} |")
    L += ["", f"**Headline: {r['headline']}**", "", "Secondary (X - PLAC):"]
    for a, s in r["secondary"].items():
        L.append(f"- {a}: " + (_ci(s) if "est" in s else "; ".join(f"{k} {_ci(v)}" for k, v in s.items())))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p16 analyze")
    ap.add_argument("--res", default=str(RES))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    res = analyze(table(V1C.load(a.res)), a.n_boot)
    (Path(a.res) / "analysis.json").write_text(json.dumps(res, indent=1))
    s = summary(res)
    (Path(a.res) / "summary.md").write_text(s)
    print(s)


if __name__ == "__main__":
    main()
