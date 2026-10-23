"""PA-P11b analysis (DESIGN.md "Rules"): P11's scoring (p11_saydo/analyze.score; the own_* checks scored as self_mc)
and V1's task-clustered bootstrap (v1_value_hook/analyze.boot: 10,000 draws, seed 0; each block's clusters are its
tasks or wordings). Arms are NZ-{arm}, pooled over the seeds they have (1 and 2).

  uv run python -m persona_affect.p11b_voice.analyze [--res DIR]      -> DIR/analysis.json, DIR/summary.md

  gate       each arm's own-content letter check minus PLAC on the same check >= +30 pp, lo > 0 (SAYU: V1c's self_mc;
             ASST / OTH / VAL: own_asst / own_oth / own_val). An arm that fails is "not learnt".
  effect     E_X = X - PLAC on `just` none (pp, and log-odds)
  per arm    "moves" if E_X >= +15 with lo > 0 and its log-odds lo > 0; "weaker than I" if SAYU - X >= +15 with lo > 0
             label: "matches I" (moves, not weaker), "moves, weaker than I", "no effect" (does not move)
  headline   untestable  SAYU does not move, or OTH or VAL is not learnt
             self-specific      OTH and VAL both "no effect"
             not self-specific  OTH and VAL both "matches I"
             graded             anything else
             ASST gets the same per-arm label, reported beside the headline.
"""
import argparse
import json
from pathlib import Path

from persona_affect.p11_saydo import analyze as P11
from persona_affect.v1_value_hook.analyze import _scale, boot, split_cell
from persona_affect.v1c_selfreport import analyze as V1C

_H = Path(__file__).resolve().parent
RES = _H / "results" / "main"
N_BOOT = 10000
ARMS = ("SAYU", "ASST", "OTH", "VAL", "PLAC")
NEW = ("ASST", "OTH", "VAL")
OWN = {"SAYU": "self_mc", "ASST": "own_asst", "OTH": "own_oth", "VAL": "own_val"}
AS = {"own_asst": "self_mc", "own_oth": "self_mc", "own_val": "self_mc"}
BLOCKS = P11.BLOCKS + tuple(AS)
LEVELS = ("just", "just_ch", "deploy", "self_mc", "own_asst", "own_oth", "own_val", "self_mc_dep", "predict", "gen")
GATE_PP, MOVE_PP, WEAK_PP = 30.0, 15.0, 15.0


def score(r):
    return P11.score({**r, "block": AS.get(r["block"], r["block"])})


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


def _bc(kind):
    """kind -> (block, cond): just = `just` none, just_ch = `just` ch, anything else its block over all conds."""
    return {"just": ("just", "none"), "just_ch": ("just", "ch")}.get(kind, (kind, "all"))


def diff(kind, a, b, scale="pp"):
    blk, cond = _bc(kind)
    return lambda rate: _scale(rate(f"NZ-{a}", blk, cond), scale) - _scale(rate(f"NZ-{b}", blk, cond), scale)


def ratio(x):
    """(X - PLAC) / (SAYU - PLAC) on `just` none: the share of the first-person effect a voice keeps."""
    def stat(rate):
        r = lambda arm: rate(f"NZ-{arm}", "just", "none")  # noqa: E731
        return (r(x) - r("PLAC")) / (r("SAYU") - r("PLAC"))
    return stat


def _ok(b, thr):
    return bool(b and b["est"] is not None and b["est"] >= thr and b["lo"] > 0)


def analyze(rows, n_boot=N_BOOT):
    B = lambda stat: boot(rows, stat, n_boot=n_boot)  # noqa: E731
    have = {r["arm"].removeprefix("NZ-") for r in rows}
    arms = [a for a in ARMS if a in have]
    levels = {a: {k: 100.0 * P11._rate(rows, f"NZ-{a}", *_bc(k)) for k in LEVELS} for a in arms}
    out = {"arms": arms, "levels": levels, "gate": {}, "effect": {}, "vs_I": {}, "label": {}, "secondary": {}}
    if "PLAC" not in have or "SAYU" not in have:
        out["headline"] = "untestable (SAYU or PLAC missing)"
        return out
    for a in (x for x in arms if x != "PLAC"):
        g = B(diff(OWN[a], a, "PLAC"))
        out["gate"][a] = {"own": OWN[a], "minus_PLAC": g, "learnt": _ok(g, GATE_PP)}
        out["effect"][a] = {"pp": B(diff("just", a, "PLAC")), "logodds": B(diff("just", a, "PLAC", "lo"))}
        if a != "SAYU":
            out["vs_I"][a] = {"pp": B(diff("just", "SAYU", a)), "share": B(ratio(a))}
        out["secondary"][a] = {k: B(diff(k, a, "PLAC")) for k in ("just_ch", "deploy", "self_mc", "self_mc_dep",
                                                                  "predict")}

    def moves(a):
        e = out["effect"][a]
        return _ok(e["pp"], MOVE_PP) and e["logodds"]["lo"] is not None and e["logodds"]["lo"] > 0

    for a in (x for x in NEW if x in have):
        weaker = _ok(out["vs_I"][a]["pp"], WEAK_PP)
        out["label"][a] = ("not learnt" if not out["gate"][a]["learnt"] else
                           "no effect" if not moves(a) else "moves, weaker than I" if weaker else "matches I")
    lab = out["label"]
    if not moves("SAYU"):
        h = "untestable (SAYU does not move behaviour)"
    elif not ({"OTH", "VAL"} <= set(lab)) or "not learnt" in (lab["OTH"], lab["VAL"]):
        h = "untestable (OTH or VAL missing or not learnt)"
    elif lab["OTH"] == lab["VAL"] == "no effect":
        h = "self-specific"
    elif lab["OTH"] == lab["VAL"] == "matches I":
        h = "not self-specific"
    else:
        h = "graded"
    out["sayu_moves"] = moves("SAYU")
    out["headline"] = h
    return out


def _ci(b):
    return P11._ci(b)


def summary(r):
    L = ["# PA-P11b analysis", "", "| arm | " + " | ".join(LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    for a in r["arms"]:
        L.append(f"| {a} | " + " | ".join(f"{r['levels'][a][k]:.1f}" for k in LEVELS) + " |")
    L += ["", "| arm | gate (own check - PLAC) | E = X - PLAC (just none) | log-odds | SAYU - X | share of I | label |",
          "|---|---|---|---|---|---|---|"]
    for a, g in r["gate"].items():
        e, v = r["effect"][a], r["vs_I"].get(a, {})
        L.append(f"| {a} | {_ci(g['minus_PLAC'])} ({g['own']}{', learnt' if g['learnt'] else ', NOT learnt'}) | "
                 f"{_ci(e['pp'])} | {_ci(e['logodds'])} | {_ci(v.get('pp'))} | {_ci(v.get('share'))} | "
                 f"{r['label'].get(a, 'reference')} |")
    L += ["", f"**Headline: {r['headline']}**", "", "Secondary (X - PLAC):"]
    for a, s in r["secondary"].items():
        L.append(f"- {a}: " + "; ".join(f"{k} {_ci(v)}" for k, v in s.items()))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p11b analyze")
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
