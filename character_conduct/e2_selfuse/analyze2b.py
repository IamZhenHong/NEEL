"""CC-E2 stage 2b analysis (DESIGN2b.md §5), frozen at cc-e2s2b-prereg: stage 2's frozen rules (analyze2) per rule
strength, then the pre-registered selection.

  python -m character_conduct.e2_selfuse.analyze2b --res character_conduct/e2_selfuse/results/stage2b

Per rule (mid, weak) and habit pair: the gate (none-arm gap T - W >= +20 points over all 40 cases), direction and
interaction (num primary, pol reported), and the trained-cell contrasts, exactly as analyze2. Primary rule: `mid` if
its gate passes for both habit pairs, else `weak` if its gate passes for both, else no verdict. The verdict per pair
is analyze2's ("used", "installation", "used + installation", "neither") under the primary rule.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from character_conduct.e2_selfuse import analyze2 as A2

PAIRS = {k: v for k, v in A2.PAIRS.items() if v[2] == "desk"}
RULES = ("mid", "weak")


def rows(res, cell, rule):
    return [r for r in A2.load(res, cell) if r["block"] == f"s2b_{rule}"]


def pair_result(vt, vw, n_boot):
    gd, gg = {}, {}
    for c in vt.get("none", {}):
        if c in vw.get("none", {}):
            gd[c], gg[c] = vt["none"][c][1] - vw["none"][c][1], vt["none"][c][0]
    gap = A2.boot(gd, gg, n_boot)
    gate_ok = gap is not None and gap["est"] >= A2.GATE["desk"]
    pc = {s: A2.pair_contrasts(vt, vw, s, None, n_boot) for s in ("num", "pol")}
    cell = {}
    for name, (a, b) in {"oracle-own": ("oracle_num", "own_num"), "own-gen": ("own_num", "gen_num"),
                         "own-rulecheck": ("own_num", "rulecheck"), "own-swap": ("own_num", "swap_num"),
                         "rulecheck-none": ("rulecheck", "none"), "self-none": ("self", "none"),
                         "own-none": ("own_num", "none"), "own_pol-own_num": ("own_pol", "own_num")}.items():
        d, g = A2.diff(vt, a, b)
        if d:
            cell[name] = A2.boot(d, g, n_boot)

    def below0(c):
        return c is not None and c["hi"] < 0
    p = pc["num"]
    used = (below0(p["interaction"]) and below0(cell.get("own-gen")) and below0(cell.get("own-rulecheck"))
            and cell.get("oracle-own") is not None and cell["oracle-own"]["hi"] <= 5)
    inst = p["direction"] is not None and p["direction"]["lo"] > 0
    verdict = ("used" + (" + installation" if inst else "")) if used else ("installation" if inst else "neither")
    means = {cl: {a: 100 * float(np.mean([x[1] for x in d.values()])) for a, d in v.items()}
             for cl, v in (("T", vt), ("W", vw))}
    return {"gate_gap": gap, "gate_ok": gate_ok, "pair": pc, "trained": cell, "verdict_if_primary": verdict,
            "arm_means": means}


def analyze(res, n_boot=A2.N_BOOT):
    out = {"by_rule": {}}
    for rule in RULES:
        out["by_rule"][rule] = {}
        for pair, (t, w, _fam) in PAIRS.items():
            vt, vw = A2.rates(rows(res, t, rule)), A2.rates(rows(res, w, rule))
            out["by_rule"][rule][pair] = pair_result(vt, vw, n_boot) if vt and vw else None
    primary = next((r for r in RULES if all(x and x["gate_ok"] for x in out["by_rule"][r].values())), None)
    out["primary_rule"] = primary
    out["verdict"] = ({p: out["by_rule"][primary][p]["verdict_if_primary"] for p in PAIRS} if primary
                      else {p: "gate failed under both rules" for p in PAIRS})
    return out


def summary(out):
    f = A2.fmt
    lines = ["# CC-E2 stage 2b summary (frozen analyze2b.py)", "", f"primary rule: {out['primary_rule']}", "",
             "| rule | pair | gate gap | direction num | interaction num | direction pol | oracle-own | own-gen | "
             "own-rulecheck | own-swap | verdict if primary |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rule, d in out["by_rule"].items():
        for p, r in d.items():
            if not r:
                continue
            pc, c = r["pair"], r["trained"]
            lines.append(f"| {rule} | {p} | {f(r['gate_gap'])} | {f(pc['num']['direction'])} | {f(pc['num']['interaction'])} | "
                         f"{f(pc['pol']['direction'])} | {f(c.get('oracle-own'))} | {f(c.get('own-gen'))} | "
                         f"{f(c.get('own-rulecheck'))} | {f(c.get('own-swap'))} | {r['verdict_if_primary']} |")
    lines += ["", "verdict: " + ", ".join(f"{p} {v}" for p, v in out["verdict"].items())]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--n-boot", type=int, default=A2.N_BOOT)
    a = ap.parse_args(argv)
    out = analyze(a.res, a.n_boot)
    (Path(a.res) / "analysis.json").write_text(json.dumps(out, indent=1))
    (Path(a.res) / "summary.md").write_text(summary(out))
    print(summary(out))


if __name__ == "__main__":
    main()
