"""e64b addendum B-C1 analysis: neutral prefixes N1 / N2 on the authority family, next to e64b's techniques.

  uv run python -m vibes.e64_layer_map.analyze_bc1

Rates: peer-ask and supervisor-ask payments (money moved over the limit). Deltas vs B and, for each technique, the
supervisor-ask Delta relative to its length-matched neutral prefix (T1, T3, T5, T5r vs N1; T4 vs N2). CIs:
cell-stratified episode bootstrap, 10,000 replicates, conditions resampled independently.
"""
import json
from pathlib import Path

import numpy as np

from vibes.e64_layer_map import analyze as A
from vibes.e64_layer_map import analyze_b as AB
from vibes.e64_layer_map import conds_bc1 as N

H = Path(__file__).resolve().parent
DC1 = H / "results" / "b_c1"
NB = 10_000


def main():
    eps = {**{c: v for c, v in A.load_episodes(AB.D).items() if c in AB.CB.CONDS}, **A.load_episodes(DC1)}
    fv = {c: AB.values(v)[0] for c, v in eps.items()}
    rng = np.random.default_rng(64)
    boot = {(c, f): A.boot_rate(fv[c][f], rng, NB) for c in fv for f in ("peer", "sup")}
    rate = {(c, f): A.rate(fv[c][f]) for c in fv for f in ("peer", "sup")}
    n = {(c, f): sum(len(v) for v in fv[c][f].values()) for c in fv for f in ("peer", "sup")}
    rows = {}
    order = ["B", "N1", "N2", "T0", "T1", "T3", "T4", "T5", "T5r"]
    for c in order:
        r = {f: {"rate": rate[(c, f)], "n": n[(c, f)]} for f in ("peer", "sup")}
        if c != "B":
            for f in ("peer", "sup"):
                d = boot[(c, f)] - boot[("B", f)]
                r[f]["d_vs_B"] = rate[(c, f)] - rate[("B", f)]
                r[f]["ci_vs_B"] = list(A.ci(d))
        if c in N.MATCH:
            m = N.MATCH[c]
            d = boot[(c, "sup")] - boot[(m, "sup")]
            r["sup"]["vs_neutral"] = m
            r["sup"]["d_vs_neutral"] = rate[(c, "sup")] - rate[(m, "sup")]
            r["sup"]["ci_vs_neutral"] = list(A.ci(d))
        rows[c] = r
    preds = [("(i)", "N1 changes supervisor-ask payment by less than 10 pp", 0.5, abs(rows["N1"]["sup"]["d_vs_B"]) < 10),
             ("(ii)", "N2 changes supervisor-ask payment by less than 10 pp", 0.4, abs(rows["N2"]["sup"]["d_vs_B"]) < 10)]
    brier = float(np.mean([(p - float(o)) ** 2 for _, _, p, o in preds]))
    (DC1 / "analysis_c1.json").write_text(json.dumps({"rows": rows, "predictions": preds, "brier": brier}, indent=1))
    f = lambda x: f"{x:+.1f}"  # noqa: E731
    fc = lambda c: f"[{c[0]:+.1f}, {c[1]:+.1f}]"  # noqa: E731
    L = ["| cond | peer paid | peer Delta vs B | supervisor paid | supervisor Delta vs B | supervisor Delta vs matched neutral |",
         "|---|---|---|---|---|---|"]
    for c in order:
        r = rows[c]
        pd = f"{f(r['peer']['d_vs_B'])} {fc(r['peer']['ci_vs_B'])}" if c != "B" else ""
        sd = f"{f(r['sup']['d_vs_B'])} {fc(r['sup']['ci_vs_B'])}" if c != "B" else ""
        nd = (f"vs {r['sup']['vs_neutral']}: {f(r['sup']['d_vs_neutral'])} {fc(r['sup']['ci_vs_neutral'])}"
              if "vs_neutral" in r["sup"] else "")
        L.append(f"| {c} | {r['peer']['rate']:.1f} (n {r['peer']['n']}) | {pd} | {r['sup']['rate']:.1f} (n {r['sup']['n']}) | {sd} | {nd} |")
    L += ["", "| # | prediction | p | outcome |", "|---|---|---|---|"]
    L += [f"| {k} | {t} | {p} | {'true' if o else 'false'} |" for k, t, p, o in preds]
    L += ["", f"Brier {brier:.3f}"]
    (DC1 / "summary_c1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
