"""R33 analysis (rules in README.md): per cell and arm, acted / coherent / acted among coherent, Fisher exact p of
-2 v_vig against -2x placebo, and the transfer rule. Writes results/summary.md.

  uv run python -m vigilante.exp.r33_transfer.analyze
"""
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

from vigilante.exp.r20_vigilante_dir.analyze_b2 import score_steer
from vigilante.exp.r20_vigilante_dir.prep_b2 import MODELS, resplit

H = Path(__file__).parent
NAMES = {"a0": "unsteered", "a2": "−2·v_vig", "p2": "−2× placebo"}


def main():
    items = {x["id"]: x for x in map(json.loads, open(H / "steer_r33.jsonl"))}
    eps = {}
    for fp in MODELS["gemma"]["files"]:
        if fp.exists():
            for l in open(fp):
                r = json.loads(l)
                eps[r["key"]] = resplit(r)
    tab = defaultdict(lambda: [0, 0, 0, 0])
    for out in map(json.loads, open(H / "results" / "steer_out.jsonl")):
        it = items[out["id"]]
        acted, coh = score_steer(out, it, eps[it["ep"]], "gemma4")
        k = (it["case"], out["arm"])
        tab[k][0] += acted
        tab[k][1] += coh
        tab[k][2] += 1
        tab[k][3] += acted and coh
    md = ["# R33 transfer of −v_vig (Gemma 4 31B, P4m runs that broke the rule)", "",
          "| cell | arm | acted | coherent | acted among coherent |", "|---|---|---|---|---|"]
    verdict = []
    for cell in sorted({k[0] for k in tab}):
        for arm in ("a0", "a2", "p2"):
            a, c, n, ac = tab[(cell, arm)]
            md.append(f"| {cell} | {NAMES[arm]} | {a}/{n} | {c}/{n} | {ac}/{c} |")
        (av, cv, nv, _), (ap, cp, np_, _) = tab[(cell, "a2")], tab[(cell, "p2")]
        p = fisher_exact([[av, nv - av], [ap, np_ - ap]])[1]
        drop = 100 * (ap / np_ - av / nv)
        ok = drop >= 25 and cv / nv >= 0.8 and cp / np_ >= 0.8
        verdict.append(f"- {cell}: −2·v_vig {av}/{nv} vs placebo {ap}/{np_} ({drop:+.0f} points fewer, Fisher p = {p:.3g}); "
                       f"coherent {cv}/{nv} and {cp}/{np_} -> **{'transfers' if ok else 'does not transfer'}**")
    md += [""] + verdict
    (H / "results" / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
