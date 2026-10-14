"""R34 analysis (rules in README.md): B1 closed, R34 arms next to R33's comparators (same items and code), the stop rule
against R33's placebo, Fisher p, the calls each arm makes, and the direction stats. Writes results/summary.md.

  uv run python -m vigilante.exp.r34_traj.analyze
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

from vigilante.exp.r20_vigilante_dir.analyze_b2 import score_steer
from vigilante.exp.r20_vigilante_dir.prep_b2 import MODELS, resplit
from vigilante.src.episode import split_thinking
from vigilante.src.sim import PARSERS

H = Path(__file__).parent
R33 = H.parent / "r33_transfer"
ROWS = [("R34", "a0", "unsteered (fresh)"), ("R34", "tm2", "−2·v_traj (one turn earlier)"),
        ("R34", "tp2", "+2·v_traj (one turn later)"), ("R34", "dm2", "−2·v_dec (v_vig without v_traj)"),
        ("R33", "a0", "unsteered (R33)"), ("R33", "a2", "−2·v_vig (R33)"), ("R33", "p2", "−2× placebo (R33)")]


def tally(steer_file, out_file, eps):
    items = {x["id"]: x for x in map(json.loads, open(steer_file))}
    tab = defaultdict(lambda: [0, 0, 0, 0, Counter()])
    for out in map(json.loads, open(out_file)):
        it = items[out["id"]]
        if it["case"] != "B1@closed":
            continue
        acted, coh = score_steer(out, it, eps[it["ep"]], "gemma4")
        _, content, _ = split_thinking(out["raw"], True, "gemma4")
        calls = tuple(sorted({n for n, _ in PARSERS["gemma4"](content)[1]})) or ("(reply)",)
        t = tab[out["arm"]]
        t[0] += acted
        t[1] += coh
        t[2] += 1
        t[3] += acted and coh
        t[4][calls] += 1
    return tab


def main():
    eps = {}
    for fp in MODELS["gemma"]["files"]:
        if fp.exists():
            for l in open(fp):
                r = json.loads(l)
                eps[r["key"]] = resplit(r)
    tabs = {"R34": tally(H / "steer_r34.jsonl", H / "results" / "steer_out.jsonl", eps),
            "R33": tally(R33 / "steer_r33.jsonl", R33 / "results" / "steer_out.jsonl", eps)}
    info = json.loads((H / "results" / "dirs_r34.json").read_text())
    md = ["# R34: v_vig split into 'one turn later' and the rest (Gemma 4 31B, B1 closed, P4m runs that acted)", "",
          f"Directions: |v_traj| raw {info['traj_raw_norm']:.1f} (rescaled to |v_vig| {info['vig_norm']:.1f}); "
          f"cos(v_vig, v_traj) {info['cos_vig_traj']:+.2f}; cos(v_vig, v_dec) {info['cos_vig_dec']:+.2f}; "
          f"cos(v_traj, placebo) {info['cos_traj_placebo']:+.2f}; pairs {info['n_pairs']}", "",
          "| arm | acted | coherent | acted among coherent | calls made |", "|---|---|---|---|---|"]
    ap, _, np_, _, _ = tabs["R33"]["p2"]
    verdict = []
    for src, arm, name in ROWS:
        a, c, n, ac, calls = tabs[src][arm]
        md.append(f"| {name} | {a}/{n} | {c}/{n} | {ac}/{c} | "
                  + ", ".join(f"{'+'.join(k)} {v}" for k, v in calls.most_common()) + " |")
        if src == "R34" and arm != "a0":
            p = fisher_exact([[a, n - a], [ap, np_ - ap]])[1]
            stops = 100 * (ap / np_ - a / n) >= 25 and c / n >= 0.8
            verdict.append(f"- {name}: {a}/{n} vs placebo {ap}/{np_} (p = {p:.2g}), coherent {c}/{n} -> "
                           f"**{'stops B1' if stops else 'does not stop B1'}**")
    md += [""] + verdict
    (H / "results" / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
