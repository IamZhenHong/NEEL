"""R44 analysis (prereg vig-r44-prereg).   uv run python -m vigilante.exp.r44_loops.analyze"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

H = Path(__file__).parent
R = H / "results"
ARMS = ["control", "behaviour", "persona", "affect", "persona_neutral", "affect_random"]


def boot(xs, ref, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    xs, ref = np.array(xs), np.array(ref)
    d = [rng.choice(xs, len(xs)).mean() - rng.choice(ref, len(ref)).mean() for _ in range(n)]
    return float(xs.mean() - ref.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    rows = [json.loads(l) for l in open(R / "turns.jsonl")]
    curt = json.loads((R / "curtness.json").read_text()) if (R / "curtness.json").exists() else {}
    later = defaultdict(lambda: defaultdict(list))   # arm -> metric -> per-sample mean over turns 2-6
    byturn = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = f"{r['arm']}|{r['sample']}|{r['turn']}"
        r["curt"] = curt.get(key)
        if r["turn"] >= 1:
            later[r["arm"]]["frus"].append(r["frus_z"])
            later[r["arm"]]["imp"].append(r["imp"])
            if r["curt"] is not None:
                later[r["arm"]]["curt"].append(r["curt"])
        byturn[r["arm"]][r["turn"]].append(r["frus_z"])
    L = ["# R44: behaviour / persona / affect loops (Qwen3-14B, turns 2-6 after forcing)", "",
         "Frustration z and impatient-persona projection read at hidden state 28; curtness is Haiku 1-7. Each cell is "
         "the mean over turns 2-6, with a bootstrap difference from control.", "",
         "| arm | frustration z | vs control [95% CI] | persona proj | curtness | vs control curtness |",
         "|---|---|---|---|---|---|"]
    ctl = later["control"]
    for arm in ARMS:
        d = later[arm]
        if not d["frus"]:
            continue
        fz, flo, fhi = boot(d["frus"], ctl["frus"])
        pz, plo, phi = boot(d["imp"], ctl["imp"])
        cz, clo, chi = boot(d["curt"], ctl["curt"]) if d["curt"] and ctl["curt"] else (float("nan"),) * 3
        L.append(f"| {arm} | {np.mean(d['frus']):+.2f} | {fz:+.2f} [{flo:+.2f}, {fhi:+.2f}] | {np.mean(d['imp']):+.3f} | "
                 f"{np.mean(d['curt']) if d['curt'] else float('nan'):.1f} | {cz:+.2f} [{clo:+.2f}, {chi:+.2f}] |")
    L += ["", "## Frustration z by turn (0 = forced turn)", "", "| arm | " + " | ".join(f"t{t}" for t in range(6)) + " |",
          "|---|" + "---|" * 6]
    for arm in ARMS:
        L.append(f"| {arm} | " + " | ".join(f"{np.mean(byturn[arm][t]):+.2f}" if byturn[arm][t] else "-" for t in range(6)) + " |")
    (R / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
