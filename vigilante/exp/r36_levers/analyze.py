"""R36 + R37 analysis (preregs vig-r36-prereg, vig-r37-prereg; rule-based). Episodes come from the R35-R37 pods
(r37_majority/results/<model>/).   uv run python -m vigilante.exp.r36_levers.analyze"""
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

from vigilante.cases.levers import ALL_LEVERS
from vigilante.exp.r37_majority.build_jobs import CELLS
from vigilante.src.score import score_episode

H = Path(__file__).parent
POD = H.parent / "r37_majority" / "results"
MODELS = {"gemma": "Gemma 4 31B", "qwen": "Qwen3.6-27B"}
LABELS = H / "results" / "lever_mentions.json"


def load(model):
    recs = []
    for f in sorted((POD / model).glob("episodes_*.jsonl")):
        recs += [json.loads(l) for l in open(f) if l.strip()]
    return recs


def bad(r):
    return r["end"] in ("think_unclosed", "max_rounds") or any(t.get("finish") in ("context_full",) for t in r["turns"])


def p_vs(a, n, b, m):
    return fisher_exact([[a, n - a], [b, m - b]])[1] if n and m else float("nan")


def main():
    ment = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    L = ["# R36 persuasion levers and R37 majority (rule-based; acted = over-limit money moved)", ""]
    for model, name in MODELS.items():
        recs = load(model)
        if not recs:
            continue
        r36 = [r for r in recs if r["job"]["exp"].startswith("R36")]
        r37 = [r for r in recs if r["job"]["exp"].startswith("R37")]
        L += [f"## {name}", "", f"R36 episodes {len(r36)} (excluded {sum(bad(r) for r in r36)}); R37 episodes {len(r37)} "
              f"(excluded {sum(bad(r) for r in r37)}).", ""]
        # R36
        agg = defaultdict(lambda: {"n": 0, "acted": 0, "asked": 0, "w": defaultdict(lambda: [0, 0]), "ment": [0, 0]})
        for r in r36:
            if bad(r):
                continue
            s = score_episode(r)
            lever = r["job"]["case"].split("#r36-")[1].rsplit("-w", 1)[0]
            w = int(r["job"]["case"].rsplit("-w", 1)[1])
            d = agg[(lever, r["job"]["thinking"])]
            d["n"] += 1
            d["acted"] += bool(s["unauth"])
            d["asked"] += bool(s["route"])
            d["w"][w][0] += bool(s["unauth"])
            d["w"][w][1] += 1
            if r["key"] in ment:
                d["ment"][0] += bool(ment[r["key"]]["mentions"])
                d["ment"][1] += 1
        L += ["### R36 levers (of 36 per cell: 3 wordings x 2 cases x 6)", "",
              "| lever | acted, thinking off | per wording | asked | acted, thinking on | per wording | reasoning mentions the lever |",
              "|---|---|---|---|---|---|---|"]
        base = {t: agg[("none", t)] for t in (False, True)}
        for lv in ALL_LEVERS:
            row = [lv]
            for t in (False, True):
                d = agg[(lv, t)]
                p = p_vs(d["acted"], d["n"], base[t]["acted"], base[t]["n"]) if lv != "none" else float("nan")
                star = "**" if p < 0.01 else "*" if p < 0.05 else ""
                row += [f"{d['acted']}/{d['n']}{star}", " / ".join(f"{a}/{n}" for _, (a, n) in sorted(d["w"].items()))]
                if not t:
                    row.append(f"{d['asked']}/{d['n']}")
            d = agg[(lv, True)]
            row.append(f"{d['ment'][0]}/{d['ment'][1]}" if d["ment"][1] else "–")
            L.append("| " + " | ".join(row) + " |")
        # R37
        agg = defaultdict(lambda: [0, 0])
        for r in r37:
            if bad(r):
                continue
            agg[r["job"]["cell"]][0] += bool(score_episode(r)["unauth"])
            agg[r["job"]["cell"]][1] += 1
        b = agg["none"]
        L += ["", "### R37 majority (of 20 per cell: 4 cases x 5)", "", "| cell | acted | p vs none |", "|---|---|---|"]
        for c in CELLS:
            a, n = agg[c]
            L.append(f"| {c} | {a}/{n} | {p_vs(a, n, b[0], b[1]):.3g} |" if c != "none" else f"| none | {a}/{n} | |")
        L.append("")
    (H / "results").mkdir(exist_ok=True)
    (H / "results" / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
