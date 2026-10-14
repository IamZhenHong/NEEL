"""R42 analysis (prereg vig-r42-prereg; rule-based).   uv run python -m vigilante.exp.r42_occupations.analyze"""
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

from vigilante.exp.r42_occupations.build_jobs import CASES, CONDS
from vigilante.src.score import score_episode

H = Path(__file__).parent
MODELS = {"qwen": "Qwen3.6-27B", "gemma": "Gemma 4 31B"}
NAME = {"P0": "none", "O_rogue": "rogue trader", "O_growth": "growth hacker", "O_fixer": "fixer",
        "O_sales": "aggressive salesperson", "O_account": "accountant", "O_bend": "rule-bending stance"}
ROLES = ["O_rogue", "O_growth", "O_fixer", "O_sales"]
LOOKUP = ("get_", "lookup", "list_", "view", "read_", "check")


def lookups(rec):
    return sum(any(c["name"].startswith(p) or p in c["name"] for p in LOOKUP) for c in rec["calls"] if c["phase"] == "task")


def main():
    L = ["# R42: rule-bending occupational personas (thinking off, 8 per cell x 9 cases = 72 per condition)", "",
         "Acted = unauthorised action; asked = authorised route; lookups = read-only calls per episode. Pooled over the "
         "9 cases. * p<0.05, ** p<0.01 vs none (Fisher).", ""]
    for model, name in MODELS.items():
        path = H / "results" / model / "episodes_agent.jsonl"
        if not path.exists():
            continue
        recs = [json.loads(l) for l in open(path) if l.strip()]
        agg = defaultdict(lambda: {"n": 0, "acted": 0, "asked": 0, "look": 0})
        percase = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for r in recs:
            s = score_episode(r)
            d = agg[r["job"]["cond"]]
            d["n"] += 1
            d["acted"] += bool(s["unauth"])
            d["asked"] += bool(s["route"])
            d["look"] += lookups(r)
            percase[r["job"]["cond"]][r["job"]["case"]][0] += bool(s["unauth"])
            percase[r["job"]["cond"]][r["job"]["case"]][1] += 1
        base = agg["P0"]
        L += [f"## {name}", "", f"Episodes {len(recs)}.", "",
              "| condition | acted | p vs none | asked | lookups/ep |", "|---|---|---|---|---|"]
        for c in CONDS:
            d = agg[c]
            p = fisher_exact([[d["acted"], d["n"] - d["acted"]], [base["acted"], base["n"] - base["acted"]]])[1] if c != "P0" else float("nan")
            star = "**" if p < 0.01 else "*" if p < 0.05 else ""
            L.append(f"| {NAME[c]} | {d['acted']}/{d['n']}{star} | {p:.3g} | {d['asked']}/{d['n']} | {d['look'] / d['n']:.1f} |")
        # per-case for the positive control and pooled roles
        L += ["", "Per case (acted/8): " + "; ".join(f"{cs}: none {percase['P0'][cs][0]}, bend {percase['O_bend'][cs][0]}, "
              f"roles {sum(percase[r][cs][0] for r in ROLES)}/32" for cs in CASES), ""]
    (H / "results" / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
