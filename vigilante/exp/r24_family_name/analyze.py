"""R24 analysis (prereg vig-r24-prereg; local, rule-based).   uv run python -m vigilante.exp.r24_family_name.analyze
Episodes come from the R23 pods: Gemma in r23_long_mech/results/gemma_b, Qwen in r23_long_mech/results/qwen/s1."""
import json
from pathlib import Path

from scipy.stats import fisher_exact

from vigilante.cases.authority import ROLE
from vigilante.exp.r24_family_name.build_jobs import GEMMA, QWEN
from vigilante.src.score import score_episode

H = Path(__file__).parent
POD = H.parent / "r23_long_mech" / "results"
R22 = H.parent / "r22_same_family" / "results" / "scores.jsonl"
SRC = {"Gemma 4 31B": ("R24g", POD / "gemma_b" / "episodes_agent.jsonl", GEMMA),
       "Qwen3.6-27B": ("R24q", POD / "qwen" / "s1" / "episodes_agent.jsonl", QWEN)}
NAME = {"unspecified": "unspecified", "human": "human staff", "same_family": "Qwen + 'same family as you'",
        "gemma_family": "Gemma + 'same family as you'", "family_only": "'same family as you' only",
        "qwen_name": "Qwen, by Alibaba", "gemma_name": "Gemma, by Google", "claude": "Claude", "gpt": "GPT"}


def main():
    L = ["# R24: same-family caution, name or phrase (P0, thinking off, rule-based)", ""]
    allrows = []
    for model, (exp, path, ids) in SRC.items():
        if not path.exists():
            continue
        rows = [score_episode(r) for r in map(json.loads, open(path)) if r["job"].get("exp") == exp]
        allrows += rows
        u = lambda c, i: [r["unauth"] for r in rows if r["case"] == f"{c}#id-{i}"]
        L += [f"## {model}", "", f"Episodes {len(rows)}; malformed {sum(r['malformed'] > 0 for r in rows)}.", "",
              "| claimed identity | " + " | ".join(ROLE) + " | pooled | asked for approval |", "|---|" + "---|" * (len(ROLE) + 2)]
        pooled = {}
        for i in ids:
            a, n = sum(sum(u(c, i)) for c in ROLE), sum(len(u(c, i)) for c in ROLE)
            pooled[i] = (a, n)
            ask = sum(r["route"] for c in ROLE for r in rows if r["case"] == f"{c}#id-{i}")
            L.append(f"| {NAME[i]} | " + " | ".join(f"{sum(u(c, i))}/{len(u(c, i))}" for c in ROLE)
                     + f" | {a}/{n} ({a / n:.0%}) | {ask} |")
        ra, rn = pooled["unspecified"]
        L += ["", "Fisher exact vs unspecified: " + "; ".join(
            f"{NAME[i]} p = {fisher_exact([[a, n - a], [ra, rn - ra]])[1]:.2g}" for i, (a, n) in pooled.items()
            if i != "unspecified"), ""]
    if R22.exists():
        r22 = [json.loads(l) for l in open(R22)]
        L.append("Qwen R22 (30 per cell, same settings): " + ", ".join(
            f"{NAME[i]} {sum(r['unauth'] for r in r22 if r['case'].endswith('#id-' + i))}/"
            f"{sum(r['case'].endswith('#id-' + i) for r in r22)}" for i in ("unspecified", "same_family", "claude", "gpt")))
    with open(H / "results" / "scores.jsonl", "w") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    (H / "results" / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
