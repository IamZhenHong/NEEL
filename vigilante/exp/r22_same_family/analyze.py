"""R22 analysis (prereg vig-r22-prereg; local, rule-based).   uv run python -m vigilante.exp.r22_same_family.analyze"""
import json
from pathlib import Path

from scipy.stats import fisher_exact

from vigilante.cases.authority import IDENTITIES, ROLE
from vigilante.src.score import score_episode

H = Path(__file__).parent
R5 = H.parent / "r5_identity" / "results" / "scores.jsonl"


def main():
    rows = [score_episode(json.loads(l)) for l in open(H / "results" / "episodes_agent.jsonl")]
    with open(H / "results" / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    r5 = [json.loads(l) for l in open(R5)]
    u = lambda rs, c, i: [r["unauth"] for r in rs if r["case"] == f"{c}#id-{i}"]
    L = ["# R22 (Qwen3.6-27B, P0, thinking off, 30 new per cell; R5 had 10)", "",
         f"Episodes {len(rows)}; malformed {sum(r['malformed'] > 0 for r in rows)}.", "",
         "| payment made | " + " | ".join(IDENTITIES) + " |", "|---|" + "---|" * len(IDENTITIES)]
    for c in ROLE:
        L.append(f"| {c} R22 | " + " | ".join(f"{sum(u(rows, c, i))}/{len(u(rows, c, i))}" for i in IDENTITIES) + " |")
    pooled = {i: (sum(sum(u(rows, c, i)) for c in ROLE), sum(len(u(rows, c, i)) for c in ROLE)) for i in IDENTITIES}
    both = {i: (pooled[i][0] + sum(sum(u(r5, c, i)) for c in ROLE), pooled[i][1] + sum(len(u(r5, c, i)) for c in ROLE))
            for i in IDENTITIES}
    L.append("| **pooled R22** | " + " | ".join(f"{a}/{b} ({a / b:.0%})" for a, b in pooled.values()) + " |")
    L.append("| pooled R5 + R22 | " + " | ".join(f"{a}/{b} ({a / b:.0%})" for a, b in both.values()) + " |")
    sa, sn = pooled["same_family"]
    L += [""]
    for ref in ("unspecified", "human", "gpt", "claude"):
        ra, rn = pooled[ref]
        p = fisher_exact([[sa, sn - sa], [ra, rn - ra]])[1]
        L.append(f"Same family vs {ref} (R22 only): {sa}/{sn} vs {ra}/{rn}, Fisher p = {p:.2g}")
    ask = {i: sum(r["route"] for c in ROLE for r in rows if r["case"] == f"{c}#id-{i}") for i in IDENTITIES}
    L += ["", "Asked for approval (request_approval), pooled of 90: " + ", ".join(f"{i} {v}" for i, v in ask.items())]
    (H / "results" / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
