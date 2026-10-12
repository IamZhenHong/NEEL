"""Shared quick tables for R5-R7: payment made / route, by case and by a variant key."""
import json
from collections import defaultdict
from pathlib import Path

from vigilante.exp.q0_quick.analyze import rate
from vigilante.src.score import score_episode


def load(res: Path):
    rows = [score_episode(json.loads(l)) for p in sorted(res.glob("episodes_*.jsonl")) for l in open(p) if l.strip()]
    with open(res / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return rows


def grid(rows, key, cols_order, case_key="v_base_case", title=""):
    g = defaultdict(list)
    for r in rows:
        g[(r.get(case_key) or r["case"], key(r))].append(r)
    cases = sorted({c for c, _ in g})
    out = [f"### {title}", "", "| case | " + " | ".join(cols_order) + " |", "|---|" + "---|" * len(cols_order)]
    for c in cases:
        out.append(f"| {c} | " + " | ".join(
            (f"{sum(r['unauth'] for r in g[(c, v)])}/{len(g[(c, v)])}" if g[(c, v)] else "-") for v in cols_order) + " |")
    pooled = [sum(sum(r["unauth"] for r in g[(c, v)]) for c in cases) for v in cols_order]
    n = [sum(len(g[(c, v)]) for c in cases) for v in cols_order]
    out.append("| **pooled** | " + " | ".join(f"{a}/{b}" for a, b in zip(pooled, n)) + " |")
    return "\n".join(out) + "\n"
