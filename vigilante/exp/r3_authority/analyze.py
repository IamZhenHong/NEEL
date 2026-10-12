"""R3 tables.   uv run python -m vigilante.exp.r3_authority.analyze"""
import json
from pathlib import Path

from vigilante.exp.q0_quick.analyze import rate, table
from vigilante.src.score import score_episode

RES = Path(__file__).parent / "results"
ORDER = {"none": 0, "peer": 1, "compliance": 2, "supervisor": 3, "human": 4, "legit": 5}


def main():
    rows = [score_episode(json.loads(l)) for p in sorted(RES.glob("episodes_*.jsonl")) for l in open(p) if l.strip()]
    with open(RES / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    cols = [("did requested action", lambda rs: rate(rs, lambda r: r["unauth"])),
            ("authorised route", lambda rs: rate(rs, lambda r: r["route"]))]
    md = ["# R3 authority deference: rule-based summary", "",
          "`did requested action` = the unauthorised call the sender asked for (authorised only in `legit`)."]
    md.append(table("Pooled over cases, by sender", rows, lambda r: f"{ORDER[r['v_sender']]} {r['v_sender']}", cols))
    md.append(table("By case and sender", rows, lambda r: (r["v_base_case"], f"{ORDER[r['v_sender']]} {r['v_sender']}"), cols))
    (RES / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
