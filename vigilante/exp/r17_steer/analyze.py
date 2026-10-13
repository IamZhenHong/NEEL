"""R17 table by arm: unauthorised action (rule-based), thought closed, and judge labels on the reasoning.
   uv run python -m vigilante.exp.r17_steer.analyze"""
import json
from collections import defaultdict
from pathlib import Path

from vigilante.src.score import score_episode

RES = Path(__file__).parent / "results"
ARMS = ["none", "bat2", "bat6", "rand6", "p4dir6", "promptP1", "promptP4"]


def main():
    recs = [json.loads(l) for l in open(RES / "episodes_agent.jsonl")]
    J = {}
    if (RES / "judge_steer.json").exists():
        J = {k: v["labels"] for k, v in json.load(open(RES / "judge_steer.json"))["sonnet"].items() if v["labels"]}
    g = defaultdict(lambda: defaultdict(int))
    for r in recs:
        s = score_episode(r)
        arm = r["job"]["arm"]
        c = g[(arm, "all")]
        c["n"] += 1
        c["unauth"] += s["unauth"]
        c["route"] += s["route"]
        c["unclosed"] += any(not t.get("think_closed", True) for t in r["turns"])
        g[(arm, s["case"].split("@")[0])]["unauth"] += s["unauth"]
        g[(arm, s["case"].split("@")[0])]["n"] += 1
        L = J.get(r["key"])
        if L:
            c["judged"] += 1
            for k in ("persona_framing", "weighs_permissions", "coherent"):
                c[k] += L[k]
    md = ["# R17 steering during reasoning: summary", "",
          "| arm | unauthorised | A1 / C1 / E1 | route used | thought unclosed | Batman-like reasoning | weighs permissions | coherent |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        c = g[(arm, "all")]
        per = " / ".join(f"{g[(arm, k)]['unauth']}/{g[(arm, k)]['n']}" for k in ("A1", "C1", "E1:proven"))
        jd = c["judged"] or 1
        md.append(f"| {arm} | {c['unauth']}/{c['n']} | {per} | {c['route']}/{c['n']} | {c['unclosed']}/{c['n']} | "
                  f"{c['persona_framing']}/{c['judged']} | {c['weighs_permissions']}/{c['judged']} | {c['coherent']}/{c['judged']} |")
    (RES / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
