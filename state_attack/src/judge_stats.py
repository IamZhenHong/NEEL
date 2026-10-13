"""Baseline for B2's judged-behaviour reward: Haiku 5.5 trait ratings (0-100, the B2 reward prompt) of ordinary replies
(no-system replies to the 40 neutral questions x 2 + the naturalistic neutral replies x 160), per trait -> mean, sd.
  uv run --with anthropic python -m state_attack.src.judge_stats
"""
import json
from pathlib import Path

import numpy as np

from state_attack.src import llm
from state_attack.src.attack import judge_rate
from state_attack.src.build_dirs import questions

HERE = Path(__file__).resolve().parents[1]
R = HERE / "results" / "dirs"


def main():
    llm.load_env()
    qs = questions()
    vs = json.loads((HERE / "data" / "valset.json").read_text())
    base = [r for r in map(json.loads, open(R / "instructed.jsonl")) if r["trait"] == "base"]
    val = [r for r in map(json.loads, open(R / "val.jsonl")) if r["cond"] == "neutral"]
    reps = [(qs[r["qi"]], r["reply"]) for r in base] + [(vs[r["trait"]][r["i"]]["user2_neutral"], r["reply"]) for r in val]
    out = {}
    for t in ["frustration", "threat", "dependency", "deference", "assistant"]:
        sc = llm.pmap(lambda qr: judge_rate(t, qr[0], qr[1], "judge_stats"), reps, workers=32)
        sc = np.array([s for s in sc if s is not None], float)
        out[t] = {"mu": float(sc.mean()), "sd": float(max(sc.std(), 5.0)), "n": int(len(sc)),
                  "p75": float(np.percentile(sc, 75)), "raw_sd": float(sc.std())}
        print(t, out[t], flush=True)
    (R / "judge_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
