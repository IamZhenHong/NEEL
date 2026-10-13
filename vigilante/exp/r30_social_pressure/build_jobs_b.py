"""R30b2 jobs (addendum vig-r30b2-prereg, after the R30 results): the peer-reporting case with two other wordings of
the keep-quiet request and the peer saying it will handle it without asking (PEER_CONDS_B), both wordings, P0,
20 per cell, Gemma 4 31B and Qwen3.6-27B.
  uv run python -m vigilante.exp.r30_social_pressure.build_jobs_b"""
import json
from pathlib import Path

from vigilante.cases.social import PEER_CONDS_B
from vigilante.exp.r30_social_pressure.build_jobs import N, job

H = Path(__file__).parent


def jobs(exp, fmt=None):
    js = [job(exp, f"C1#r30-peer-{c}-{w}", "none", None, i, fmt) for c in PEER_CONDS_B for w in ("W0", "W1") for i in range(N)]
    for j in js:
        j["group"] = "r30b2"
    return js


if __name__ == "__main__":
    for name, exp, fmt in (("gemma", "R30b2g", "gemma4"), ("qwen", "R30b2q", None)):
        js = jobs(exp, fmt)
        with open(H / f"jobs_b2_{name}.jsonl", "w") as f:
            f.writelines(json.dumps(j) + "\n" for j in js)
        print(f"{exp}: {len(js)} jobs")
