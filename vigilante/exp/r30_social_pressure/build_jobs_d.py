"""R30d jobs (addendum vig-r30d-prereg, after the R30 results): R30c's three R26 cells under no log, the R30 comply log,
a plain reminder of the limit, the comply log without the limit, and a neutral log of the same peers. P0, thinking
off, 20 per cell, Qwen3.6-27B only (Gemma was 0/60 under every log but the deviant one). New seeds: exp R30dq.
  uv run python -m vigilante.exp.r30_social_pressure.build_jobs_d"""
import json
from collections import Counter
from pathlib import Path

from vigilante.cases import ALL
from vigilante.cases.social import NORMS
from vigilante.exp.r30_social_pressure.build_jobs import C_CELLS, N, job

H = Path(__file__).parent
LINES = ("none", "comply", "reminder", "comply_bare", "neutral")


def jobs(exp, fmt=None):
    js = []
    for c in C_CELLS:
        dep = ALL[c].variant["base_case"]
        for line_id in LINES:
            js += [job(exp, c, line_id, NORMS[dep].get(line_id), i, fmt) for i in range(N)]
    for j in js:
        j["group"] = "r30d"
    assert len({j["key"] for j in js}) == len(js)
    return js


if __name__ == "__main__":
    js = jobs("R30dq")
    with open(H / "jobs_d_qwen.jsonl", "w") as f:
        f.writelines(json.dumps(j) + "\n" for j in js)
    print(f"R30dq: {len(js)} jobs", Counter(j["line"] for j in js))
