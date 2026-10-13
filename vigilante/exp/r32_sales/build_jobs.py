"""R32 jobs (prereg vig-r32-prereg): persona (P0 / S_sales) x revenue line (none / incentive) x customer state
(neutral / cued / stated), 20 per cell, thinking off, Gemma 4 31B and Qwen3.6-27B.
  uv run python -m vigilante.exp.r32_sales.build_jobs"""
import hashlib
import json
from pathlib import Path

from vigilante.cases import ALL
from vigilante.cases.sales import INCENTIVE, STATES

H = Path(__file__).parent
N = 20


def jobs(exp, fmt=None):
    js = []
    for st in STATES:
        for cond in ("P0", "S_sales"):
            for inc in ("none", "incentive"):
                for i in range(N):
                    key = f"{exp}|{cond}|SALES#{st}|{inc}|{i}"
                    j = {"key": key, "exp": exp, "group": "r32", "cond": cond, "case": f"SALES#{st}", "line": inc,
                         "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i,
                         "thinking": False, "handover": False}
                    if inc == "incentive":
                        j["system_suffix"] = INCENTIVE
                    if fmt:
                        j["fmt"] = fmt
                    js.append(j)
    assert all(j["case"] in ALL for j in js) and len({j["key"] for j in js}) == len(js)
    return js


if __name__ == "__main__":
    for name, exp, fmt in (("gemma", "R32g", "gemma4"), ("qwen", "R32q", None)):
        js = jobs(exp, fmt)
        with open(H / f"jobs_{name}.jsonl", "w") as f:
            f.writelines(json.dumps(j) + "\n" for j in js)
        print(f"{exp}: {len(js)} jobs")
