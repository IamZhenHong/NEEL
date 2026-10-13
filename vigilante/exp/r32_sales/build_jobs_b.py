"""R32b jobs (addendum vig-r32b-prereg, after the R32 results): which part of the salesperson persona keeps Qwen's
customers on Plus. P0, the R32 salesperson, two paraphrases, a matched warm non-sales persona, the name alone; no
revenue line; neutral and cued customers; 20 per cell; Gemma 4 31B and Qwen3.6-27B (new seeds: exp R32b).
  uv run python -m vigilante.exp.r32_sales.build_jobs_b"""
import hashlib
import json
from pathlib import Path

H = Path(__file__).parent
N = 20
CONDS = ("P0", "S_sales", "S_sales2", "S_sales3", "S_warm", "S_name")


def jobs(exp, fmt=None):
    js = []
    for st in ("neutral", "cued"):
        for cond in CONDS:
            for i in range(N):
                key = f"{exp}|{cond}|SALES#{st}|none|{i}"
                j = {"key": key, "exp": exp, "group": "r32b", "cond": cond, "case": f"SALES#{st}", "line": "none",
                     "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i, "thinking": False,
                     "handover": False}
                if fmt:
                    j["fmt"] = fmt
                js.append(j)
    return js


if __name__ == "__main__":
    for name, exp, fmt in (("gemma", "R32bg", "gemma4"), ("qwen", "R32bq", None)):
        js = jobs(exp, fmt)
        with open(H / f"jobs_b_{name}.jsonl", "w") as f:
            f.writelines(json.dumps(j) + "\n" for j in js)
        print(f"{exp}: {len(js)} jobs")
