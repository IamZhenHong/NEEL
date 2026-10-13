"""R32c jobs (addendum vig-r32c-prereg, after the R32/R32b results): the three options tasks of cases/options.py under
P0, S_warm and S_name, 20 per cell, thinking off, no handover. Qwen3.6-27B and Gemma 4 31B (offline engine); Llama 3.3
70B (AWQ) and Mistral Small 3 24B (vLLM server, as R28) also get R32's SALES#neutral as an anchor.
  uv run python -m vigilante.exp.r32_sales.build_jobs_c"""
import hashlib
import json
from pathlib import Path

from vigilante.cases import ALL

H = Path(__file__).parent
N = 20
CONDS = ("P0", "S_warm", "S_name")
TASKS = ("OPT#ship", "OPT#delay", "OPT#save")
MODELS = {"qwen": ("R32cq", None, TASKS), "gemma": ("R32cg", "gemma4", TASKS),
          "llama": ("R32cl", None, TASKS + ("SALES#neutral",)), "mistral": ("R32cm", None, TASKS + ("SALES#neutral",))}


def jobs(exp, fmt, tasks):
    js = []
    for case in tasks:
        for cond in CONDS:
            for i in range(N):
                key = f"{exp}|{cond}|{case}|none|{i}"
                j = {"key": key, "exp": exp, "group": "r32c", "cond": cond, "case": case, "line": "none",
                     "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i, "thinking": False,
                     "handover": False}
                if fmt:
                    j["fmt"] = fmt
                js.append(j)
    assert all(j["case"] in ALL for j in js) and len({j["key"] for j in js}) == len(js)
    return js


if __name__ == "__main__":
    for name, (exp, fmt, tasks) in MODELS.items():
        js = jobs(exp, fmt, tasks)
        with open(H / f"jobs_c_{name}.jsonl", "w") as f:
            f.writelines(json.dumps(j) + "\n" for j in js)
        print(f"{exp}: {len(js)} jobs")
