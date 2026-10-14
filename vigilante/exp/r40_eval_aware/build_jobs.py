"""R40 jobs (prereg vig-r40-prereg), Qwen3.6-27B, thinking off.
  behaviour: the 8 over-limit R26 cells x 46 variants x 3 samples (episodes, rule-based acted)
  ratings:   all 12 cells x 46 variants x 2 questions (greedy, read as a number) -> written by pod_r40.py
  reads:     all 12 cells x 46 variants, last prompt token (transformers) -> pod_r40.py
  uv run python -m vigilante.exp.r40_eval_aware.build_jobs"""
import hashlib
import json
from pathlib import Path

from vigilante.exp.r40_eval_aware.stimuli import OVER, variants

H = Path(__file__).parent
N = 3

if __name__ == "__main__":
    js = []
    for vid, kind, lab, sfx in variants():
        for cid in OVER:
            for i in range(N):
                key = f"R40|{vid}|{cid}|{i}"
                js.append({"key": key, "exp": "R40", "group": "agent", "cond": "P0", "case": cid, "vid": vid,
                           "kind": kind, "labels": lab, "system_suffix": sfx, "sample": i, "thinking": False,
                           "handover": False, "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)})
    with open(H / "jobs_qwen.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(len(js), "jobs;", len(variants()), "variants;", len(OVER), "over-limit cells")
