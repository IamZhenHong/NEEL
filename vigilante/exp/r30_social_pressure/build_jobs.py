"""R30 jobs (prereg vig-r30-prereg): bystander lines (a), peer reporting (b), descriptive norms (c). P0, thinking off,
20 per cell, Gemma 4 31B and Qwen3.6-27B. The no-line baseline of the three shared cells serves parts a and c.
  uv run python -m vigilante.exp.r30_social_pressure.build_jobs"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from vigilante.cases import ALL
from vigilante.cases.social import NORMS, PEER_CONDS, REVIEW

H = Path(__file__).parent
N = 20
A_CELLS = ["C1#r26-cancel-198", "C3#r26-error-110", "C3#r26-error-240", "C1#r26-orig9"]
C_CELLS = A_CELLS[:3]


def job(exp, case, line_id, suffix, i, fmt):
    key = f"{exp}|P0|{case}|{line_id}|{i}"
    j = {"key": key, "exp": exp, "group": "r30", "cond": "P0", "case": case, "line": line_id,
         "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i, "thinking": False,
         "handover": True}
    if suffix:
        j["system_suffix"] = suffix
    if fmt:
        j["fmt"] = fmt
    return j


def jobs(exp, fmt=None):
    js = []
    for c in A_CELLS:
        dep = ALL[c].variant["base_case"]
        for line_id in ("none", "reviewer", "final"):
            js += [job(exp, c, line_id, REVIEW[dep].get(line_id), i, fmt) for i in range(N)]
    for c in C_CELLS:
        dep = ALL[c].variant["base_case"]
        for line_id in ("comply", "deviate", "sanction", "stated"):
            js += [job(exp, c, line_id, NORMS[dep][line_id], i, fmt) for i in range(N)]
    for cond in PEER_CONDS:
        for w in ("W0", "W1"):
            js += [job(exp, f"C1#r30-peer-{cond}-{w}", "none", None, i, fmt) for i in range(N)]
    assert all(j["case"] in ALL for j in js) and len({j["key"] for j in js}) == len(js)
    return js


if __name__ == "__main__":
    for name, exp, fmt in (("gemma", "R30g", "gemma4"), ("qwen", "R30q", None)):
        js = jobs(exp, fmt)
        with open(H / f"jobs_{name}.jsonl", "w") as f:
            f.writelines(json.dumps(j) + "\n" for j in js)
        print(f"{exp}: {len(js)} jobs", Counter(j["line"] for j in js))
