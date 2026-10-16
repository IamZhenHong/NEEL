"""e64 agent-episode jobs.

  pilot: base rates of the six candidate families on the vLLM base (48 per family) + 24 HF-base episodes (timing and
         engine check, 4 per family).
  main:  the 3 selected families x 120 episodes x every condition (README Implementation).

  uv run python -m vibes.e64_layer_map.build_jobs pilot
  uv run python -m vibes.e64_layer_map.build_jobs main --families flag limit auth
  uv run python -m vibes.e64_layer_map.build_jobs tiny       # local smoke jobs
"""
import argparse
import hashlib
import json
from pathlib import Path

from vibes.e64_layer_map import conds as C
from vibes.e64_layer_map import families as F

H = Path(__file__).resolve().parent
JOBS = H / "jobs"
MAX_TOKENS = 1024          # per assistant turn, every condition and engine
N_MAIN = 120


def seed(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def job(phase, cond, case, i):
    key = f"E64|{phase}|{cond}|{case}|{i}"
    eng, _, lora, hook = C.CONDS[cond]
    j = {"key": key, "exp": "E64", "phase": phase, "group": cond, "cond": cond, "case": case,
         "family": F.family_of(case), "seed": seed(key), "sample": i, "thinking": False, "handover": False,
         "fmt": "qwen3", "max_tokens": MAX_TOKENS, "engine": eng}
    if lora:
        j["lora"] = lora
    if hook:
        j["hook"] = hook
    return j


def per_cell(fam, total):
    cells = F.FAMILIES[fam]
    assert total % len(cells) == 0, (fam, total)
    return cells, total // len(cells)


def family_jobs(phase, cond, fam, total):
    cells, k = per_cell(fam, total)
    return [job(phase, cond, c, i) for c in cells for i in range(k)]


def write(path, js):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert len({j["key"] for j in js}) == len(js)
    with open(path, "w") as f:
        f.writelines(json.dumps(j) + "\n" for j in js)
    print(f"{path.name}: {len(js)} jobs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["pilot", "main", "tiny"])
    ap.add_argument("--families", nargs="*", default=None)
    a = ap.parse_args()
    if a.which == "pilot":
        js = [j for fam in F.FAMILIES for j in family_jobs("pilot", "B", fam, 48)]
        write(JOBS / "pilot_vllm.jsonl", js)
        hf = []
        for fam in F.FAMILIES:
            cells = F.FAMILIES[fam]
            hf += [job("pilot", "B_hf", cells[i % len(cells)], i) for i in range(4)]
        write(JOBS / "pilot_hf.jsonl", hf)
    elif a.which == "main":
        assert a.families and len(a.families) == 3, "give the 3 selected families"
        js = [j for cond in C.CONDS for fam in a.families for j in family_jobs("main", cond, fam, N_MAIN)]
        write(JOBS / "main_vllm.jsonl", [j for j in js if j["engine"] == "vllm"])
        write(JOBS / "main_hf.jsonl", [j for j in js if j["engine"] == "hf"])
    else:   # tiny: one episode per family under B, T3, T8 (vLLM path) and B_hf, T6, T7 (HF path)
        js = [job("tiny", cond, F.FAMILIES[fam][0], 0) for cond in ("B", "T3") for fam in F.FAMILIES]
        hf = [job("tiny", cond, F.FAMILIES[fam][0], 0) for cond in ("B_hf", "T6", "T6p", "T7") for fam in ("flag", "fc")]
        for j in js + hf:
            j["max_tokens"] = 48
        write(JOBS / "tiny_vllm.jsonl", js)
        write(JOBS / "tiny_hf.jsonl", hf)


if __name__ == "__main__":
    main()
