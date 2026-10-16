"""e64b agent-episode jobs (DESIGN_b.md). Qwen3.6-27B, fmt qwen (XML tool calls), thinking off, at most 1,024 new
tokens per assistant turn, no handover note (e64 settings).

  pilot  B only: 20 peer-ask + 20 supervisor-ask episodes (templates cycled) for the gate.
  main   every condition x (peer 60, sup 60, limit 60, disc 60).

  uv run python -m vibes.e64_layer_map.build_jobs_b
"""
import hashlib
import json
from pathlib import Path

from vibes.e64_layer_map import conds_b as CB
from vibes.e64_layer_map import families_b as FB

H = Path(__file__).resolve().parent
JOBS = H / "jobs"
N_MAIN = 60


def job(phase, cond, case, i):
    key = f"E64B|{phase}|{cond}|{case}|{i}"
    return {"key": key, "exp": "E64B", "phase": phase, "group": cond, "cond": cond, "case": case,
            "family": FB.family_of(case), "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i,
            "thinking": False, "handover": False, "fmt": "qwen", "max_tokens": 1024, "engine": "vllm"}


def cycled(phase, cond, fam, n):
    cells = FB.FAMILIES[fam]
    return [job(phase, cond, cells[i % len(cells)], i) for i in range(n)]


def write(path, js):
    assert len({j["key"] for j in js}) == len(js)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(j) + "\n" for j in js)
    print(f"{path.name}: {len(js)} jobs")


def main():
    write(JOBS / "b_pilot.jsonl", cycled("pilot", "B", "peer", 20) + cycled("pilot", "B", "sup", 20))
    write(JOBS / "b_main.jsonl", [j for c in CB.CONDS for fam in FB.FAMILIES for j in cycled("main", c, fam, N_MAIN)])
    tiny = [job("tiny", c, FB.FAMILIES[f][0], 0) for c in ("B", "T5r") for f in FB.FAMILIES]
    for j in tiny:
        j["max_tokens"] = 48
    write(JOBS / "b_tiny.jsonl", tiny)


if __name__ == "__main__":
    main()
