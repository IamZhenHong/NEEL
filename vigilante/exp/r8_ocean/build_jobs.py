"""R8 (queue item 26): Persona Cartography OCEAN adapters on Qwen3-32B in the authority and route-closed cells.
   uv run python -m vigilante.exp.r8_ocean.build_jobs"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from vigilante.cases import ALL

HERE = Path(__file__).parent
ROOT = "fine_tuning/qwen-3-32b-it"
ADAPTERS = {
    "control": f"{ROOT}/other/ocean_def_control/amplifier/ocean_const_paired_dpo_nothink_s1vs2/lora/ocean_def_control_full-persona",
}
for trait, short in (("agreeableness", "A"), ("conscientiousness", "C"), ("neuroticism", "N")):
    for d, sign, verb in (("amplifier", "+", "amplifying"), ("suppressor", "-", "suppressing")):
        ADAPTERS[f"{short}{sign}"] = f"{ROOT}/ocean/{trait}/{d}/ocean_const_paired_dpo_nothink/lora/{trait}_{verb}_full-persona"
CONDS = [None] + list(ADAPTERS)            # None = base model, no adapter
CELLS = [f"{c}#{s}-nl" for c in ("C1", "C3", "C4") for s in ("peer", "supervisor")] + ["C1@closed"]
N = 8


def main():
    js = []
    for cell in CELLS:
        assert cell in ALL
        for lo in CONDS:
            for i in range(N):
                key = f"R8|P0|{lo or 'base'}|{cell}|{i}"
                js.append({"key": key, "exp": "R8", "group": "agent", "cond": "P0", "case": cell, "lora": lo,
                           "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i,
                           "thinking": False, "handover": True, "fmt": "qwen3"})
    with open(HERE / "jobs.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    (HERE / "adapters.json").write_text(json.dumps(ADAPTERS, indent=1))
    print(f"{len(js)} jobs", dict(Counter(j["lora"] or "base" for j in js)))


if __name__ == "__main__":
    main()
