"""s4 job files.
  pilot (card gate; <= 10% of the main n per model): N = 0 only. P1 at every ladder rung L0-L4 x variants 0-1 x 10
        samples (100), and P2 x variants 0-1 x 10 (20, base rate only). 120 per model.
  main: N = 0 (one shared cell per probe: with no history, H/C and self/other are the same prompt) and N in 5/20/50 x
        history H/C x authorship self/other, x probes P1 (chosen rung) and P2, x 12 variants x 5 samples.
        Per model 2 x (60 + 3 x 2 x 2 x 60) = 1,560; 3,120 in all.
Jobs are grouped by N (each group is written when it finishes) and sorted so that episodes sharing a history prefix are
adjacent (prefix cache). Thinking off, temperature 0.7, no handover note, at most 8 tool steps, 1,000 tokens per step.
  uv run python -m selfknow.s4_accumulation.build_jobs pilot
  uv run python -m selfknow.s4_accumulation.build_jobs main --rung-qwen K --rung-gemma K
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from selfknow.s4_accumulation.case import CASES, N_VARIANTS, NS, P1_LADDER

H = Path(__file__).parent
MODELS = {"qwen": "qwen36_27b", "gemma4": "gemma4_31b"}
SAMPLES_MAIN, SAMPLES_PILOT, PILOT_VARIANTS = 5, 10, (0, 1)


def job(stage, fmt, v, probe, rung, N, hist, auth, i, group):
    cid = f"s4|v{v:02d}|P1|L{rung}" if probe == "P1" else f"s4|v{v:02d}|P2"
    assert cid in CASES, cid
    key = f"s4|{MODELS[fmt]}|{stage}|N{N}|{hist}|{auth}|v{v:02d}|{probe}{'L' + str(rung) if probe == 'P1' else ''}|{i}"
    return {"key": key, "exp": "s4", "stage": stage, "group": group, "model": MODELS[fmt], "fmt": fmt, "cond": "P0",
            "case": cid, "variant": v, "N": N, "hist": hist, "auth": auth, "probe": probe,
            "rung": rung if probe == "P1" else None, "sample": i,
            "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "thinking": False, "handover": False,
            "max_tokens": 1000}


def pilot():
    js = []
    for fmt in MODELS:
        for v in PILOT_VARIANTS:
            for k in P1_LADDER:
                js += [job("pilot", fmt, v, "P1", k, 0, None, None, i, "pilot") for i in range(SAMPLES_PILOT)]
            js += [job("pilot", fmt, v, "P2", None, 0, None, None, i, "pilot") for i in range(SAMPLES_PILOT)]
    return js


def main_jobs(rungs: dict):
    js = []
    for fmt in MODELS:
        for N in NS:
            group = f"N{N}"
            conds = [(None, None)] if N == 0 else [(h, a) for h in ("H", "C") for a in ("self", "other")]
            for v in range(N_VARIANTS):
                for hist, auth in conds:            # adjacent: the same history prefix
                    for probe in ("P1", "P2"):
                        js += [job("main", fmt, v, probe, rungs[fmt], N, hist, auth, i, group)
                               for i in range(SAMPLES_MAIN)]
    return js


def write(path, js):
    keys = [j["key"] for j in js]
    assert len(set(keys)) == len(keys), "duplicate keys"
    with open(path, "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    c = Counter((j["model"], j["group"]) for j in js)
    print(f"{path.name}: {len(js)} jobs; " + ", ".join(f"{m}/{g} {n}" for (m, g), n in sorted(c.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "main"])
    ap.add_argument("--rung-qwen", type=int)
    ap.add_argument("--rung-gemma", type=int)
    a = ap.parse_args()
    if a.stage == "pilot":
        write(H / "jobs_pilot.jsonl", pilot())
    else:
        assert a.rung_qwen in P1_LADDER and a.rung_gemma in P1_LADDER
        write(H / "jobs_main.jsonl", main_jobs({"qwen": a.rung_qwen, "gemma4": a.rung_gemma}))
