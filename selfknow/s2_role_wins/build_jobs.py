"""Job files for s2.   uv run python -m selfknow.s2_role_wins.build_jobs

  jobs/pilot_<model>.jsonl : 16 single-identity controls x (P1 x 3, P2 x 3, P3 x 1) + 6 main configs x 3 probes x 1
                             = 130 per model (9.7% of the main n of 1344)
  jobs/main_<model>.jsonl  : 96 configs x 3 probes x 4 samples + 16 controls x 3 probes x 4 samples = 1344 per model
Seeds are sha256 of the key, so pilot and main draws differ.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
MODELS = {"qwen": "qwen", "gemma": "gemma4"}           # tag -> chat/tool format
PILOT_CONFIGS = ("c00", "c16", "c32", "c48", "c64", "c80")


def job(model, phase, cf, probe, sample):
    key = f"s2|{phase}|{model}|{cf['config']}|{probe}|{sample}"
    return {"key": key, "phase": phase, "model": model, "fmt": MODELS[model], "config": cf["config"],
            "control": cf.get("control", False), "slots": cf["slots"], "probe": probe, "sample": sample,
            "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)}


def main():
    d = json.loads((HERE / "configs.json").read_text())
    configs, ctl = d["configs"], d["controls"]
    (HERE / "jobs").mkdir(exist_ok=True)
    for m in MODELS:
        pilot = [job(m, "pilot", cf, p, s) for cf in ctl for p, n in (("P1", 3), ("P2", 3), ("P3", 1)) for s in range(n)]
        pilot += [job(m, "pilot", cf, p, 0) for cf in configs if cf["config"] in PILOT_CONFIGS for p in ("P1", "P2", "P3")]
        main_ = [job(m, "main", cf, p, s) for cf in configs + ctl for p in ("P1", "P2", "P3") for s in range(4)]
        # pilot2 (after the pilot-driven header fix): gate quantities only, controls x (P1 x 3, P2 x 1) = 64
        pilot2 = [job(m, "pilot2", cf, p, s) for cf in ctl for p, n in (("P1", 3), ("P2", 1)) for s in range(n)]
        for name, js in ((f"pilot_{m}", pilot), (f"pilot2_{m}", pilot2), (f"main_{m}", main_)):
            with open(HERE / "jobs" / f"{name}.jsonl", "w") as f:
                for j in js:
                    f.write(json.dumps(j) + "\n")
            print(f"{name}: {len(js)} jobs")


if __name__ == "__main__":
    main()
