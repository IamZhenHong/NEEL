"""s6 jobs. Main: per model, 4 personas x 2 inductions x 20 variants x 3 samples + B0 x 20 x 3 = 540 episodes.
Pilot (<= 10% of main n): system-prompt induction, P0 only, 4 personas x 9 variants + B0 x 9, plus 8 full-script
episodes (format checks) = 53 per model, own seeds (keys prefixed S6pilot).

  uv run python -m selfknow.s6_reflective_identity.build_jobs
"""
import json
from pathlib import Path

from selfknow.s6_reflective_identity import s6

H = Path(__file__).parent
MODELS = {"qwen": "Qwen/Qwen3.6-27B", "gemma": "google/gemma-4-31B-it"}
N_SAMPLES = 3


def job(prefix, model, persona, induction, v, s):
    key = f"{prefix}|{model}|{persona}|{induction}|v{v}|s{s}"
    return {"key": key, "model": model, "persona": persona, "induction": induction, "v": v, "sample": s}


def main_jobs(model):
    js = []
    for s in range(N_SAMPLES):            # sample-major order: an early stop leaves balanced cells
        for v in range(s6.N_VARIANTS):
            for p in s6.SIGNED:
                for ind in s6.INDUCTIONS:
                    js.append(job("S6", model, p, ind, v, s))
            js.append(job("S6", model, "B0", "none", v, s))
    return js


def pilot_jobs(model):
    """Gate and base rates: P0 only (system prompt, 4 personas x 9 variants + B0 x 9). Format checks of the whole
    script (reads, corrections, in-context rendering): 8 full episodes on variant 9 (each persona, both inductions)."""
    js = []
    for v in range(9):
        for p in s6.SIGNED:
            js.append({**job("S6pilot", model, p, "sys", v, 0), "only_p0": True})
        js.append({**job("S6pilot", model, "B0", "none", v, 0), "only_p0": True})
    for p in s6.SIGNED:
        for ind in s6.INDUCTIONS:
            js.append(job("S6pilot", model, p, ind, 9, 0))
    return js


def pilot2_jobs(model):
    """Pilot 2 (after pilot 1 showed B0 never calls recommend_plan): the request now asks for the checkout page.
    P0 only, system prompt, 4 personas x 5 variants + B0 x 5, new variants 10-14 and own seeds."""
    js = []
    for v in range(10, 15):
        for p in s6.SIGNED:
            js.append({**job("S6pilot2", model, p, "sys", v, 0), "only_p0": True})
        js.append({**job("S6pilot2", model, "B0", "none", v, 0), "only_p0": True})
    return js


def write(path, js):
    with open(path, "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(path.name, len(js))


if __name__ == "__main__":
    (H / "jobs").mkdir(exist_ok=True)
    for m in MODELS:
        write(H / "jobs" / f"main_{m}.jsonl", main_jobs(m))
        write(H / "jobs" / f"pilot_{m}.jsonl", pilot_jobs(m))
        write(H / "jobs" / f"pilot2_{m}.jsonl", pilot2_jobs(m))
