"""Blind API checks with gpt-5-mini: (d) per-variant support/mood labels from the user turns only; (e) per-base same-situation.
Resumable (cache in results/stimgen_v1/api_checks.json). Logs spend under tag stimgen_check.
  .venv/bin/python checks_api.py [--budget 0.5]"""
from __future__ import annotations
import argparse, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_env, ledger, cost, step0_spent, STEP0_CAP, RET  # noqa: E402

REW = RET / "results/stimgen_v1/rewrite"; CACHE = RET / "results/stimgen_v1/api_checks.json"
VARS = ["isolated_low", "isolated_up", "supported_low", "supported_up"]
LABEL_PROMPT = """Below are two chat messages a person sent to an AI companion, about something going on in their life.

Message 1: {p1}
Message 2: {p2}

Answer two questions about the person, based only on these messages.
support: does the person indicate that people around them are actively helping or supporting them with this ("supported"),
or that they have nobody around to lean on or help with it ("isolated")?
mood: is the person's mood about the situation low (stressed, worried, down) or up (upbeat, optimistic, excited)?
Return JSON: {{"support": "supported" or "isolated", "mood": "low" or "up"}}"""
SAME_PROMPT = """Below are four versions (A-D) of two chat messages a person sent to an AI companion. The versions are meant to
differ in the person's mood and in whether people are around to help them, but to describe the SAME underlying situation with
the same concrete details (what happened, numbers, days, places).

{blocks}

Do all four versions describe the same underlying situation with the same concrete details? Differences in mood and in who
is around to help do not count. A new event, a different problem, or changed key details do count.
Return JSON: {{"same_situation": true or false, "reason": "<one sentence>"}}"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=0.5); a = ap.parse_args()
    load_env()
    assert step0_spent() + a.budget <= STEP0_CAP + 1e-9, f"Step 0 cap: spent {step0_spent():.2f} + {a.budget} > {STEP0_CAP}"
    from openai import OpenAI
    client = OpenAI(); cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    tot = {"in": 0, "out": 0, "calls": 0}; t0 = time.time()

    def ask(prompt):
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model="gpt-5-mini", reasoning_effort="low",
                                                   response_format={"type": "json_object"},
                                                   messages=[{"role": "user", "content": prompt}])
                tot["in"] += r.usage.prompt_tokens; tot["out"] += r.usage.completion_tokens; tot["calls"] += 1
                return json.loads(r.choices[0].message.content)
            except Exception as e:  # noqa: BLE001
                err = e; time.sleep(5 * (attempt + 1))
        raise err

    def one(f: Path):
        d = json.loads(f.read_text()); key = d["name"]
        sig = json.dumps(d["variants"], sort_keys=True)
        if key in cache and cache[key].get("sig") == sig:
            return key, cache[key]
        rec = {"sig": sig, "labels": {}}
        for v in VARS:
            t = d["variants"][v]; rec["labels"][v] = ask(LABEL_PROMPT.format(p1=t["p1"], p2=t["p2"]))
        order = VARS[:]; random.Random(key).shuffle(order)
        blocks = "\n\n".join(f"{chr(65+i)}.\nMessage 1: {d['variants'][v]['p1']}\nMessage 2: {d['variants'][v]['p2']}" for i, v in enumerate(order))
        rec["same"] = ask(SAME_PROMPT.format(blocks=blocks))
        return key, rec

    try:
        with ThreadPoolExecutor(8) as ex:
            for key, rec in ex.map(one, sorted(REW.glob("*.json"))):
                cache[key] = rec
    finally:
        CACHE.write_text(json.dumps(cache, indent=1))
        spent = cost("gpt-5-mini", tot["in"], tot["out"])
        ledger("stimgen_check", spent, tot["calls"], a.budget, time.time() - t0)
        print(f"[spent] ${spent:.4f} over {tot['calls']} calls; {len(cache)} bases cached")


if __name__ == "__main__":
    main()
