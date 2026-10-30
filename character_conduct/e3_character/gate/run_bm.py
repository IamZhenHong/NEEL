"""CC-E3 step 0 (pod): run the blackmail jobs against the local vLLM server (OpenAI-compatible, served as "m").

One completion per job, with the job's fixed seed; temperature / top_p / max_tokens as DESIGN.md §Step 0. Resumable:
keys already in --out are skipped, and each row is written as soon as it returns (the launcher's stall guard sees
progress lines).

  /root/vllm/bin/python -m character_conduct.e3_character.gate.run_bm --jobs JOBS --out OUT --model-tag qwen
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

TEMPERATURE, TOP_P, MAX_TOKENS = 0.7, 0.9, 4000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    jobs = [json.loads(x) for x in a.jobs.read_text().splitlines() if x.strip()][:a.limit]
    done = set()
    if a.out.exists():
        done = {json.loads(x)["key"] for x in a.out.read_text().splitlines() if x.strip()}
    todo = [j for j in jobs if j["key"] not in done]
    print(f"[bm] {a.model_tag}: {len(todo)} to run ({len(done)} done)", flush=True)
    client = OpenAI(base_url=a.base_url, api_key="x", timeout=600)
    lock, t0, n = threading.Lock(), time.time(), 0

    def one(j):
        r = client.chat.completions.create(model="m", messages=j["messages"], temperature=TEMPERATURE, top_p=TOP_P,
                                           max_tokens=MAX_TOKENS, seed=j["seed"])
        c = r.choices[0]
        return {"key": j["key"], "cell": j["cell"], "sample": j["sample"], "seed": j["seed"], "model": a.model_tag,
                "text": c.message.content, "finish_reason": c.finish_reason,
                "completion_tokens": r.usage.completion_tokens if r.usage else None}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(one, j) for j in todo]):
            row = fut.result()
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 20 == 0 or n == len(todo):
                    print(f"[bm] {a.model_tag}: {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
