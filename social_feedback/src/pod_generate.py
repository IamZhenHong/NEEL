"""Generic generation job for the pod (DESIGN.md §3): runs prebuilt jobs (build_jobs.py) with the two-stage protocol.

    /root/vllm/bin/python -m social_feedback.src.pod_generate --jobs social_feedback/materials/jobs/dev_v1/jobs.jsonl \
        --out social_feedback/results/dev_v1 [--thinking-cap N]

Without --thinking-cap, thinking is limited only by the context (the dev sample, §6.3).
"""
import argparse
import json
from pathlib import Path

from .common import load_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking-cap", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = load_jsonl(a.jobs)
    from .qwen_gen import Qwen
    gen = Qwen()
    (out / "settings.json").write_text(json.dumps({**gen.settings, "thinking_cap": a.thinking_cap,
                                                   "jobs": a.jobs, "n_jobs": len(jobs)}, indent=1))
    print(f"[generate] {len(jobs)} jobs, thinking cap {a.thinking_cap}", flush=True)
    res = gen.generate(jobs, thinking_cap=a.thinking_cap)
    write_jsonl(out / "generations.jsonl", [{"key": j["key"], **j["meta"], **res[j["key"]]} for j in jobs])
    print("[generate] done", flush=True)


if __name__ == "__main__":
    main()
