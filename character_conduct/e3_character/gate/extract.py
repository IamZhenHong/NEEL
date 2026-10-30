"""CC-E3: inspect AgentHarm logs (.eval) -> plain JSONL, one row per sample x epoch, so scoring runs in the project venv
without inspect. Run with inspect available, e.g.

  uv run --no-project --python 3.12 --with inspect-ai==0.3.285 --with inspect-evals==0.26.0 \
      python -I character_conduct/e3_character/gate/extract.py --logs DIR --out FILE.jsonl --model-tag qwen

Row: task ("harmful" / "benign"), model, id, epoch, metadata (id_original, name, category, detailed_prompt,
hint_included, target_functions, grading_function), messages [{role, text, tool_calls: [{function, arguments}],
function, error}], error (sample error or null), limit (message limit hit or null).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

META = ("id_original", "name", "category", "detailed_prompt", "hint_included", "target_functions", "grading_function")


def text_of(m) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    return "".join(getattr(x, "text", "") or "" for x in c)


def msg(m) -> dict:
    d = {"role": m.role, "text": text_of(m)}
    if m.role == "assistant":
        d["tool_calls"] = [{"function": tc.function, "arguments": tc.arguments} for tc in (m.tool_calls or [])]
    if m.role == "tool":
        d["function"] = m.function
        d["error"] = m.error.message if m.error else None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-tag", required=True)
    a = ap.parse_args()
    rows = []
    for p in sorted(a.logs.glob("*.eval")):
        log = read_eval_log(str(p))
        task = "benign" if log.eval.task.endswith("agentharm_benign") else "harmful"
        for s in log.samples or []:
            rows.append({"task": task, "model": a.model_tag, "id": str(s.id), "epoch": s.epoch, "log": p.name,
                         "metadata": {k: s.metadata.get(k) for k in META},
                         "messages": [msg(m) for m in s.messages],
                         "error": s.error.message if s.error else None,
                         "limit": s.limit.type if getattr(s, "limit", None) else None})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    by = {}
    for r in rows:
        by[r["task"]] = by.get(r["task"], 0) + 1
    print(f"[extract] {a.model_tag}: {by} -> {a.out}")


if __name__ == "__main__":
    main()
