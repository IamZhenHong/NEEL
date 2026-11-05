"""Build the prompt file for an open-weight judge run (prereg retention/prereg/retention_openjudge_v1.md).

Every judged reply of the November runs (support x mood, Pod A, soft exits) with exactly the system prompt and user payload
the production judges saw: RUBRIC (cd/judge.py) for goodbye replies, SOFTEXIT_RUBRIC for soft-exit replies, payload =
cd.judge._payload(cell context, reply). The 100 judge-calibration items (labeling/judgecal_v1) are November replies, so
they are inside this set and are matched later by run/cell/idx.
  python3 retention/open_judge_prep.py -> retention/data/open_judge/november_v1.jsonl"""
from __future__ import annotations
import glob, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = {"retention_userstate2x2": "goodbye", "retention_podA": "goodbye", "retention_softexit": "softexit"}


def rubric(path, name):
    m = re.search(name + r' = """(.*?)"""', (HERE / path).read_text(), re.S)
    return m.group(1)


def main():
    rub = {"goodbye": rubric("cd/judge.py", "RUBRIC"), "softexit": rubric("judge_softexit.py", "SOFTEXIT_RUBRIC")}
    out = HERE / "data/open_judge/november_v1.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as f:
        for run, kind in RUNS.items():
            for p in sorted(glob.glob(str(HERE / "results" / run / "cells" / "*.json"))):
                cell = Path(p).stem
                for i, it in enumerate(json.load(open(p))):
                    if not it.get("scores"):
                        continue
                    user = f"CONVERSATION CONTEXT (most recent last):\n{it['context']}\n\nASSISTANT REPLY TO SCORE:\n{it['reply']}"
                    f.write(json.dumps({"id": f"{run}/{cell}/{i}", "kind": kind, "system": rub[kind], "user": user}) + "\n"); n += 1
    print(f"{n} prompts -> {out}")


if __name__ == "__main__":
    sys.exit(main())
