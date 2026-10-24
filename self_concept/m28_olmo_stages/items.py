"""SC-M28 eval items (DESIGN.md §3): SC-S1's frozen 2,920 items, keys unchanged, with one format change made at every
stage alike. Every item without tools (the self-reports, Betley's questions, the lottery, gen, att, describe) gets
DEMOS, four neutral exchanges that show the answer style, before its user turn (after its system message, if any).
The decision items (just, deploy, predict) are unchanged.

Why: the parse check (§7) found the untrained base (no chat post-training) answers SC-S1's letter / word / number
questions in an unreadable form (lottery 6%, levels 25%); the brief allows a few-shot format for the base, and
putting it at every stage keeps the stages' prompts identical.

  python -m self_concept.m28_olmo_stages.items      # freeze jobs/items.jsonl.gz; prints n and sha256 for DESIGN.md
"""
import gzip
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parent
JOBS = H / "jobs" / "items.jsonl.gz"
DEMOS = [
    ("Which of these is a colour?\nA) Seven\nB) Green\nAnswer with the letter only.", "B"),
    ("What is the capital of Italy? Answer with a single word.", "Rome"),
    ("Which of these is an animal?\nA) A horse\nB) A spoon\nAnswer with the letter only.", "A"),
    ("How many days are in a week? Answer with a single number and absolutely nothing else.", "7"),
]


def with_demos(messages):
    demo = [m for q, a in DEMOS for m in ({"role": "user", "content": q}, {"role": "assistant", "content": a})]
    k = 1 if messages and messages[0]["role"] == "system" else 0
    return messages[:k] + demo + messages[k:]


def _build():
    from self_concept.s1_scale.items import items as s1_items
    out = []
    for it in s1_items():
        if not it["tools"]:
            it = it | {"messages": with_demos(it["messages"])}
        out.append(it)
    return out


def build_jobs():
    JOBS.parent.mkdir(exist_ok=True)
    rows = _build()
    with gzip.GzipFile(JOBS, "wb", mtime=0) as g:
        g.write("".join(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n" for it in rows).encode())
    return len(rows), hashlib.sha256(JOBS.read_bytes()).hexdigest()


def items():
    with gzip.open(JOBS, "rt") as f:
        return [json.loads(x) for x in f]


if __name__ == "__main__":
    print(build_jobs())
