"""Print episodes compactly for reading.

  uv run python -m vigilante.src.show RESULTS_DIR [--exp Q2] [--cond P4] [--case A1] [--unauth 1] [--n 3] [--seed 0]
      [--full]   (full tool outputs and reasoning; default truncates them)
"""
import argparse
import json
import random
from pathlib import Path


def fmt(rec, full=False, width=400):
    j = rec["job"]
    cut = (lambda s: s) if full else (lambda s: s if len(s) <= width else s[:width] + f" ...[{len(s) - width} more]")
    out = [f"######## {rec['key']}  end={rec['end']}"]
    for m in rec["messages"]:
        if m["role"] == "system":
            out.append("[system] " + cut(m["content"]).replace("\n", " | "))
        elif m["role"] == "user":
            out.append("[user] " + m["content"])
        elif m["role"] == "tool":
            out.append("  [tool->] " + cut(m["content"]).replace("\n", " | "))
        else:
            if m.get("reasoning_content"):
                out.append("  [think] " + cut(m["reasoning_content"]).replace("\n", " "))
            out.append("[agent] " + m["content"].strip())
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--exp")
    ap.add_argument("--cond")
    ap.add_argument("--case")
    ap.add_argument("--qid")
    ap.add_argument("--unauth", type=int, default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    res = Path(a.results)
    scores = {}
    if (res / "scores.jsonl").exists():
        scores = {r["key"]: r for r in (json.loads(l) for l in open(res / "scores.jsonl"))}
    recs = [json.loads(l) for p in sorted(res.glob("episodes_*.jsonl")) for l in open(p) if l.strip()]
    sel = [r for r in recs if (not a.exp or r["job"].get("exp") == a.exp) and (not a.cond or r["job"]["cond"] == a.cond)
           and (not a.case or r["job"].get("case") == a.case) and (not a.qid or r["job"].get("qid") == a.qid)
           and (a.unauth is None or bool(scores.get(r["key"], {}).get("unauth")) == bool(a.unauth))]
    random.Random(a.seed).shuffle(sel)
    print(f"{len(sel)} matching; showing {min(a.n, len(sel))}\n")
    for r in sel[: a.n]:
        print(fmt(r, a.full) + "\n")


if __name__ == "__main__":
    main()
