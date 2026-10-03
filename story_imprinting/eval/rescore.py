"""Re-apply the current detector to an existing responses file (all rows, all conditions).
Writes <file> in place after saving the previous version as <file>.v<old_detector_version>.bak.
Usage: uv run python story_imprinting/eval/rescore.py results/.../base_dev.jsonl [...]
"""
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "scoring"))
from detector import DETECTOR_VERSION, score_response
from run_eval import ASSIGNMENTS

for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.strip()]
    old = rows[0]["scores"]["detector_version"] if rows else "?"
    if old == DETECTOR_VERSION:
        print(f"{f}: already at {DETECTOR_VERSION}"); continue
    shutil.copy(f, f"{f}.v{old}.bak")
    changed = 0
    with open(f, "w") as out:
        for r in rows:
            new = score_response(r["response"], r["messages"][-1]["content"], r.get("forbidden_regex"), r.get("finish_reason"))
            changed += new != r["scores"]
            r["scores"] = new
            m = ASSIGNMENTS.get(r.get("assignment"))
            if m:
                r["Y_H"] = new[f"tracer_{m['helpful']}_enacted"]; r["Y_D"] = new[f"tracer_{m['dismissive']}_enacted"]
            out.write(json.dumps(r) + "\n")
    print(f"{f}: rescored {len(rows)} rows {old} -> {DETECTOR_VERSION}; {changed} rows changed")
