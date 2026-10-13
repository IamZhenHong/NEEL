"""R21 (red-team follow-up to R19b), Gemma 4 31B, thinking off, 10 per cell.
  jobs_hist.jsonl : the 16 ordinary customer chats alone, 3 samples per case (C1, C3), to record the long history.
  jobs_a.jsonl    : short grid, C1/C3 x message (supervisor asks / supervisor "approved" / peer "approved" / bare
                    record) x line (none / old / new / both) = 320; real approval returned by request_approval x line
                    = 80.
  jobs_long.jsonl : written on the pod by make_long.py from the recorded history.
  uv run python -m vigilante.exp.r21_approval_fix.build_jobs"""
import hashlib
import json
from pathlib import Path

from vigilante.cases import ALL
from vigilante.cases.authority import APPROVAL, LINES, R21_MSGS

H = Path(__file__).parent


def job(case, i, exp="R21", **kw):
    assert case in ALL, case
    key = f"{exp}|P0|{case}|{i}"
    return {"key": key, "exp": exp, "group": "agent", "cond": "P0", "case": case, "fmt": "gemma4",
            "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i, "thinking": False,
            "handover": True, **kw}


def write(name, js):
    with open(H / name, "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"{name}: {len(js)} jobs")


def main():
    write("jobs_hist.jsonl", [job(f"{c}#r21-hist", i, exp="R21H", handover=False) for c in APPROVAL for i in range(3)])
    cells = [f"{c}#r21-{m}-{ln}" for c in APPROVAL for m in R21_MSGS for ln in LINES]
    cells += [f"{c}#r21-legit-{ln}" for c in APPROVAL for ln in LINES]
    write("jobs_a.jsonl", [job(c, i) for c in cells for i in range(10)])


if __name__ == "__main__":
    main()
