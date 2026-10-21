"""Runs IH-Challenge grader code (untrusted, from the dataset) safely enough for a laptop:
1. static check: imports only re / json / csv / string / math; no exec / eval / open / getattr-style names; no
   attribute or name starting with "_" that is a dunder;
2. each grader runs in a separate `python -I` worker (sandbox_worker.py) with whitelisted builtins, an import hook that
   admits only those five modules, and a 2 s alarm per call.
A grader that fails the check, raises, times out or returns something unreadable counts as a grader error (excluded,
never a pass or a fail)."""
import ast
import json
import subprocess
import sys
from pathlib import Path

ALLOWED_IMPORTS = {"re", "json", "csv", "string", "math"}
BANNED = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars",
          "input", "breakpoint", "exit", "quit", "help", "memoryview", "__builtins__", "classmethod", "staticmethod",
          "super", "object"}
WORKER = Path(__file__).resolve().parent / "sandbox_worker.py"


def check(code):
    try:
        t = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax {e}"
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            bad = [a.name for a in n.names if a.name not in ALLOWED_IMPORTS]
            if bad:
                return False, f"import {bad}"
        elif isinstance(n, ast.ImportFrom):
            if n.module not in ALLOWED_IMPORTS:
                return False, f"from {n.module}"
        elif isinstance(n, ast.Name) and (n.id in BANNED or n.id.startswith("__")):
            return False, f"name {n.id}"
        elif isinstance(n, ast.Attribute) and n.attr.startswith("__"):
            return False, f"attr {n.attr}"
    names = {n.name for n in t.body if isinstance(n, ast.FunctionDef)}
    if "grade_output_correct" not in names:
        return False, "no grade_output_correct"
    return True, ""


def grade_many(jobs, timeout=600):
    """jobs: [(code, input_text, response)] -> [True | False | None] (None = grader error)."""
    out = [None] * len(jobs)
    todo = []
    for j, (code, inp, resp) in enumerate(jobs):
        ok, _ = check(code)
        if ok:
            todo.append(j)
    for s in range(0, len(todo), 200):
        chunk = todo[s:s + 200]
        payload = "\n".join(json.dumps({"j": j, "code": jobs[j][0], "inp": jobs[j][1], "resp": jobs[j][2]})
                            for j in chunk) + "\n"
        r = subprocess.run([sys.executable, "-I", str(WORKER)], input=payload, capture_output=True, text=True,
                           timeout=timeout)
        for line in r.stdout.splitlines():
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[x["j"]] = x["ok"]
    return out
