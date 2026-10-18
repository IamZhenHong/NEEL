"""J12 sandbox: write a workspace (dict path -> text) to a temp dir and run every test_* function in tests/*.py in a
fresh isolated Python process (timeout 10 s). Returns {test name: (passed, message)} in file order."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

RUNNER = r'''
import importlib.util, inspect, json, sys, traceback
root = sys.argv[1]
sys.path.insert(0, root)
res = []
for f in sorted(__import__("glob").glob(root + "/tests/test_*.py")):
    name = f.rsplit("/", 1)[1][:-3]
    try:
        spec = importlib.util.spec_from_file_location("tests." + name, f)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    except Exception as e:
        res.append([name + "::<import>", False, f"{type(e).__name__}: {e}"[:300]]); continue
    for k, fn in list(vars(m).items()):
        if k.startswith("test_") and callable(fn):
            try:
                fn(); res.append([k, True, ""])
            except AssertionError:
                src = [l.strip() for l in inspect.getsource(fn).splitlines() if l.strip().startswith("assert")]
                res.append([k, False, "AssertionError: " + (src[-1] if src else "")])
            except Exception as e:
                res.append([k, False, f"{type(e).__name__}: {e}"[:300]])
print("@@RESULT@@" + json.dumps(res))
'''


def run_suite(ws, override=None):
    files = dict(ws)
    if override is not None:
        files = {k: v for k, v in files.items() if not k.startswith("tests/")}
        files.update(override)
    with tempfile.TemporaryDirectory() as d:
        for p, txt in files.items():
            q = Path(d) / p
            if not q.resolve().is_relative_to(Path(d).resolve()):
                continue
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text(txt)
        (Path(d) / "tests").mkdir(exist_ok=True)
        (Path(d) / "_runner.py").write_text(RUNNER)
        try:
            p = subprocess.run([sys.executable, "-I", str(Path(d) / "_runner.py"), d], cwd=d, capture_output=True,
                               text=True, timeout=10)
            out = p.stdout.split("@@RESULT@@")
            if len(out) < 2:
                return {"<runner>": (False, (p.stderr or "no output")[-300:])}
            return {k: (ok, msg) for k, ok, msg in json.loads(out[-1])}
        except subprocess.TimeoutExpired:
            return {"<runner>": (False, "Timeout: tests took longer than 10 s")}
