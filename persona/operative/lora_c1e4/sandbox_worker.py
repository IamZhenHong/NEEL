"""Worker for sandbox.py (run as `python -I sandbox_worker.py`): JSON lines in {j, code, inp, resp}, JSON lines out
{j, ok} with ok True / False / None. Whitelisted builtins; imports limited to five stdlib modules; 2 s per call."""
import builtins
import csv
import json
import math
import re
import signal
import string
import sys

MODS = {"re": re, "json": json, "csv": csv, "string": string, "math": math}
SAFE = ["abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "hex", "int", "isinstance", "iter", "len", "list", "map", "max", "min", "next", "oct", "ord", "pow", "print",
        "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip", "callable",
        "bin", "bytes", "Exception", "ValueError", "KeyError", "IndexError", "TypeError", "AttributeError",
        "StopIteration", "ZeroDivisionError", "OverflowError", "True", "False", "None"]


def _imp(name, globals=None, locals=None, fromlist=(), level=0):
    if name in MODS:
        return MODS[name]
    raise ImportError(name)


def _alarm(signum, frame):
    raise TimeoutError()


def main():
    safe = {k: getattr(builtins, k) for k in SAFE if hasattr(builtins, k)}
    safe["__import__"] = _imp
    safe["__build_class__"] = builtins.__build_class__
    signal.signal(signal.SIGALRM, _alarm)
    for line in sys.stdin:
        x = json.loads(line)
        ok = None
        try:
            signal.alarm(2)
            g = {"__builtins__": safe, "__name__": "grader"}
            exec(x["code"], g)                                       # noqa: S102  (checked code, restricted builtins)
            r = g["grade_output_correct"](x["inp"], x["resp"])
            signal.alarm(0)
            if isinstance(r, bool):
                ok = r
            elif isinstance(r, dict) and isinstance(r.get("passed"), bool):
                ok = r["passed"]
        except BaseException:                                        # noqa: BLE001
            signal.alarm(0)
            ok = None
        sys.stdout.write(json.dumps({"j": x["j"], "ok": ok}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
