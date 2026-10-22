"""Run vigilante's vLLM pod runner after importing the minis' case modules, which register their cases and conds.

    /root/vllm/bin/python -m persona_affect.common.pod_gen --register persona_affect.p12_named_traits.cases [MOD ...] \\
        -- --jobs JOBS --out OUT --model MODEL --fmt qwen|gemma4 --max-model-len N

Everything after `--` is the command line of vigilante.src.pod_run, handed on unchanged. The modules are imported in the
vLLM venv (not the uv project env), so they should import little: stdlib, numpy, vigilante, persona_affect.common.vig.

    uv run python -m persona_affect.common.pod_gen --register MOD [MOD ...] --check JOBS --fmt qwen|gemma4

checks a jobs file here, without vLLM, and exits (launch_gen.sh runs it before it launches a pod): after the imports,
every job must have the pod's fmt (a job without "fmt" is qwen; the episode parser reads the job's fmt, the template
conversion the pod's --fmt, so a mismatch silently mis-parses or mis-renders every turn) and init_episode must accept
it, so an unregistered case or cond id, or a REGISTER module that imports but registers nothing, fails here rather
than on the pod after the vLLM setup and the model load.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

SHOW = 5        # examples per kind of problem in a --check report


def check_jobs(path: str, fmt: str) -> tuple[int, list[str]]:
    """(number of jobs, problems) for the jobs file at `path` on a pod run with --fmt `fmt`. Reads the file the way
    pod_run does (one JSON object per non-blank line); problems is empty when every job is fine."""
    from vigilante.src.episode import init_episode
    from vigilante.src.sim import PARSERS

    problems, bad_line, bad_fmt, bad_init, n = [], [], [], [], 0
    if fmt not in PARSERS:
        problems.append(f"--fmt {fmt!r} is not a format the engine knows ({', '.join(sorted(PARSERS))})")
    with open(path, encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                job = json.loads(line)
            except ValueError as e:
                bad_line.append(f"line {no}: not JSON ({e})")
                continue
            if not isinstance(job, dict):
                bad_line.append(f"line {no}: not a JSON object")
                continue
            n += 1
            key = job.get("key", f"line {no}")
            if job.get("fmt", "qwen") != fmt:
                bad_fmt.append(f"{key!r} has fmt {job.get('fmt', 'qwen (none given)')!r}")
            try:
                init_episode(job)
            except Exception as e:             # an unregistered case (KeyError) or cond, a missing field, ...
                bad_init.append(f"{key!r}: {type(e).__name__}: {e}")
    if bad_line:
        problems.append(f"{len(bad_line)} unreadable line(s): " + "; ".join(bad_line[:SHOW]))
    if n == 0 and not bad_line:
        problems.append("no jobs in the file")
    if bad_fmt:
        problems.append(f"{len(bad_fmt)} of {n} jobs do not have the pod's fmt {fmt!r} (a job without one is qwen), "
                        f"e.g. " + "; ".join(bad_fmt[:SHOW]))
    if bad_init:
        problems.append(f"{len(bad_init)} of {n} jobs fail init_episode (an unregistered case or cond id, a missing "
                        f"field), e.g. " + "; ".join(bad_init[:SHOW]))
    return n, problems


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="python -m persona_affect.common.pod_gen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", nargs="*", action="extend", default=[], metavar="MOD",
                    help="modules to import first (each calls persona_affect.common.vig.register)")
    ap.add_argument("--check", metavar="JOBS", help="check this jobs file without vLLM and exit (see above)")
    ap.add_argument("--fmt", help="with --check: the pod's --fmt (default qwen)")
    head, tail = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, None)
    a, extra = ap.parse_known_args(head)
    if extra:
        ap.error(f"unexpected arguments before `--`: {extra}")
    if a.check is None:
        if a.fmt is not None:
            ap.error("--fmt goes with --check; pod_run's own --fmt goes after `--`")
        if tail is None:
            ap.error("put vigilante.src.pod_run's arguments after `--`")
    elif tail is not None:
        ap.error("--check runs no pod: give no pod_run arguments after `--`")

    import vigilante.cases as vcases
    from vigilante.src import personas
    for mod in a.register:
        n_cases, n_conds = len(vcases.ALL), len(personas.FAMOUS)
        importlib.import_module(mod)
        print(f"[gen] imported {mod}: +{len(vcases.ALL) - n_cases} cases, +{len(personas.FAMOUS) - n_conds} conds",
              flush=True)

    if a.check is not None:
        fmt = a.fmt or "qwen"
        n, problems = check_jobs(a.check, fmt)
        if problems:
            raise SystemExit(f"[gen] check FAILED for {a.check} (--fmt {fmt}, --register {' '.join(a.register) or '-'}):"
                             + "".join(f"\n  {p}" for p in problems))
        print(f"[gen] check ok: {n} jobs in {a.check}, every one with fmt {fmt} and a case and cond that are registered",
              flush=True)
        return

    from vigilante.src import pod_run          # looked up here so a test can stand in for pod_run.main
    sys.argv = [*sys.argv[:1], *tail]
    pod_run.main()


if __name__ == "__main__":
    main()
