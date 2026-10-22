"""Run vigilante's vLLM pod runner after importing the minis' case modules, which register their cases and conds.

    /root/vllm/bin/python -m persona_affect.common.pod_gen --register persona_affect.p12_named_traits.cases [MOD ...] \\
        -- --jobs JOBS --out OUT --model MODEL --fmt qwen|gemma4 --max-model-len N

Everything after `--` is the command line of vigilante.src.pod_run, handed on unchanged. The modules are imported in the
vLLM venv (not the uv project env), so they should import little: stdlib, numpy, vigilante, persona_affect.common.vig.
"""
from __future__ import annotations

import argparse
import importlib
import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="python -m persona_affect.common.pod_gen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", nargs="*", action="extend", default=[], metavar="MOD",
                    help="modules to import first (each calls persona_affect.common.vig.register)")
    head, tail = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, None)
    a, extra = ap.parse_known_args(head)
    if tail is None:
        ap.error("put vigilante.src.pod_run's arguments after `--`")
    if extra:
        ap.error(f"unexpected arguments before `--`: {extra}")

    import vigilante.cases as vcases
    from vigilante.src import personas
    for mod in a.register:
        n_cases, n_conds = len(vcases.ALL), len(personas.FAMOUS)
        importlib.import_module(mod)
        print(f"[gen] imported {mod}: +{len(vcases.ALL) - n_cases} cases, +{len(personas.FAMOUS) - n_conds} conds",
              flush=True)

    from vigilante.src import pod_run          # looked up here so a test can stand in for pod_run.main
    sys.argv = [*sys.argv[:1], *tail]
    pod_run.main()


if __name__ == "__main__":
    main()
