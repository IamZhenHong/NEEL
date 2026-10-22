"""Run infra/launch_run.py so that SIGTERM and SIGHUP stop it the way Ctrl-C does. launch_gen.sh starts the launcher
through this:

    uv run python persona_affect/common/launch_wrap.py <infra/launch_run.py arguments>

launch_run.py has no handler for either signal, so by default a TERM (a harness stop, a TaskStop) or a HUP (a closed
terminal) would end it at once and skip its `finally`, which terminates the pod and prints "[runpod] terminated <id>;
elapsed cost ~$X": the pod would bill until its boot self-destruct, and the booking would have only an upper bound.
Here the first TERM or HUP raises KeyboardInterrupt, which launch_run.py handles as it handles Ctrl-C, and every later
one is ignored so that the cleanup is not cut short: `uv run` passes a TERM sent to the process group on to its child,
so the launcher gets that TERM twice. Ctrl-C (INT) keeps its usual meaning. The exit code is launch_run.py's, or 128 +
the signal after a stop by TERM or HUP. infra/launch_run.py itself is not edited.
"""
from __future__ import annotations

import runpy
import signal
import sys
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[2] / "infra" / "launch_run.py"
STOPS = tuple(s for s in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)) if s is not None)
_stopped: list[int] = []


def _stop(signum, frame):
    if _stopped:                               # a stop is under way: let the launcher's cleanup finish
        return
    _stopped.append(signum)
    for s in STOPS:
        signal.signal(s, signal.SIG_IGN)
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    for s in STOPS:
        signal.signal(s, _stop)
    sys.argv = [str(LAUNCHER), *argv]
    sys.path.insert(0, str(LAUNCHER.parent))   # what `python infra/launch_run.py` puts first on the path
    try:
        runpy.run_path(str(LAUNCHER), run_name="__main__")
    except KeyboardInterrupt:
        if not _stopped:
            raise                              # a Ctrl-C: exit the way Python does on one
        print(f"[launch] stopped by {signal.Signals(_stopped[0]).name}: the launcher's cleanup has run", file=sys.stderr,
              flush=True)
        raise SystemExit(128 + _stopped[0]) from None


if __name__ == "__main__":
    main()
