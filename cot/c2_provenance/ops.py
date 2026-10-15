"""Operator helpers for the running c2 pod (ssh target read from launch.log).

  uv run python -m cot.c2_provenance.ops pull            # copy results/* from the pod (mid-run)
  uv run python -m cot.c2_provenance.ops put <repo path> # upload one file before the stage that imports it
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from cot.c2_provenance.common import HERE

OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=30", "-i", str(Path("~/.ssh/id_ed25519").expanduser())]
REMOTE = "/root/NEEL/cot/c2_provenance"


def target():
    m = re.findall(r"\[pod\] ssh root@(\S+) -p (\d+)", (HERE / "launch.log").read_text())
    if not m:
        sys.exit("no ssh target in launch.log yet")
    return m[-1]


def ssh(cmd):
    h, p = target()
    return subprocess.run(["ssh", *OPTS, "-p", p, f"root@{h}", cmd], capture_output=True, text=True, timeout=120)


def main():
    what = sys.argv[1]
    h, p = target()
    if what == "pull":
        (HERE / "results").mkdir(exist_ok=True)
        r = subprocess.run(f"ssh {' '.join(OPTS)} -p {p} root@{h} 'cd {REMOTE} && tar -czf - results' | tar -xzf - -C {HERE}",
                           shell=True, capture_output=True, text=True, timeout=600)
        print(r.returncode, r.stderr[-300:])
        for f in sorted((HERE / "results").iterdir()):
            print(f"  {f.name} {f.stat().st_size}")
    elif what == "put":   # upload a code file (fix before GO): ops put cot/c2_provenance/common.py
        src = sys.argv[2]
        r = subprocess.run(["scp", *OPTS, "-P", p, src, f"root@{h}:/root/NEEL/{src}"], capture_output=True, text=True)
        print("scp", r.returncode, r.stderr[-200:])


if __name__ == "__main__":
    main()
