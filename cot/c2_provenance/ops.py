"""Operator helpers for the running c2 pod (ssh target read from launch.log).

  uv run python -m cot.c2_provenance.ops pull            # copy results/* from the pod (mid-run)
  uv run python -m cot.c2_provenance.ops go              # upload data/items_main.json.gz, then run/GO (sha256 inside)
  uv run python -m cot.c2_provenance.ops stop            # write run/STOP (gate failed: no main run)
"""
import hashlib
import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

from cot.c2_provenance.common import DATA, HERE

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
    elif what == "go":
        gz = DATA / "items_main.json.gz"
        sha = hashlib.sha256(gzip.open(gz).read()).hexdigest()
        print(ssh(f"mkdir -p {REMOTE}/run").returncode)
        r = subprocess.run(["scp", *OPTS, "-P", p, str(gz), f"root@{h}:{REMOTE}/run/items_main.json.gz"],
                           capture_output=True, text=True, timeout=600)
        print("scp", r.returncode, r.stderr[-200:])
        r = ssh(f"sha256sum {REMOTE}/run/items_main.json.gz; echo '{json.dumps({'items_sha256': sha})}' > {REMOTE}/run/GO.tmp "
                f"&& mv {REMOTE}/run/GO.tmp {REMOTE}/run/GO && cat {REMOTE}/run/GO")
        print(r.stdout, r.stderr[-200:])
    elif what == "gosha":   # the pod built the same item set: GO names its sha256, no upload
        sha = sys.argv[2]
        r = ssh(f"sha256sum {REMOTE}/results/build/items_main.json; echo '{json.dumps({'items_sha256': sha})}' > "
                f"{REMOTE}/run/GO.tmp && mv {REMOTE}/run/GO.tmp {REMOTE}/run/GO && cat {REMOTE}/run/GO")
        print(r.stdout, r.stderr[-200:])
    elif what == "stop":
        print(ssh(f"mkdir -p {REMOTE}/run && touch {REMOTE}/run/STOP").returncode)
    elif what == "put":   # upload a code file (fix before GO): ops put cot/c2_provenance/common.py
        src = sys.argv[2]
        r = subprocess.run(["scp", *OPTS, "-P", p, src, f"root@{h}:/root/NEEL/{src}"], capture_output=True, text=True)
        print("scp", r.returncode, r.stderr[-200:])


if __name__ == "__main__":
    main()
