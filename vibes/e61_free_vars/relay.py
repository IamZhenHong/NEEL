"""Local relay for the pilot pod (runs alongside launch.sh): waits for the pod's generation, pulls it, runs the Haiku
pairing filter + register match (standard API, thread-pooled, so the pod waits minutes rather than a batch window),
uploads data/train/*.jsonl and READY, then runs the judge validation (gpt-5-mini) from the cached labels.

  caffeinate -i uv run --with anthropic python vibes/e61_free_vars/relay.py --log vibes/e61_free_vars/launch_pilot.log
The ssh target is read from the launcher log ("[pod] ssh root@HOST -p PORT"); nothing else on the account is touched."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

E61 = Path(__file__).resolve().parent
REMOTE = "/root/NEEL/vibes/e61_free_vars"
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", "-i", os.path.expanduser("~/.ssh/id_ed25519")]


def target(log):
    m = re.findall(r"\[pod\] ssh root@([\d.]+) -p (\d+)", Path(log).read_text(errors="replace")) if Path(log).exists() else []
    return (m[-1][0], int(m[-1][1])) if m else None


def ssh(host, port, cmd, timeout=90):
    return subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd], capture_output=True, text=True,
                          timeout=timeout)


def scp(args, timeout=600):
    return subprocess.run(["scp", *SSH_OPTS, *args], capture_output=True, text=True, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--give-up-min", type=float, default=120)
    a = ap.parse_args()
    t0 = time.time()
    host = port = None
    while True:
        if time.time() - t0 > a.give_up_min * 60:
            sys.exit("[relay] gave up waiting for the pod's generation")
        tg = target(a.log)
        if tg:
            host, port = tg
            try:
                r = ssh(host, port, f"test -f {REMOTE}/data/gen.done && echo yes")
                if r.stdout.strip() == "yes":
                    break
            except subprocess.TimeoutExpired:
                pass
        time.sleep(a.poll)
    print(f"[relay] generation done on {host}:{port} after {time.time() - t0:.0f}s; pulling", flush=True)
    (E61 / "data").mkdir(exist_ok=True)
    for attempt in range(4):
        r = scp(["-P", str(port), f"root@{host}:{REMOTE}/data/gen_raw.jsonl", str(E61 / "data" / "gen_raw.jsonl")])
        if r.returncode == 0:
            break
        time.sleep(15)
    else:
        sys.exit(f"[relay] pull failed: {r.stderr[-300:]}")
    n = sum(1 for _ in open(E61 / "data" / "gen_raw.jsonl"))
    print(f"[relay] pulled gen_raw.jsonl ({n} rows); running the pairing filter", flush=True)
    py = [sys.executable, str(E61 / "pairing.py"), "--direct"]
    r = subprocess.run(py + ["--no-validate"], text=True)
    if r.returncode != 0:
        sys.exit("[relay] pairing failed; the pod keeps waiting until WAIT_MIN, then exits")
    files = sorted((E61 / "data" / "train").glob("*.jsonl"))
    ssh(host, port, f"mkdir -p {REMOTE}/data/train")
    r = scp(["-P", str(port), *map(str, files), f"root@{host}:{REMOTE}/data/train/"])
    if r.returncode != 0:
        sys.exit(f"[relay] upload failed: {r.stderr[-300:]}")
    r = ssh(host, port, f"cd {REMOTE}/data/train && sha256sum *.jsonl && touch READY")
    print(f"[relay] uploaded {len(files)} training sets; pod sha256:\n{r.stdout}", flush=True)
    print("[relay] validation (gpt-5-mini) from cached labels", flush=True)
    subprocess.run(py, text=True)
    print(f"[relay] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
