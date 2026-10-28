"""Outage-tolerant pull for a detached pod job (2026-10-28): for a pod whose launcher was stopped by hand.

  uv run python infra/rescue_pull.py --pod-id ID --host H --port P --rdir character_conduct/e2_selfuse/results/q14 \\
      --deadline-epoch EPOCH [--poll 120]

It polls /root/job.exit over ssh and tolerates any number of failed polls (network down) until DEADLINE-EPOCH. That's
the time after which the pod's own boot self-destruct may have fired, so set it a few minutes before. When the job
has exited, it pulls RDIR (tar over ssh, adapter weights excluded; retried until the deadline), terminates the pod by
id, and prints "[runpod] terminated ID; elapsed cost ~$X" from RunPod's uptime x costPerHr, which persona_affect's
book_pod.sh can read. It never terminates a pod whose job is still running.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_run import SSH_OPTS  # noqa: E402
from runpod_guard import rest, terminate  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def ssh(host, port, cmd, timeout=60):
    try:
        return subprocess.run(["ssh", *SSH_OPTS, "-o", "ServerAliveInterval=30", "-p", str(port), f"root@{host}", cmd],
                              capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def target(pod_id):
    """(host, port) from RunPod, or None when unreachable."""
    try:
        g = rest("GET", f"/pods/{pod_id}")
    except (OSError, RuntimeError):
        return None
    host, ports = g.get("publicIp"), g.get("portMappings") or {}
    port = ports.get("22") or ports.get(22)
    return (host, int(port)) if host and port else None


def pull(host, port, rdir):
    remote = f"tar -C /root/NEEL -czf - --exclude=adapter_model.safetensors {rdir}"
    try:
        p1 = subprocess.Popen(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", remote], stdout=subprocess.PIPE)
        p2 = subprocess.run(["tar", "-C", str(REPO), "-xzf", "-"], stdin=p1.stdout, timeout=1800)
        p1.wait(timeout=60)
        return p1.returncode == 0 and p2.returncode == 0
    except Exception as e:  # noqa: BLE001
        print(f"[rescue] pull error: {e}", flush=True)
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-id", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--rdir", required=True)
    ap.add_argument("--deadline-epoch", type=int, required=True)
    ap.add_argument("--poll", type=int, default=120)
    a = ap.parse_args(argv)
    host, port, fails = a.host, a.port, 0
    while time.time() < a.deadline_epoch:
        r = ssh(host, port, "cat /root/job.exit 2>/dev/null; tail -c 300 /root/job.log 2>/dev/null")
        if r is None or r.returncode not in (0, 1):
            fails += 1
            print(f"[rescue] {time.strftime('%H:%M:%S')} no contact ({fails}); job keeps running", flush=True)
            if fails % 5 == 0:
                t = target(a.pod_id)
                if t:
                    host, port = t
            time.sleep(a.poll)
            continue
        fails = 0
        out = r.stdout.decode("utf-8", "replace").strip().splitlines()
        rc = out[0].strip() if out and out[0].strip().lstrip("-").isdigit() else None
        if rc is None:
            print(f"[rescue] {time.strftime('%H:%M:%S')} running: {(out[-1] if out else '')[:160]}", flush=True)
            time.sleep(a.poll)
            continue
        print(f"[rescue] job exited rc {rc}; pulling {a.rdir}", flush=True)
        while time.time() < a.deadline_epoch and not pull(host, port, a.rdir):
            print("[rescue] pull failed; retrying in 60 s", flush=True)
            time.sleep(60)
            host, port = target(a.pod_id) or (host, port)
        try:
            g = rest("GET", f"/pods/{a.pod_id}")
            cost = (g.get("runtime") or {}).get("uptimeInSeconds", 0) / 3600 * float(g.get("costPerHr") or 0)
        except Exception:  # noqa: BLE001
            cost = None
        terminate(a.pod_id)
        print(f"[runpod] terminated {a.pod_id}; elapsed cost ~${cost:.3f}" if cost is not None else
              f"[runpod] terminated {a.pod_id}; elapsed cost unknown", flush=True)
        return 0
    print("[rescue] deadline reached before the job finished or before the pull succeeded; the pod self-destructs",
          flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
