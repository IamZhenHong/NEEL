"""Run a job on a pod the USER started by hand (no create/terminate here): sync code, bootstrap, run detached, pull results.

Usage (from repo root):
  uv run python infra/attach_run.py --host 1.2.3.4 --port 12345 --max-minutes 40 --ship retention \
      --results-dir retention/results/<run> -- python retention/run_x.py --out retention/results/<run>

Reuses launch_run.py's sync / bootstrap / detached-poll / verified-pull. Spend guard: the pod is the user's, so the
only guard here is a local wall-clock limit (stop waiting and pull at --max-minutes). Optional --pod-id + --terminate
deletes the pod via the RunPod API after a VERIFIED pull (only when the user has said so).
"""
from __future__ import annotations

import argparse, shlex, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_run import BOOTSTRAP, REPO, SSH_OPTS, pull_results, run_detached, ssh, wait_ssh  # noqa: E402
from runpod_guard import terminate  # noqa: E402


class _Stub:
    """The bits of PodSession that run_detached uses; budget = local wall clock only."""
    def __init__(self, host, port, pod_id, max_minutes):
        self.host, self.port, self.pod_id, self.max_minutes, self.t0 = host, port, pod_id or "unknown", max_minutes, time.time()

    def check_budget(self):
        if time.time() - self.t0 > self.max_minutes * 60:
            raise RuntimeError(f"local wall-clock limit {self.max_minutes} min reached")

    def ssh_target(self, timeout_s=60):
        return self.host, self.port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True); ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--pod-id", default=""); ap.add_argument("--terminate", action="store_true")
    ap.add_argument("--max-minutes", type=int, required=True)
    ap.add_argument("--ship", action="append", default=[], help="repo-relative paths to sync (pyproject.toml, uv.lock added)")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--skip-bootstrap", action="store_true", help="pod already bootstrapped by an earlier attach")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd or not a.ship:
        ap.error("need --ship paths and a command after --")
    if a.terminate and not a.pod_id:
        ap.error("--terminate needs --pod-id")
    host, port, stub = a.host, a.port, _Stub(a.host, a.port, a.pod_id, a.max_minutes)
    print(f"[attach] ssh root@{host} -p {port}", flush=True)
    wait_ssh(host, port)
    excl = ["--exclude=./.git", "--exclude=./.venv", "--exclude=./results", "--exclude=./logs", "--exclude=./.env",
            "--exclude=__pycache__", "--exclude=./stimgen/.venv"]
    paths = list(dict.fromkeys(f"./{p.strip('/')}" for p in [*a.ship, "pyproject.toml", "uv.lock"]))
    tar = subprocess.Popen(["tar", "--no-xattrs", "--no-mac-metadata", "-C", str(REPO), *excl, "-czf", "-", *paths], stdout=subprocess.PIPE)
    subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}",
                    "mkdir -p /root/NEEL && tar --no-same-owner --no-same-permissions --warning=no-unknown-keyword -xzf - -C /root/NEEL"],
                   stdin=tar.stdout, check=True)
    tar.wait(); print("[attach] code synced", flush=True)
    if not a.skip_bootstrap:
        ssh(host, port, BOOTSTRAP, timeout=15 * 60)
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    job = (f"cd /root/NEEL && export GIT_REV={rev} && export PATH=$HOME/.local/bin:$PATH HF_HOME=/root/hf PYTHONUNBUFFERED=1 "
           f"UV_PROJECT_ENVIRONMENT=/root/venv && uv run " + " ".join(shlex.quote(c) for c in cmd))
    rc = 1
    try:
        rc = run_detached(host, port, job, stub)
        print(f"[attach] job exit code {rc}; {(time.time() - stub.t0) / 60:.1f} min", flush=True)
    finally:
        ok = False
        for attempt in range(5):
            try:
                ok = pull_results(host, port, a.results_dir.strip("/"))
            except Exception as e:  # noqa: BLE001
                print(f"[attach] pull error: {e}", flush=True)
            if ok:
                break
            time.sleep(30)
        if ok and a.terminate:
            terminate(a.pod_id); print(f"[attach] pod {a.pod_id} terminated (results verified first)", flush=True)
        elif ok:
            print("[attach] results verified locally: the pod can be terminated now", flush=True)
        else:
            print("[attach] results NOT verified: do not terminate the pod yet", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
