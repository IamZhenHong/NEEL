"""Launch a guarded RunPod pod, sync the repo, run a scoring job, pull results, terminate.

Usage (from repo root):
  uv run python scripts/launch_run.py --max-minutes 40 --max-cost 0.60 \
      -- python scripts/run_scoring.py --items items/onboarding_001.yaml --out results/pilot_001.jsonl

Everything after `--` runs inside the pod at /workspace/NEEL with `uv run`.
Results directory is rsynced back to ./results/ before the pod is terminated.
"""
from __future__ import annotations

import argparse, os, shlex, subprocess, sys, time
os.environ.setdefault("PYTHONUNBUFFERED", "1")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from runpod_guard import PodSession, account  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-i", os.path.expanduser("~/.ssh/id_ed25519")]

BOOTSTRAP = r"""
set -e
cd /workspace/NEEL
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; fi
export PATH="$HOME/.local/bin:$PATH"
uv sync -q
"""


def ssh(host, port, cmd, check=True, capture=False):
    full = ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd]
    return subprocess.run(full, check=check, text=True, capture_output=capture)


def wait_ssh(host, port, timeout=600):
    t = time.time()
    while time.time() - t < timeout:
        if ssh(host, port, "echo ok", check=False, capture=True).returncode == 0:
            return
        time.sleep(10)
    raise TimeoutError("ssh never came up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=int, required=True)
    ap.add_argument("--max-cost", type=float, required=True)
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    ap.add_argument("--fallbacks", nargs="*", default=[
        "NVIDIA GeForce RTX 4090:ALL", "NVIDIA GeForce RTX 3090:ALL", "NVIDIA RTX A6000:ALL"],
        help="gpu:cloud pairs tried in order when the first choice has no stock")
    ap.add_argument("--rounds", type=int, default=6, help="how many times to cycle the fallback list")
    ap.add_argument("--round-wait", type=int, default=45, help="seconds between rounds")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("give the pod command after --")

    a = account()
    print(f"[pre] balance ${a.balance:.2f}, pods {len(a.pods)}")
    env = {"HF_HOME": "/workspace/hf"}
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token

    attempts = [f"{args.gpu}:ALL"] + [f for f in args.fallbacks if f != f"{args.gpu}:ALL"]
    session = None
    for rnd in range(args.rounds):
        for att in attempts:
            gpu, cloud = att.rsplit(":", 1)
            try:
                session = PodSession(max_minutes=args.max_minutes, max_cost_usd=args.max_cost, gpu=gpu, env=env, cloud=cloud)
                session.__enter__()
                break
            except RuntimeError as e:
                if "SUPPLY_CONSTRAINT" in str(e) or "no longer any instances" in str(e):
                    print(f"[runpod] round {rnd+1}: no stock for {att}")
                    session = None
                    continue
                raise
        if session:
            break
        if rnd < args.rounds - 1:
            time.sleep(args.round_wait)
    if session is None:
        sys.exit("no GPU stock on any fallback; nothing launched, nothing charged")

    pod = session
    rc = 1
    try:
        host, port = pod.ssh_target()
        print(f"[pod] ssh root@{host} -p {port}")
        wait_ssh(host, port)
        ssh(host, port, pod.self_destruct_cmd())
        print(f"[pod] self-destruct armed: container exits in {args.max_minutes} min")
        pod.check_budget()

        # sync code (not results, not the 2 MB corpus, not .env)
        subprocess.run(["rsync", "-az", "--delete",
                        "--exclude", ".git", "--exclude", ".venv", "--exclude", "results",
                        "--exclude", "reference", "--exclude", ".env", "--exclude", "__pycache__",
                        "-e", f"ssh {' '.join(shlex.quote(o) for o in SSH_OPTS)} -p {port}",
                        f"{REPO}/", f"root@{host}:/workspace/NEEL/"], check=True)
        ssh(host, port, BOOTSTRAP)
        pod.check_budget()

        job = "cd /workspace/NEEL && export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 && uv run " + " ".join(shlex.quote(c) for c in cmd)
        print(f"[pod] running: {job}")
        rc = ssh(host, port, job, check=False).returncode
        print(f"[pod] job exit code {rc}; elapsed cost ~${pod.elapsed_cost():.3f}")

        (REPO / "results").mkdir(exist_ok=True)
        subprocess.run(["rsync", "-az", "-e", f"ssh {' '.join(shlex.quote(o) for o in SSH_OPTS)} -p {port}",
                        f"root@{host}:/workspace/NEEL/results/", f"{REPO}/results/"], check=False)
        print("[local] results synced to ./results/")
    finally:
        session.__exit__(None, None, None)

    a = account()
    print(f"[post] balance ${a.balance:.2f}, pods {len(a.pods)}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
