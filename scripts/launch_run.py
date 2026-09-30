"""Launch a guarded RunPod pod, sync the repo, run a scoring job, pull results, terminate.

Usage (from repo root):
  uv run python scripts/launch_run.py --max-minutes 40 --max-cost 0.60 \
      -- python scripts/run_scoring.py --items items/onboarding_001.yaml --out results/pilot_001.jsonl

Everything after `--` runs inside the pod at /root/NEEL with `uv run`.
Results directory is rsynced back to ./results/ before the pod is terminated.
"""
from __future__ import annotations

import argparse, os, shlex, subprocess, sys, time
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ.setdefault("COPYFILE_DISABLE", "1")  # no macOS ._ files in tarballs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from runpod_guard import PodSession, account  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-i", os.path.expanduser("~/.ssh/id_ed25519")]

BOOTSTRAP = r"""
set -e
cd /root/NEEL
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; fi
export PATH="$HOME/.local/bin:$PATH" UV_PROJECT_ENVIRONMENT=/root/venv
uv sync -q
/root/venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('[pod] torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
"""


def ssh(host, port, cmd, check=True, capture=False, timeout=None):
    full = ["ssh", *SSH_OPTS, "-o", "ServerAliveInterval=30", "-p", str(port), f"root@{host}", cmd]
    return subprocess.run(full, check=check, text=True, capture_output=capture, timeout=timeout)


def wait_ssh(host, port, timeout=600):
    t = time.time()
    while time.time() - t < timeout:
        if ssh(host, port, "echo ok", check=False, capture=True, timeout=30).returncode == 0:
            return
        time.sleep(10)
    raise TimeoutError("ssh never came up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=int, required=True)
    ap.add_argument("--max-cost", type=float, required=True)
    ap.add_argument("--gpus", nargs="+", default=None,
                    help="GPU types in preference order (default: 4090, A6000, 3090, L40S)")
    ap.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    ap.add_argument("--rounds", type=int, default=5, help="retry rounds when no stock")
    ap.add_argument("--round-wait", type=int, default=30, help="seconds between rounds")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("give the pod command after --")

    a = account()
    print(f"[pre] balance ${a.balance:.2f}, pods {len(a.pods)}")
    env = {"HF_HOME": "/root/hf"}
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token

    # Stock comes and goes: try secure then community cloud, a few rounds, before giving up.
    clouds = [args.cloud] + [c for c in ("SECURE", "COMMUNITY") if c != args.cloud]
    session = None
    for rnd in range(args.rounds):
        for cloud in clouds:
            try:
                s_ = PodSession(max_minutes=args.max_minutes, max_cost_usd=args.max_cost,
                                gpus=args.gpus, cloud=cloud, env=env)
                s_.__enter__()
                session = s_
                break
            except RuntimeError as e:
                if "no instances currently available" in str(e) or "SUPPLY" in str(e):
                    print(f"[runpod] round {rnd+1}/{args.rounds}: no stock on {cloud}")
                    continue
                raise
        if session:
            break
        if rnd < args.rounds - 1:
            time.sleep(args.round_wait)
    if session is None:
        sys.exit("no GPU stock after all rounds; nothing launched, nothing charged")

    pod = session
    rc = 1
    try:
        host, port = pod.ssh_target()
        print(f"[pod] ssh root@{host} -p {port}")
        wait_ssh(host, port)
        r = ssh(host, port, pod.self_destruct_cmd(), capture=True, timeout=60)
        armed = r.stdout.split()
        print(f"[pod] self-destruct {'armed' if armed and armed[0]=='armed' else 'NOT ARMED'}; "
              f"pod-id visible to pod: {armed[-1:] == ['1']}; fires in {args.max_minutes} min", flush=True)
        pod.check_budget()

        # sync code with tar over ssh (no rsync needed on the pod); skip results, corpus, secrets
        excl = ["--exclude=./.git", "--exclude=./.venv", "--exclude=./results", "--exclude=./reference",
                "--exclude=./logs", "--exclude=./.env", "--exclude=__pycache__"]
        tar = subprocess.Popen(["tar", "--no-xattrs", "--no-mac-metadata", "-C", str(REPO), *excl, "-czf", "-", "."],
                               stdout=subprocess.PIPE)
        subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}",
                        "mkdir -p /root/NEEL && tar --no-same-owner --no-same-permissions --warning=no-unknown-keyword -xzf - -C /root/NEEL"], stdin=tar.stdout, check=True)
        tar.wait()
        print("[pod] code synced", flush=True)
        ssh(host, port, BOOTSTRAP)
        pod.check_budget()

        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        rev = rev + ("-dirty" if dirty else "")
        job = f"cd /root/NEEL && export GIT_REV={rev} && export PATH=$HOME/.local/bin:$PATH HF_HOME=/root/hf PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/root/venv && uv run " + " ".join(shlex.quote(c) for c in cmd)
        print(f"[pod] running: {job}")
        rc = ssh(host, port, job, check=False).returncode
        print(f"[pod] job exit code {rc}; elapsed cost ~${pod.elapsed_cost():.3f}")

        (REPO / "results").mkdir(exist_ok=True)
        pull = subprocess.Popen(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}",
                                 "cd /root/NEEL && tar -czf - results 2>/dev/null || true"], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-xzf", "-", "-C", str(REPO)], stdin=pull.stdout, check=False)
        pull.wait()
        print("[local] results synced to ./results/")
    finally:
        session.__exit__(None, None, None)

    a = account()
    print(f"[post] balance ${a.balance:.2f}, pods {len(a.pods)}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
