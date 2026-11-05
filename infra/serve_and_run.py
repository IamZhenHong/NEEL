"""Guarded pod + server + LOCAL client: launch a pod (same guards as launch_run.py: caps, self-destruct, stock rounds), sync the
shipped paths, bootstrap, start a server job detached on the pod (bound to the pod's localhost), tunnel its port to this
laptop, wait for its /health, run a LOCAL command against it, save the server log, terminate. API keys and the API ledger
stay on the laptop; the pod only serves the model.
  uv run python infra/serve_and_run.py --max-minutes 60 --max-cost 0.9 --gpus "NVIDIA GeForce RTX 4090" \
     --ship retention/longconv_server.py --ship retention/run_adaptive.py --ship retention/cd ... \
     --server "python retention/longconv_server.py --port 8765" --log-dir retention/results/retention_longconv_pilot \
     -- <local command>     (the local command gets RILEY_URL=http://127.0.0.1:<local-port>)"""
from __future__ import annotations
import argparse, os, shlex, shutil, subprocess, sys, threading, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_run import BOOTSTRAP, REPO, SSH_OPTS, ssh, wait_ssh  # noqa: E402
from runpod_guard import PodSession, account  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=int, required=True); ap.add_argument("--max-cost", type=float, required=True)
    ap.add_argument("--gpus", nargs="+", default=None); ap.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--round-wait", type=int, default=30)
    ap.add_argument("--container-gb", type=int, default=50); ap.add_argument("--name", default="longconv")
    ap.add_argument("--ship", action="append", default=[], required=True)
    ap.add_argument("--server", required=True, help="command run with `uv run` in /root/NEEL on the pod")
    ap.add_argument("--remote-port", type=int, default=8765); ap.add_argument("--local-port", type=int, default=18765)
    ap.add_argument("--load-minutes", type=float, default=20.0, help="give up if /health is not up after this long")
    ap.add_argument("--bootstrap-minutes", type=float, default=15.0); ap.add_argument("--ssh-wait-minutes", type=float, default=6.0)
    ap.add_argument("--log-dir", required=True, help="where server.log is saved (relative to the repo)")
    ap.add_argument("--min-free-gb", type=float, default=1.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("give the local command after --")
    free_gb = shutil.disk_usage(REPO).free / 2**30
    if free_gb < args.min_free_gb:
        sys.exit(f"[pre] REFUSING: {free_gb:.1f} GiB free (< {args.min_free_gb})")
    missing = [p for p in args.ship if not (REPO / p.strip("/")).exists()]
    if missing:
        sys.exit(f"[pre] REFUSING: --ship paths not found: {missing}")
    a = account(); print(f"[pre] disk {free_gb:.1f} GiB free; balance ${a.balance:.2f}, pods {len(a.pods)}", flush=True)

    session = None
    clouds = [args.cloud] + [c for c in ("SECURE", "COMMUNITY") if c != args.cloud]
    for rnd in range(args.rounds):
        for cloud in clouds:
            try:
                s_ = PodSession(max_minutes=args.max_minutes, max_cost_usd=args.max_cost, gpus=args.gpus, cloud=cloud,
                                env={"HF_HOME": "/root/hf"}, container_gb=args.container_gb, name=args.name)
                s_.__enter__(); session = s_; break
            except RuntimeError as e:
                if "no instances currently available" in str(e) or "SUPPLY" in str(e):
                    print(f"[runpod] round {rnd + 1}/{args.rounds}: no stock on {cloud}", flush=True); continue
                raise
        if session:
            break
        if rnd < args.rounds - 1:
            time.sleep(args.round_wait)
    if session is None:
        sys.exit("no GPU stock after all rounds; nothing launched, nothing charged")

    pod = session; rc = 1; tun = None; stop = threading.Event(); host = port = None
    log_dir = REPO / args.log_dir; log_dir.mkdir(parents=True, exist_ok=True)
    try:
        host, port = pod.ssh_target(timeout_s=int(args.ssh_wait_minutes * 60))
        print(f"[pod] ssh root@{host} -p {port}", flush=True)
        wait_ssh(host, port)
        r = ssh(host, port, pod.self_destruct_cmd(), capture=True, timeout=60)
        print(f"[pod] self-destruct {'armed' if r.stdout.split()[:1] == ['armed'] else 'NOT ARMED'}; "
              f"fires in {args.max_minutes + pod.grace_minutes} min", flush=True)
        excl = ["--exclude=./.git", "--exclude=./.venv", "--exclude=./results", "--exclude=./.env", "--exclude=__pycache__"]
        paths = list(dict.fromkeys(f"./{p.strip('/')}" for p in [*args.ship, "pyproject.toml", "uv.lock"]))
        tar = subprocess.Popen(["tar", "--no-xattrs", "--no-mac-metadata", "-C", str(REPO), *excl, "-czf", "-", *paths], stdout=subprocess.PIPE)
        subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}",
                        "mkdir -p /root/NEEL && tar --no-same-owner --no-same-permissions --warning=no-unknown-keyword -xzf - -C /root/NEEL"],
                       stdin=tar.stdout, check=True)
        tar.wait(); print(f"[pod] synced {len(paths)} paths", flush=True)
        try:
            ssh(host, port, BOOTSTRAP, timeout=args.bootstrap_minutes * 60)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"[pod] bootstrap did not finish in {args.bootstrap_minutes:g} min")
        pod.check_budget()
        job = ("cd /root/NEEL && export PATH=$HOME/.local/bin:$PATH HF_HOME=/root/hf PYTHONUNBUFFERED=1 "
               f"UV_PROJECT_ENVIRONMENT=/root/venv && exec uv run {args.server}")
        ssh(host, port, f"setsid nohup bash -c {shlex.quote(job)} > /root/server.log 2>&1 < /dev/null & echo $! > /root/server.pid", timeout=60)
        print("[pod] server starting (detached)", flush=True)

        def open_tunnel():
            return subprocess.Popen(["ssh", *SSH_OPTS, "-N", "-o", "ServerAliveInterval=20", "-o", "ExitOnForwardFailure=yes",
                                     "-L", f"{args.local_port}:127.0.0.1:{args.remote_port}", "-p", str(port), f"root@{host}"])
        tun = open_tunnel()

        def keep_tunnel():   # a dropped tunnel is reopened; the client retries with backoff meanwhile
            nonlocal tun
            while not stop.wait(10):
                if tun.poll() is not None:
                    print("[tunnel] dropped; reopening", flush=True); tun = open_tunnel()
        threading.Thread(target=keep_tunnel, daemon=True).start()

        url = f"http://127.0.0.1:{args.local_port}"; t0 = time.time(); up = False
        while time.time() - t0 < args.load_minutes * 60:
            pod.check_budget()
            try:
                with urllib.request.urlopen(url + "/health", timeout=10) as h:
                    print(f"[pod] server up: {h.read().decode()} after {time.time() - t0:.0f}s", flush=True); up = True; break
            except Exception:  # noqa: BLE001
                alive = ssh(host, port, "kill -0 $(cat /root/server.pid) 2>/dev/null && echo alive || echo dead; tail -n 2 /root/server.log",
                            check=False, capture=True, timeout=60)
                print(f"[pod] waiting for server ({time.time() - t0:.0f}s): {alive.stdout.strip()[-300:]}", flush=True)
                if alive.stdout.startswith("dead"):
                    break
                time.sleep(20)
        if not up:
            raise RuntimeError("server did not come up")
        print(f"[local] running: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, env={**os.environ, "RILEY_URL": url}).returncode
        print(f"[local] exit code {rc}; pod cost so far ~${pod.elapsed_cost():.3f}", flush=True)
    finally:
        stop.set()
        try:
            if host is None:
                raise RuntimeError("no ssh target")
            r = ssh(host, port, "cat /root/server.log", check=False, capture=True, timeout=120)
            (log_dir / "server.log").write_text(r.stdout or "")
        except Exception as e:  # noqa: BLE001
            print(f"[pod] could not fetch server.log: {e}", flush=True)
        if tun is not None and tun.poll() is None:
            tun.terminate()
        session.__exit__(None, None, None)
    a = account(); print(f"[post] balance ${a.balance:.2f}, pods {len(a.pods)}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
