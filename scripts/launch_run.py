"""Launch a guarded RunPod pod, sync the repo, run a scoring job, pull results, terminate.

Usage (from repo root):
  uv run python scripts/launch_run.py --max-minutes 40 --max-cost 0.60 \
      -- python scripts/run_scoring.py --items items/onboarding_001.yaml --out results/pilot_001.jsonl

Everything after `--` runs inside the pod at /root/NEEL with `uv run`.
Results directory is rsynced back to ./results/ before the pod is terminated.
"""
from __future__ import annotations

import argparse, json, os, shlex, subprocess, sys, time
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
    return subprocess.run(full, check=check, text=True, errors="replace", capture_output=capture, timeout=timeout)


def wait_ssh(host, port, timeout=600):
    t = time.time()
    while time.time() - t < timeout:
        try:
            if ssh(host, port, "echo ok", check=False, capture=True, timeout=30).returncode == 0:
                return
        except subprocess.TimeoutExpired:
            pass   # sshd not answering yet; keep waiting
        time.sleep(10)
    raise TimeoutError("ssh never came up")


UPLOAD_SERVER = r"""
import http.server, os, sys, json
ROOT = sys.argv[2]; os.makedirs(ROOT, exist_ok=True)
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/health":
            body = b"ok"
        else:
            body = json.dumps({f: os.path.getsize(os.path.join(ROOT, f)) for f in os.listdir(ROOT)}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_PUT(self):
        name = os.path.basename(self.path)
        n = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(n)
        with open(os.path.join(ROOT, name), "wb") as f:
            f.write(data)
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
http.server.ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1])), H).serve_forever()
"""


def start_upload_server(host, port, pod_id):
    """Start a tiny PUT server on the pod and return its RunPod-proxy URL once it answers, else None."""
    ssh(host, port, "cat > /root/upsrv.py <<'PYEOF'\n" + UPLOAD_SERVER + "\nPYEOF\n"
        "mkdir -p /root/chunks_in && (setsid nohup python3 /root/upsrv.py 8080 /root/chunks_in > /root/upsrv.log 2>&1 < /dev/null &)", timeout=60)
    url = f"https://{pod_id}-8080.proxy.runpod.net"
    for _ in range(48):   # the proxy route can take a few minutes to come up; use curl (the proxy 403s python-urllib's UA)
        r = subprocess.run(["curl", "-sS", "-m", "10", url + "/health"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip() == "ok":
            return url
        time.sleep(5)
    print("[pod] upload server not reachable through the proxy; falling back to scp", flush=True)
    return None


def sync_file_parallel(host, port, rel, streams=12, chunk_mb=8, http_url=None):
    """Copy one repo-relative file to the pod as parallel chunks over independent ssh streams, then
    reassemble and verify sha256. A single ssh/scp stream from here to a RunPod host has been seen
    at ~15 KB/s (long high-latency route); N streams get roughly N x that."""
    import hashlib, tempfile, concurrent.futures as cf
    src = REPO / rel
    data = src.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    remote_dir = f"/root/NEEL/{rel}".rsplit("/", 1)[0]
    tmp = Path(tempfile.mkdtemp(prefix="chunks_", dir=str(REPO / "logs")))
    # chunks are staged lazily, one file per in-flight upload, so at most `streams` x chunk_mb sits on disk at a time
    # (2026-10-06: staging the whole file filled a nearly full laptop disk)
    parts = []   # (path, offset, size)
    for i in range(0, max(1, len(data)), chunk_mb << 20):
        parts.append((tmp / f"p{len(parts):04d}", i, min(chunk_mb << 20, len(data) - i)))
    ssh(host, port, f"mkdir -p {shlex.quote(remote_dir)} /root/chunks_in && rm -f /root/chunks_in/p*", timeout=60)
    t0 = time.time()

    def put_scp(pth):
        for attempt in range(5):
            r = subprocess.run(["scp", *SSH_OPTS, "-o", "Compression=no", "-P", str(port), str(pth), f"root@{host}:/root/chunks_in/{pth.name}"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"chunk {pth.name} failed: {r.stderr[-200:]}")

    def put_http(pth):
        for attempt in range(6):
            r = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "300", "-T", str(pth), f"{http_url}/{pth.name}"],
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip() == "200":
                return
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"chunk {pth.name} http upload failed: rc={r.returncode} {r.stdout} {r.stderr[-200:]}")

    def put(part):
        pth, off, size = part
        pth.write_bytes(data[off:off + size])
        try:
            (put_http if http_url else put_scp)(pth)
        finally:
            pth.unlink(missing_ok=True)
    # resumable: skip chunks the pod already holds at the right size
    have = {}
    if http_url:
        r = subprocess.run(["curl", "-sS", "-m", "20", http_url + "/list"], capture_output=True, text=True)
        try:
            have = json.loads(r.stdout) if r.returncode == 0 else {}
        except Exception:  # noqa: BLE001
            have = {}
    todo = [part for part in parts if have.get(part[0].name) != part[2]]
    with cf.ThreadPoolExecutor(max_workers=streams) as ex:
        list(ex.map(put, todo))
    el = time.time() - t0
    r = ssh(host, port, f"cat /root/chunks_in/p* > {shlex.quote('/root/NEEL/' + rel)} && rm -f /root/chunks_in/p* && sha256sum {shlex.quote('/root/NEEL/' + rel)} | cut -c1-64",
            capture=True, timeout=300)
    for pth, _, _ in parts:
        pth.unlink(missing_ok=True)
    tmp.rmdir()
    ok = r.stdout.strip() == sha
    print(f"[pod] extra path synced: {rel} ({len(data)/2**20:.0f} MiB in {el:.0f}s, {len(data)/2**20/max(el,1e-9):.2f} MiB/s, "
          f"{len(todo)}/{len(parts)} chunks x {streams} streams via {'http-proxy' if http_url else 'scp'}, sha256 {'OK' if ok else 'MISMATCH'})", flush=True)
    if not ok:
        raise RuntimeError(f"sha256 mismatch after parallel sync of {rel}")


def _parse_poll(raw: bytes, offset: int):
    """Parse one poll's raw bytes: b'__SIZE__<n>\\n' + log bytes from offset + optional
    b'\\n__JOB_DONE__<rc>\\n'. Returns (size, chunk_bytes, rc_or_None)."""
    size, rc = None, None
    if raw.startswith(b"__SIZE__"):
        head, _, raw = raw.partition(b"\n")
        try:
            size = int(head[len(b"__SIZE__"):])
        except ValueError:
            size = None
    m = raw.rfind(b"__JOB_DONE__")
    if m != -1:
        tail = raw[m + len(b"__JOB_DONE__"):].split()
        if tail:
            try:
                rc = int(tail[0])
            except ValueError:
                rc = None
        raw = raw[:m].rstrip(b"\n") if m > 0 else b""
        # the wrapper prints "\n__JOB_DONE__" so strip exactly one added newline, not real log newlines
    return size, raw, rc


def run_detached(host, port, job, pod, poll_s=20, stall_min=30, max_misses=60):
    """Start the job in its own session on the pod and poll its log. A dropped ssh connection
    only interrupts polling, never the job. Completion is detected from a marker FILE
    (/root/job.exit), not only from the log stream; offsets are counted in raw bytes so
    decoding never desynchronises the poller (2026-10-03: a decoded-string offset left the
    poller blind after one bad read and a finished job was aborted as 'silent').
    Aborts if the log is silent for stall_min minutes AND no exit marker exists.
    Returns the job's exit code."""
    inner = shlex.quote(f"rm -f /root/job.exit; echo [job] start $(date -u +%H:%M:%S); {job}; rc=$?; "
                        f"echo __JOB_EXIT__$rc; echo $rc > /root/job.exit")
    ssh(host, port, f"rm -f /root/job.log /root/job.exit; (setsid nohup bash -c {inner} > /root/job.log 2>&1 < /dev/null &)",
        timeout=60)
    offset, misses, last_output = 0, 0, time.time()
    while True:
        time.sleep(poll_s)
        pod.check_budget()
        poll_cmd = (f"s=$(stat -c %s /root/job.log 2>/dev/null || echo 0); printf '__SIZE__%s\\n' \"$s\"; "
                    f"tail -c +{offset + 1} /root/job.log 2>/dev/null; "
                    f"if [ -f /root/job.exit ]; then printf '\\n__JOB_DONE__%s\\n' \"$(cat /root/job.exit)\"; fi")
        try:
            r = subprocess.run(["ssh", *SSH_OPTS, "-o", "ServerAliveInterval=30", "-p", str(port), f"root@{host}", poll_cmd],
                               capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            r = None
        if r is None or r.returncode != 0:
            misses += 1
            print(f"[pod] poll failed ({misses}/{max_misses}); job keeps running on the pod", flush=True)
            if misses % 10 == 0:   # the pod may have been re-addressed; re-resolve ip/port
                try:
                    host, port = pod.ssh_target(timeout_s=60)
                    print(f"[pod] re-resolved ssh target {host}:{port}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[pod] could not re-resolve ssh target: {e}", flush=True)
            if misses >= max_misses:   # ~20 min with poll_s=20; the pod-side self-destruct still bounds spend
                raise RuntimeError("lost contact with pod for too long")
            continue
        misses = 0
        size, chunk, rc = _parse_poll(r.stdout, offset)
        if size is not None and size < offset:     # log truncated or restarted: re-read from the start
            print(f"[pod] job.log shrank ({offset} -> {size} bytes); resetting offset", flush=True)
            offset = 0
            continue
        if chunk:
            last_output = time.time()
            offset += len(chunk)
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()
        if rc is not None:
            return rc
        if time.time() - last_output > stall_min * 60:
            raise RuntimeError(f"job log silent for {stall_min} min and no exit marker; aborting to save credit")


def pull_results(host, port, rdir):
    """Copy the results dir back. Returns True only if the ssh side completed (255 = no connection)."""
    (REPO / rdir).mkdir(parents=True, exist_ok=True)
    pull = subprocess.Popen(["ssh", *SSH_OPTS, "-o", "ConnectTimeout=30", "-p", str(port), f"root@{host}",
                             f"cd /root/NEEL && tar -czf - {shlex.quote(rdir)} 2>/dev/null || true"], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-xzf", "-", "-C", str(REPO)], stdin=pull.stdout, check=False)
    pull.wait()
    ok = pull.returncode == 0
    print(f"[local] results {'synced' if ok else 'pull FAILED (ssh rc %d)' % pull.returncode} ./{rdir}/", flush=True)
    return ok


def pull_results_persistent(host, port, rdir, pod, grace_min):
    """Retry the pull until it succeeds or the pod's self-destruct grace has elapsed (network outages)."""
    deadline = pod.t0 + (pod.max_minutes + grace_min) * 60 - 120
    while True:
        try:
            if pull_results(host, port, rdir):
                return True
        except Exception as e:  # noqa: BLE001
            print(f"[local] pull error: {e}", flush=True)
        if time.time() > deadline:
            print("[local] giving up on the results pull: pod self-destruct grace exhausted", flush=True)
            return False
        try:
            host, port = pod.ssh_target(timeout_s=60)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=int, required=True)
    ap.add_argument("--max-cost", type=float, required=True)
    ap.add_argument("--gpus", nargs="+", default=None,
                    help="GPU types in preference order (default: 4090, A6000, 3090, L40S)")
    ap.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    ap.add_argument("--rounds", type=int, default=5, help="retry rounds when no stock")
    ap.add_argument("--round-wait", type=int, default=30, help="seconds between rounds")
    ap.add_argument("--container-gb", type=int, default=50)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--exclude", action="append", default=[],
                    help="extra repo-relative paths to leave out of the code sync (repeatable)")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="launch even if another pod is running (deliberate parallel jobs only)")
    ap.add_argument("--name", default="provocation-pilot", help="pod name")
    ap.add_argument("--sync-extra", action="append", default=[],
                    help="repo-relative path to copy to the pod after the main sync, bypassing the exclude list "
                         "(bsdtar matches --exclude against path tails, so ./results also drops story_imprinting/results)")
    ap.add_argument("--sync-streams", type=int, default=12, help="parallel scp streams for --sync-extra files")
    ap.add_argument("--results-dir", default="results",
                    help="repo-relative directory pulled back from the pod when the job ends")
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
                                gpus=args.gpus, cloud=cloud, env=env, container_gb=args.container_gb,
                                name=args.name, allow_concurrent=args.allow_concurrent)
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
              f"pod-id visible to pod: {armed[-1:] == ['1']}; fires in {args.max_minutes + pod.grace_minutes} min (job limit {args.max_minutes} + {pod.grace_minutes} grace)", flush=True)
        pod.check_budget()

        # sync code with tar over ssh (no rsync needed on the pod); skip results, corpus, secrets
        excl = ["--exclude=./.git", "--exclude=./.venv", "--exclude=./results", "--exclude=./reference",
                "--exclude=./logs", "--exclude=./.env", "--exclude=__pycache__"]
        excl += [f"--exclude={e}" if "*" in e else f"--exclude=./{e.strip('/')}" for e in args.exclude]
        tar = subprocess.Popen(["tar", "--no-xattrs", "--no-mac-metadata", "-C", str(REPO), *excl, "-czf", "-", "."],
                               stdout=subprocess.PIPE)
        subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}",
                        "mkdir -p /root/NEEL && tar --no-same-owner --no-same-permissions --warning=no-unknown-keyword -xzf - -C /root/NEEL"], stdin=tar.stdout, check=True)
        tar.wait()
        http_url = start_upload_server(host, port, pod.pod_id) if args.sync_extra else None
        for extra in args.sync_extra:
            sync_file_parallel(host, port, extra.strip("/"), streams=args.sync_streams, http_url=http_url)
        print("[pod] code synced", flush=True)
        ssh(host, port, BOOTSTRAP)
        pod.check_budget()

        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        rev = rev + ("-dirty" if dirty else "")
        job = f"cd /root/NEEL && export GIT_REV={rev} && export PATH=$HOME/.local/bin:$PATH HF_HOME=/root/hf PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/root/venv && uv run " + " ".join(shlex.quote(c) for c in cmd)
        print(f"[pod] running (detached): {job}")
        rdir = args.results_dir.strip("/")
        try:
            rc = run_detached(host, port, job, pod)
        except Exception:
            # keep the evidence before the pod is terminated, and pull whatever results exist
            try:
                r = ssh(host, port, "tail -c 4000 /root/job.log; echo; cat /root/job.exit 2>/dev/null; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv; df -h /root | tail -1; du -sh /root/hf 2>/dev/null",
                        check=False, capture=True, timeout=60)
                print("[pod] last job log before abort:\n" + (r.stdout or "") + (r.stderr or ""), flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[pod] could not fetch job log: {e}", flush=True)
            pull_results_persistent(host, port, rdir, pod, pod.grace_minutes)
            raise
        print(f"[pod] job exit code {rc}; elapsed cost ~${pod.elapsed_cost():.3f}")
        pull_results_persistent(host, port, rdir, pod, pod.grace_minutes)
    finally:
        session.__exit__(None, None, None)

    a = account()
    print(f"[post] balance ${a.balance:.2f}, pods {len(a.pods)}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
