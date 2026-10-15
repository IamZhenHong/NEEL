"""Pull s7 LoRA adapters from OUR running pod while the job runs (parallel scp streams), verify each against the
sha256 the pod wrote, and mark it pulled on the pod (pod.sh waits for the marks before it exits). One ssh stream from
this laptop to a RunPod host moves ~0.6 MiB/s (persona conflict_v0: 715 MiB in ~20 min), so the launcher's end-of-job
pull of ~2.4 GiB would cost ~70 pod-minutes; pulling during the run overlaps it with compute.

  caffeinate -i uv run python selfknow/s7_selfmodel_lag/pull_adapters.py --log selfknow/s7_selfmodel_lag/launch_main.log \
      --mode main --expect 20
The ssh target is read from the launcher log ("[pod] ssh root@HOST -p PORT"); nothing else on the account is touched."""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

S7 = Path(__file__).resolve().parent
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", "-i", os.path.expanduser("~/.ssh/id_ed25519")]


def target(log):
    m = re.findall(r"\[pod\] ssh root@([\d.]+) -p (\d+)", Path(log).read_text(errors="replace")) if Path(log).exists() else []
    return (m[-1][0], int(m[-1][1])) if m else None


def ssh(host, port, cmd, timeout=90):
    return subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd], capture_output=True, text=True,
                          timeout=timeout)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pull_one(host, port, remote, local, name, want):
    dst = local / name
    tmp = local / f".{name}.part"
    if dst.is_dir() and (dst / "adapter_model.safetensors").exists() and sha(dst / "adapter_model.safetensors") == want:
        ok = True
    else:
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        t = time.time()
        for attempt in range(4):
            r = subprocess.run(["scp", *SSH_OPTS, "-o", "Compression=no", "-P", str(port),
                                f"root@{host}:{remote}/{name}/adapter_model.safetensors",
                                f"root@{host}:{remote}/{name}/adapter_config.json", str(tmp)],
                               capture_output=True, text=True)
            if r.returncode == 0 and sha(tmp / "adapter_model.safetensors") == want:
                break
            time.sleep(10 * (attempt + 1))
        else:
            return name, False, f"scp failed: {r.stderr[-200:]}"
        shutil.rmtree(dst, ignore_errors=True); tmp.rename(dst)
        mb = (dst / "adapter_model.safetensors").stat().st_size / 2**20
        print(f"[pull] {name}: {mb:.0f} MiB in {time.time() - t:.0f}s ({mb / max(time.time() - t, 1e-9):.2f} MiB/s)", flush=True)
        ok = True
    r = ssh(host, port, f"touch {remote}/{name}.pulled")
    return name, ok and r.returncode == 0, "" if r.returncode == 0 else "could not mark pulled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--mode", required=True, choices=["pilot", "main"])
    ap.add_argument("--expect", type=int, required=True, help="number of adapters the job will write")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--poll", type=int, default=45)
    ap.add_argument("--give-up-min", type=float, default=240)
    args = ap.parse_args()
    remote = f"/root/NEEL/selfknow/s7_selfmodel_lag/adapters_pod/{args.mode}"
    local = S7 / "adapters" / args.mode
    local.mkdir(parents=True, exist_ok=True)
    done, pending, t0, fails = set(), {}, time.time(), 0
    ex = cf.ThreadPoolExecutor(args.streams)
    while time.time() - t0 < args.give_up_min * 60:
        tg = target(args.log)
        if not tg:
            time.sleep(args.poll); continue
        host, port = tg
        try:
            r = ssh(host, port, f"cd {remote} 2>/dev/null && for f in *.done; do [ -e \"$f\" ] && echo \"$f $(cat $f)\"; done; "
                                "[ -f /root/job.exit ] && echo __EXIT__ || true")
        except subprocess.TimeoutExpired:
            r = None
        if r is None or r.returncode != 0:
            fails += 1
            print(f"[pull] ssh poll failed ({fails})", flush=True)
            if fails > 40:
                break
            time.sleep(args.poll); continue
        fails = 0
        listed = {}
        for line in r.stdout.splitlines():
            name, _, js = line.partition(" ")
            if not name.endswith(".done") or not js:
                continue
            try:
                listed[name[:-len(".done")]] = json.loads(js)["sha256"]
            except Exception:  # noqa: BLE001 - half-written marker; next poll
                pass
        for name, want in listed.items():
            if name not in done and name not in pending:
                pending[name] = ex.submit(pull_one, host, port, remote, local, name, want)
        for name, fut in list(pending.items()):
            if fut.done():
                n, ok, msg = fut.result()
                del pending[name]
                if ok:
                    done.add(n)
                else:
                    print(f"[pull] {n} FAILED: {msg}; will retry", flush=True)
        total = sum((local / n / "adapter_model.safetensors").stat().st_size for n in done) / 2**30
        print(f"[pull] {len(done)}/{args.expect} adapters pulled and verified ({total:.2f} GiB); pending {len(pending)}", flush=True)
        if len(done) >= args.expect:
            break
        time.sleep(args.poll)
    ex.shutdown(wait=True)
    (local / "PULLED.json").write_text(json.dumps(sorted(done), indent=1))
    print(f"[pull] finished: {len(done)}/{args.expect}", flush=True)


if __name__ == "__main__":
    main()
