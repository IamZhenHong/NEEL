"""Guarded RunPod launch without ssh (2026-10-18 cloud session: outbound TCP 22 is blocked here, so
infra/launch_run.py cannot reach the pod). Reuses infra/runpod_guard.PodSession for the spend guards (no other pod
of our name, GPU allow-list and price cap, max_minutes x rate <= max_cost, terminate on exit).
No remote command execution: the job script and code files are fixed at pod creation (a base64 tar.gz in the pod
env, unpacked by the start command into /root/code), and the job starts by itself once the named data inputs have
arrived. The pod's HTTP endpoint (port 8080, through the RunPod HTTPS proxy, token-protected) only accepts those
named data files into /root/in and serves the job log, status and /root/out. The start command also arms a
self-destruct after max_minutes + grace (runpodctl remove, else stop the container).
Usage: python p5_http_launch.py --max-minutes 48 --max-cost 1.50 --out DIR --job JOB.sh --code F.py ... --data F ...
"""
import argparse
import base64
import io
import json
import os
import secrets
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "infra"))
from runpod_guard import PodSession, account  # noqa: E402

AGENT = r'''
import http.server, json, os, signal
signal.signal(signal.SIGTERM, lambda *a: os._exit(0))
TOK = os.environ["AGENT_TOKEN"]; IN, OUT = "/root/in", "/root/out"
NAMES = set(os.environ["DATA_NAMES"].split(","))
os.makedirs(IN, exist_ok=True); os.makedirs(OUT, exist_ok=True)
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body):
        body = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def authed(self):
        if self.headers.get("X-Token") != TOK:
            self.send(403, "no"); return False
        return True
    def do_GET(self):
        if self.path == "/health": return self.send(200, "ok")
        if not self.authed(): return
        if self.path.startswith("/log/"):
            off = int(self.path[5:])
            try:
                with open("/root/job.log", "rb") as f: f.seek(off); return self.send(200, f.read(1 << 20))
            except FileNotFoundError: return self.send(200, b"")
        if self.path == "/status":
            ex = open("/root/job.exit").read().strip() if os.path.exists("/root/job.exit") else None
            return self.send(200, json.dumps({"started": os.path.exists("/root/job.log"), "exit": ex}))
        if self.path == "/ls":
            return self.send(200, json.dumps({f: os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)}))
        if self.path.startswith("/out/"):
            p = os.path.join(OUT, os.path.basename(self.path[5:]))
            return self.send(200, open(p, "rb").read()) if os.path.isfile(p) else self.send(404, "none")
        self.send(404, "none")
    def do_PUT(self):
        if not self.authed(): return
        name = os.path.basename(self.path)
        if name not in NAMES: return self.send(403, "not an expected input")
        data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        open(os.path.join(IN, name + ".part"), "wb").write(data); os.replace(os.path.join(IN, name + ".part"), os.path.join(IN, name))
        if all(os.path.exists(os.path.join(IN, n)) for n in NAMES): open(os.path.join(IN, ".ready"), "w").close()
        self.send(200, "ok")
http.server.ThreadingHTTPServer(("0.0.0.0", 8080), H).serve_forever()
'''


def call(url, tok, method="GET", data=None, timeout=60):
    req = urllib.request.Request(url, method=method, data=data, headers={"X-Token": tok, "User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=int, required=True)
    ap.add_argument("--max-cost", type=float, required=True)
    ap.add_argument("--grace-minutes", type=int, default=5)
    ap.add_argument("--gpus", nargs="+", default=["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"])
    ap.add_argument("--container-gb", type=int, default=120)
    ap.add_argument("--name", default="provocation-p5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--job", required=True, help="job script, fixed at pod creation")
    ap.add_argument("--code", nargs="*", default=[], help="code files for /root/code, fixed at pod creation")
    ap.add_argument("--data", nargs="+", required=True, help="data inputs, uploaded once the pod is up")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = secrets.token_hex(16)
    sd = (a.max_minutes + a.grace_minutes) * 60
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in [a.job, *a.code]:
            tf.add(f, arcname=os.path.basename(f))
    job = os.path.basename(a.job)
    start = ("mkdir -p /root/code /root/in /root/out; echo $AGENT_B64 | base64 -d > /root/agent.py; "
             "echo $CODE_B64 | base64 -d | tar -xzf - -C /root/code; "
             f"(sleep {sd}; runpodctl remove pod $RUNPOD_POD_ID || kill 1) > /root/self_destruct.log 2>&1 & "
             "(until [ -f /root/in/.ready ]; do sleep 5; done; "
             f"bash /root/code/{job} > /root/job.log 2>&1; echo $? > /root/job.exit) & "
             "exec python3 /root/agent.py")
    env = {"HF_TOKEN": os.environ["HF_TOKEN"], "AGENT_TOKEN": tok,
           "DATA_NAMES": ",".join(os.path.basename(f) for f in a.data),
           "AGENT_B64": base64.b64encode(AGENT.encode()).decode(),
           "CODE_B64": base64.b64encode(buf.getvalue()).decode()}
    print(f"[pre] code bundle {len(env['CODE_B64'])} b64 chars; data {[os.path.basename(f) for f in a.data]}", flush=True)
    print(f"[pre] balance ${account().balance:.2f}", flush=True)
    session, rc = None, 1
    for rnd in range(5):
        for cloud in ("SECURE", "COMMUNITY"):
            try:
                s = PodSession(max_minutes=a.max_minutes, max_cost_usd=a.max_cost, gpus=a.gpus, cloud=cloud, env=env,
                               container_gb=a.container_gb, volume_gb=0, name=a.name, grace_minutes=a.grace_minutes)
                s.body["ports"] = ["8080/http"]
                s.body["dockerStartCmd"] = ["bash", "-c", start]
                s.body.pop("volumeMountPath", None)
                session = s.__enter__()
                break
            except RuntimeError as e:
                if "no instances currently available" in str(e) or "SUPPLY" in str(e):
                    print(f"[runpod] round {rnd + 1}: no stock on {cloud}", flush=True)
                    continue
                raise
        if session:
            break
        time.sleep(30)
    if session is None:
        sys.exit("no GPU stock; nothing launched")
    base = f"https://{session.pod_id}-8080.proxy.runpod.net"
    try:
        t = time.time()
        while True:
            try:
                if call(base + "/health", tok, timeout=20) == b"ok":
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                pass
            session.check_budget()
            if time.time() - t > 900:
                raise TimeoutError("agent never answered")
            time.sleep(10)
        print(f"[pod] agent up after {time.time() - t:.0f}s; self-destruct in {sd // 60} min", flush=True)
        for f in a.data:
            call(f"{base}/in/{os.path.basename(f)}", tok, "PUT", open(f, "rb").read(), timeout=300)
        print(f"[pod] uploaded {len(a.data)} data files; the job starts on its own", flush=True)
        off, last_pull, logf = 0, 0.0, open(os.path.join(a.out, "job.log"), "ab")
        while True:
            time.sleep(20)
            try:
                chunk = call(f"{base}/log/{off}", tok)
                if chunk:
                    off += len(chunk)
                    logf.write(chunk); logf.flush()
                    sys.stdout.write(chunk.decode(errors="replace")); sys.stdout.flush()
                st = json.loads(call(base + "/status", tok))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                print(f"[pod] poll error: {e}", flush=True)
                st = {"exit": None}
            if time.time() - last_pull > 300 or st["exit"] is not None:
                pull(base, tok, a.out)
                last_pull = time.time()
            if st["exit"] is not None:
                rc = int(st["exit"])
                print(f"[pod] job exit {rc}; cost ~${session.elapsed_cost():.3f}", flush=True)
                break
            session.check_budget()
    finally:
        try:
            pull(base, tok, a.out)
        except Exception as e:  # noqa: BLE001
            print(f"[pod] final pull failed: {e}", flush=True)
        session.__exit__(None, None, None)
    print(f"[post] balance ${account().balance:.2f}, pods {len(account().pods)}", flush=True)
    sys.exit(rc)


def pull(base, tok, out):
    for f in json.loads(call(base + "/ls", tok)):
        open(os.path.join(out, f), "wb").write(call(f"{base}/out/{f}", tok, timeout=300))


if __name__ == "__main__":
    main()
