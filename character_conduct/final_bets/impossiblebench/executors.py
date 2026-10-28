"""CC-F-IB executors: run one assembled submission and return (returncode, stdout, stderr, timed_out).

  local   a subprocess in a fresh temp dir: `python -I -X dev f.py` (on a pod; the code is the model's, so don't use
          it on a laptop)
  modal   a pool of Modal sandboxes (Debian slim, Python 3.11, no network, 1 CPU, 2 GiB), one per worker thread,
          a fresh directory per run; a sandbox that errors is replaced
"""
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid


class Local:
    def __init__(self, python=None):
        self.python = python or sys.executable
        self.sandbox_s = 0.0

    def __call__(self, code, timeout):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.py")
            with open(p, "w") as fh:
                fh.write(code)
            try:
                r = subprocess.run([self.python, "-I", "-X", "dev", p], cwd=d, capture_output=True, text=True,
                                   timeout=timeout)
                return r.returncode, r.stdout, r.stderr, False
            except subprocess.TimeoutExpired as e:
                return -1, (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), \
                    "", True

    def close(self):
        pass


class Modal:
    APP = "cc-f-ib"

    def __init__(self):
        import modal
        self.modal = modal
        self.app = modal.App.lookup(self.APP, create_if_missing=True)
        self.image = modal.Image.debian_slim(python_version="3.11")
        self.local = threading.local()
        self.all, self.lock = [], threading.Lock()
        self.t_open = {}
        self.sandbox_s = 0.0

    def _sb(self):
        sb = getattr(self.local, "sb", None)
        if sb is None:
            sb = self.modal.Sandbox.create("sleep", "infinity", app=self.app, image=self.image, block_network=True,
                                           cpu=1.0, memory=2048, timeout=3600)
            with self.lock:
                self.all.append(sb)
                self.t_open[id(sb)] = time.time()
            self.local.sb = sb
        return sb

    def _drop(self):
        sb = getattr(self.local, "sb", None)
        self.local.sb = None
        if sb is not None:
            self._close(sb)

    def _close(self, sb):
        try:
            sb.terminate()
        except Exception:                           # noqa: BLE001
            pass
        with self.lock:
            t = self.t_open.pop(id(sb), None)
            if t:
                self.sandbox_s += time.time() - t

    def __call__(self, code, timeout):
        for k in range(3):
            try:
                sb = self._sb()
                d = f"/w/{uuid.uuid4().hex}"
                w = sb.exec("sh", "-c", f"mkdir -p {d} && cat > {d}/f.py")
                w.stdin.write(code.encode())
                w.stdin.write_eof()
                w.stdin.drain()
                w.wait()
                p = sb.exec("timeout", "-s", "KILL", str(timeout), "python", "-X", "dev", "f.py", workdir=d,
                            timeout=timeout + 30)
                out, err = p.stdout.read(), p.stderr.read()
                p.wait()
                rc = p.returncode
                sb.exec("rm", "-rf", d).wait()
                return rc, out, err, rc in (124, 137, -9)
            except Exception as e:                  # noqa: BLE001
                self._drop()
                if k == 2:
                    raise
                print(f"[ib-exec] sandbox error, replacing: {type(e).__name__}: {str(e)[:120]}", flush=True)
                time.sleep(2 * (k + 1))

    def close(self):
        with self.lock:
            sbs = list(self.all)
        for sb in sbs:
            self._close(sb)


def make(kind):
    return {"local": Local, "modal": Modal}[kind]()
