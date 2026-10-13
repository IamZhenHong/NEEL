"""Laptop-side client for the target server (state_attack/src/server.py), reached via an ssh tunnel."""
import base64
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

URL = os.environ.get("SA_URL", "http://127.0.0.1:8765")


def _post(path, obj, timeout=3600, tries=None):
    """Retries until it gets a reply: the laptop sleeps on battery and the tunnels drop; the server caches finished
    replies by request hash, so a retry after a drop returns at once instead of recomputing."""
    body = json.dumps(obj).encode()
    k = 0
    while True:
        k += 1
        try:
            req = urllib.request.Request(URL + path, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read()[:500]
            if e.code < 500 or k >= 5:
                raise RuntimeError(f"{path}: HTTP {e.code} {msg}")
            print(f"[client] {path} HTTP {e.code} {msg}; retry", flush=True)
            time.sleep(10)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            if tries and k >= tries:
                raise
            w = min(60, 10 * k)
            print(f"[client] {path} {e!r}; retry {k} in {w}s", flush=True)
            time.sleep(w)


def health():
    with urllib.request.urlopen(URL + "/health", timeout=20) as r:
        return json.loads(r.read())


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def unmeans(r):
    return np.frombuffer(base64.b64decode(r["means"]), dtype=np.float16).reshape(r["shape"])


def generate(convs, max_new=400, seed=0, cap=False, sample=True, means=False, steer=None, chunk=48):
    """Returns the server dict; long lists are sent in chunks so one request never runs for too long."""
    out = None
    for k in range(0, len(convs), chunk):
        q = {"convs": convs[k:k + chunk], "max_new": max_new, "seed": seed * 7919 + k, "cap": cap,
             "sample": sample, "means": means}
        if steer:
            q["steer"] = steer
        r = _post("/generate", q)
        if means:
            r["means"] = unmeans(r)
            r.pop("shape", None)
        if out is None:
            out = r
        else:
            for key in r:
                out[key] = np.concatenate([out[key], r[key]]) if key == "means" else out[key] + r[key]
    return out


def acts(convs, replies, cap=False, chunk=64):
    ms, pj = [], []
    for k in range(0, len(convs), chunk):
        r = _post("/acts", {"convs": convs[k:k + chunk], "replies": replies[k:k + chunk], "cap": cap})
        ms.append(unmeans(r))
        pj += r["proj"]
    return np.concatenate(ms), pj


def letters(convs, letters=("A", "B", "C", "D"), cap=False, steer=None):
    q = {"convs": convs, "letters": list(letters), "cap": cap}
    if steer:
        q["steer"] = steer
    return _post("/letters", q)["probs"]


def set_dirs(dirs):
    """dirs: {name: (h, vec)}"""
    return _post("/set_dirs", {"dirs": {k: {"h": int(h), "vec": b64(v)} for k, (h, v) in dirs.items()}})
