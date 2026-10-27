"""CC-E5 wave 2: the ODCV-Bench environment on Modal Sandboxes.

Built on the scoping run (wave2_scope/modal_env.py, its --fixes variant): each scenario's own Dockerfile with
apt pointed at archive.debian.org and Debian's faketime package added; every agent call runs as the harness ran it
(/bin/sh -c in /app, 30 s timeout, stdin from /dev/null) with the clock set by libfaketime; network blocked.

File state: `manifest` hashes every file under SCOPE (/app, /usr/local/bin and the /var paths the scenario's
Dockerfile copies), leaving out the harness's own files (EXCLUDE).
"""
import json
import re
import tempfile
from pathlib import Path

import modal

from character_conduct.e5_recognition import traj as T
from character_conduct.e5_recognition.wave2_scope import modal_env as ME

APP = "cc-e5-wave2"
EXCLUDE = ("/app/task_executor_api.py", "/app/templates/", "/app/__pycache__/", "/app/requirements.txt",
           "/app/prompt.json")
SANDBOX = {"timeout": 3600, "block_network": True, "cpu": 1.0, "memory": 1024}
TIMEOUT_S = 30
READ = ("import json,sys\nout={}\nfor p in sys.argv[2:]:\n try:\n  b=open(p,'rb').read(int(sys.argv[1]))\n"
        "  out[p]=b.decode('utf-8','replace')\n except Exception as e:\n  out[p]=None\nprint(json.dumps(out))")


def app():
    return modal.App.lookup(APP, create_if_missing=True)


def scope(variant, scenario):
    df = (T.ODCV / f"{variant}_scenarios" / scenario / "Dockerfile").read_text()
    var = sorted({m.rstrip("/") for m in re.findall(r"^COPY\s+\S+\s+(/var/\S+)", df, re.M)})
    return ["/app", "/usr/local/bin"] + var


def sandbox(a, variant, scenario, image=None):
    """A fresh Sandbox of the scenario (pristine image), or of a snapshot `image`."""
    if image is not None:
        return modal.Sandbox.create("sleep", "infinity", app=a, image=image, **SANDBOX)
    with tempfile.TemporaryDirectory() as tmp:
        ctx = ME.context(variant, scenario, tmp, fixes=True)
        img = modal.Image.from_dockerfile(ctx / "Dockerfile", context_dir=ctx)
        return modal.Sandbox.create("sleep", "infinity", app=a, image=img, **SANDBOX)


def run(sb, cmd, clock=None):
    """One agent call, the harness's way; the result shaped as agent_main._dispatch_shell returns it."""
    import time
    t0 = time.time()
    r = ME.run_cmd(sb, cmd, fixes=True, clock=clock)
    if r["returncode"] == -1 and time.time() - t0 >= TIMEOUT_S - 1:   # subprocess.run(timeout=30) in the harness
        r = {"stdout": "", "stderr": "Command timed out", "returncode": -1}
    return {"stdout": r["stdout"], "stderr": r["stderr"], "role": "tool", "returncode": r["returncode"]}


def manifest(sb, paths):
    """{path: sha256} for every file under `paths`, minus EXCLUDE."""
    cmd = "find " + " ".join(paths) + " -type f -print0 2>/dev/null | sort -z | xargs -0 -r sha256sum"
    out = ME.run_cmd(sb, cmd)["stdout"]
    m = {}
    for line in out.splitlines():
        h, _, p = line.partition("  ")
        if p and not p.startswith(EXCLUDE):
            m[p] = h
    return m


def read(sb, paths, cap=4096):
    """{path: first `cap` bytes as text (None if unreadable)}."""
    if not paths:
        return {}
    p = sb.exec("python3", "-c", READ, str(cap), *paths, timeout=60)
    out = p.stdout.read()
    p.wait()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {x: None for x in paths}


def diff(before, after):
    return {"changed": sorted(p for p in before if p in after and after[p] != before[p]),
            "deleted": sorted(p for p in before if p not in after),
            "created": sorted(p for p in after if p not in before)}
