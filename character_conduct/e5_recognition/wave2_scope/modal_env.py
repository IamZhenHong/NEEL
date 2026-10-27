"""CC-E5 wave-2 scoping (10-28, coordinator's request; no model runs): can Modal host ODCV-Bench's scenario environments?

For each released Qwen3.6-27B run asked for:
  build    the scenario's image from its own Dockerfile, with the build context run_benchmarks.py assembles
           (orchestrator_api.zip, then the scenario folder copied over it)
  replay   a Sandbox (network blocked, 1 CPU) and the run's bash calls in order through sandbox.exec, the way
           task_executor_api.execute_command runs them (/bin/sh -c, cwd /app, 30 s timeout, fresh shell per call, so
           only the filesystem carries state), each result compared with the recorded tool output
           (stdout, stderr, returncode)
  restore  at step k (the middle bash call), snapshot_filesystem(), a fresh Sandbox from the snapshot, and the /app
           manifest (sha256 of every file) compared with the replayed one; then the remaining calls replayed in the
           restored Sandbox
  --fixes  two departures from a bare exec, both to match how the harness ran a call: stdin from /dev/null (Flask's
           subprocess.run inherited a closed stdin; an open exec pipe hangs scripts that read stdin, e.g.
           validate_crisis_response.sh's `input=$(cat)`), and the clock: libfaketime (Debian's faketime package, added
           to the image) sets each call's clock to the time printed in that call's recorded output, else the latest
           earlier one in the run, else the run's first, else the minute of an `ls -l` line (dates feed file names,
           which later calls read back)

  /Applications/miniconda3/bin/python -m character_conduct.e5_recognition.wave2_scope.modal_env \
      --runs mandated:Ai-Healthcare-Administrator
  ... --all [--fixes]   every one of the 80 released runs
"""
import argparse
import json
import re
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from character_conduct.e5_recognition import traj as T

E = Path(__file__).resolve().parent
OUT = E / "results"
APP = "cc-e5-odcv-scope"
# Debian 11 (bullseye) left LTS in 08/2026: deb.debian.org's debian-security pool no longer holds its packages (404),
# so the 40 bullseye-slim scenario Dockerfiles no longer build anywhere as released. archive.debian.org holds the
# same versions (jq 1.6-2.1+deb11u3 etc.); this line, after FROM, is the only change to a scenario's Dockerfile
# without --fixes.
ARCHIVE_FIX = ("RUN printf 'deb http://archive.debian.org/debian bullseye main\\n"
               "deb http://archive.debian.org/debian bullseye-updates main\\n"
               "deb http://archive.debian.org/debian-security bullseye-security main\\n' > /etc/apt/sources.list"
               " && rm -f /etc/apt/sources.list.d/*")
FAKETIME_PKG = "RUN apt-get update && apt-get install -y faketime && rm -rf /var/lib/apt/lists/*"
LIBFAKETIME = "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1"
MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
CLOCK = [(re.compile(r"(2026)(\d\d)(\d\d)_(\d\d)(\d\d)(\d\d)"), lambda g: g),
         (re.compile(r"(2026)-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)"), lambda g: g),
         (re.compile(r"\w{3} (\w{3}) +(\d+) (\d\d):(\d\d):(\d\d) UTC (2026)"),
          lambda g: (g[5], f"{MON[g[0]]:02d}", f"{int(g[1]):02d}", g[2], g[3], g[4]))]
LS = re.compile(r" (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +(\d+) (\d\d):(\d\d) ")
MANIFEST = "find /app -type f ! -path '/app/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum"
COSMETIC = re.compile(r"\bls\b[^|;]*\s-[a-zA-Z]*l|\bstat\b|\bdu\b|\bdf\b")


def calls(msgs):
    """[(assistant turn index, bash command, recorded result dict)] in order; task_complete calls carry no
    environment action."""
    out = []
    a = 0
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        tools = [x for x in msgs[i + 1:i + 1 + len(m["tool_calls"])] if x["role"] == "tool"]
        for c, r in zip(m["tool_calls"], tools):
            if c["function"]["name"] == "bash":
                out.append((a, c["function"]["arguments"].get("command", ""), json.loads(r["content"])))
        a += 1
    return out


def clock_of(res):
    """The first full 2026 timestamp printed in a recorded result, as 'YYYY-MM-DD hh:mm:ss', or None."""
    txt = f"{res.get('stdout', '')}\n{res.get('stderr', '')}"
    for rx, f in CLOCK:
        m = rx.search(txt)
        if m:
            y, mo, d, h, mi, s = f(m.groups())
            return f"{y}-{mo}-{d} {h}:{mi}:{s}"
    return None


def clocks(cs):
    """Per call, the fake clock (see --fixes), or None when the run prints no time at all."""
    own = [clock_of(r) for _, _, r in cs]
    first = next((c for c in own if c), None)
    if first is None:
        for _, _, r in cs:
            m = LS.search(r.get("stdout", ""))
            if m:
                first = f"2026-{MON[m.group(1)]:02d}-{int(m.group(2)):02d} {m.group(3)}:{m.group(4)}:00"
                break
    out, last = [], first
    for c in own:
        last = c or last
        out.append(last)
    return out


def context(variant, scenario, tmp, fixes):
    ctx = Path(tmp) / "ctx"
    with zipfile.ZipFile(T.ODCV / "orchestrator_api.zip") as z:
        z.extractall(tmp)
    shutil.move(str(Path(tmp) / "orchestrator_api"), ctx)
    shutil.copytree(T.ODCV / f"{variant}_scenarios" / scenario, ctx, dirs_exist_ok=True)
    df = ctx / "Dockerfile"
    lines = df.read_text().splitlines()
    i = next(n for n, x in enumerate(lines) if x.startswith("FROM "))
    add = [ARCHIVE_FIX] if lines[i].strip() == "FROM debian:bullseye-slim" else []
    if fixes:
        add.append(FAKETIME_PKG)
    lines[i + 1:i + 1] = add
    df.write_text("\n".join(lines) + "\n")
    return ctx


def run_cmd(sb, cmd, fixes=False, clock=None):
    if fixes:
        env = {"LD_PRELOAD": LIBFAKETIME, "FAKETIME": f"@{clock}", "TZ": "UTC"} if clock else None
        p = sb.exec("/bin/sh", "-c", 'exec /bin/sh -c "$1" </dev/null', "sh", cmd, workdir="/app", timeout=30,
                    env=env)
    else:
        p = sb.exec("/bin/sh", "-c", cmd, workdir="/app", timeout=30)
    out, err = p.stdout.read(), p.stderr.read()
    rc = p.wait()
    return {"stdout": out, "stderr": err, "returncode": rc}


def same(a, b):
    return all(a[k] == b.get(k) for k in ("stdout", "stderr", "returncode"))


def one(app, variant, scenario, fixes):
    rec = {"variant": variant, "scenario": scenario, "fixes": fixes}
    msgs = T.parse(T.run_dir(variant, scenario) / "messages_record.txt")
    cs = calls(msgs)
    clk = clocks(cs) if fixes else [None] * len(cs)
    rec["n_calls"] = len(cs)
    rec["clock"] = next((c for c in clk if c), None)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        ctx = context(variant, scenario, tmp, fixes)
        try:
            img = modal.Image.from_dockerfile(ctx / "Dockerfile", context_dir=ctx)
            sb = modal.Sandbox.create("sleep", "infinity", app=app, image=img, timeout=900, block_network=True,
                                      cpu=1.0, memory=1024)
        except Exception as e:                      # noqa: BLE001
            rec["build"] = f"FAIL {type(e).__name__}: {str(e)[:300]}"
            return rec
    rec["build"] = "ok"
    rec["build_s"] = round(time.time() - t0, 1)
    k = len(cs) // 2
    try:
        res, t1 = [], time.time()
        for j, (turn, cmd, want) in enumerate(cs):
            if j == k:
                man_k = run_cmd(sb, MANIFEST)["stdout"]
                snap = sb.snapshot_filesystem()
            got = run_cmd(sb, cmd, fixes, clk[j])
            ok = same(got, want)
            res.append({"j": j, "turn": turn, "same": ok, "cosmetic": bool(COSMETIC.search(cmd)), "cmd": cmd[:200],
                        **({} if ok else {"got": {x: str(got[x])[:400] for x in got},
                                          "want": {x: str(want.get(x))[:400] for x in got}})})
        rec["replay_s"] = round(time.time() - t1, 1)
        rec["match"] = sum(r["same"] for r in res)
        rec["calls"] = [{x: r[x] for x in ("j", "turn", "same", "cosmetic")} for r in res]
        rec["mismatch"] = [r for r in res if not r["same"]]
        sb.terminate()
        if cs:
            t2 = time.time()
            sb2 = modal.Sandbox.create("sleep", "infinity", app=app, image=snap, timeout=900, block_network=True,
                                       cpu=1.0, memory=1024)
            rec["restore_manifest_same"] = run_cmd(sb2, MANIFEST)["stdout"] == man_k
            rec["restore_n_files"] = len(man_k.splitlines())
            tail = [same(run_cmd(sb2, cmd, fixes, clk[k + i]), want) for i, (_, cmd, want) in enumerate(cs[k:])]
            rec["restore_tail_match"] = f"{sum(tail)}/{len(tail)}"
            rec["restore_s"] = round(time.time() - t2, 1)
            sb2.terminate()
    except Exception as e:                          # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        try:
            sb.terminate()
        except Exception:                           # noqa: BLE001
            pass
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fixes", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="modal_env.json")
    a = ap.parse_args(argv)
    todo = [tuple(r.split(":")) for r in a.runs]
    if a.all:
        todo = [(v, s) for v in T.VARIANTS for s in T.scenarios()]
    app = modal.App.lookup(APP, create_if_missing=True)
    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(lambda r: one(app, *r, a.fixes), todo))
    OUT.mkdir(exist_ok=True)
    (OUT / a.out).write_text(json.dumps(recs, indent=1))
    for r in recs:
        print(r["variant"], r["scenario"], r.get("build"), f"match {r.get('match')}/{r['n_calls']}",
              "restore", r.get("restore_manifest_same"), r.get("restore_tail_match"), r.get("error", ""))


if __name__ == "__main__":
    main()
