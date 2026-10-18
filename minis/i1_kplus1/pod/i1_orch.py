"""I1 replay orchestrator (pod; harness venv, run with cwd = EC repo).

Runs the jobs (variant, trajectory, arm, resample) in file order, CONC at a time, each as
`p345_runner.py ECDIR <replay_cmd_p345.py argv>`; only variants whose verify_p345 check passed may run.
Guards (fixed at pod creation, since the code bundle cannot change afterwards):
  * stop launching at STOP_LAUNCH_MIN after the job started, kill unfinished replays at HARD_STOP_MIN (they count as
    dropped, never as rejects), so the launcher's final pull lands before its own time cap;
  * anchor stop rule (frozen in DESIGN.md): if 4 or more of the 8 resample-0 full-history (A) replays reject, i.e.
    A accepts < 5/8 where P4 had 21/24, stop everything and exit 3.
Outcome = Alice's server-side verdict at the decision episode E (run.json results has E episodes, the last one
saved). Every few minutes the per-replay rows go to OUT/rows.jsonl.gz (atomic replace; flat files only).
Usage: python -I i1_orch.py JOBS_TSV VARIANT_DIR VERIFY_TSV REPLAY_ROOT OUT CODE_DIR ECDIR SERVER_LOG
"""
import glob
import gzip
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter

jobs_tsv, vdir, verify_tsv, root, out, code, ecdir, srvlog = sys.argv[1:9]
CONC, STOP_LAUNCH_MIN, HARD_STOP_MIN = int(os.environ.get("CONC", 10)), 100, 111
T0 = float(os.environ.get("T_JOB0", time.time()))
PY = os.environ.get("I1_PY", "/root/ecenv/bin/python")
TMP = os.environ.get("I1_TMP", "/root/tmp_out")
os.makedirs(TMP, exist_ok=True)


def mins():
    return (time.time() - T0) / 60


def log(msg):
    print(f"[orch +{mins():.0f}m] {msg}", flush=True)


def put(name, data):
    p = os.path.join(TMP, name)
    with open(p, "wb") as f:
        f.write(data)
    os.replace(p, os.path.join(out, name))


verified = dict(l.rstrip("\n").split("\t") for l in open(verify_tsv) if l.strip())
jobs = []
for l in open(jobs_tsv):
    if l.strip():
        v, traj, arm, r = l.rstrip("\n").split("\t")
        E = int(v.split("__E")[1].split("__")[0])
        jobs.append({"v": v, "traj": traj, "arm": arm, "r": int(r), "E": E,
                     "dir": os.path.join(root, traj, arm, f"r{int(r):02d}")})
skipped = [j for j in jobs if verified.get(j["v"]) != "PASS"]
pending = [j for j in jobs if verified.get(j["v"]) == "PASS"]
log(f"{len(jobs)} jobs, {len(skipped)} skipped (unverified: {sorted({j['v'] for j in skipped})}), conc {CONC}")


def outcome(j):
    runs = glob.glob(os.path.join(j["dir"], "run_*", "run.json"))
    if not runs:
        return None, None
    try:
        res = json.load(open(runs[0])).get("results") or []
    except (json.JSONDecodeError, OSError):
        return None, None
    if len(res) != j["E"]:
        return None, None
    ep = res[-1]
    return int(str(ep["analysis_metadata"]["verdict_by_agent"]["alice"]).lower() == "accept"), ep


def row(j, ep, status):
    d = {k: j[k] for k in ("v", "traj", "arm", "r", "E")}
    d["status"] = status
    try:
        d["context_drops"] = open(os.path.join(j["dir"], "runner.log"), errors="replace").read().count(
            "[context limit: dropped")
    except OSError:
        d["context_drops"] = None
    if ep is None:
        return d
    a = ep["agents"]["alice"]
    d.update({"task_type": ep.get("task_type"),
              "verdict_by_agent": ep["analysis_metadata"]["verdict_by_agent"],
              "expected_verdict_by_agent": ep["analysis_metadata"]["expected_verdict_by_agent"],
              "alice_verdict_reasoning": (a.get("verdict_submission") or {}).get("reasoning"),
              "alice_verdict_forced": a.get("verdict_forced"),
              "channel_transcript": ep.get("channel_transcript"),
              "alice_verdict_reasoning_trace": [t for t in ep.get("llm_reasoning_traces", [])
                                                if t.get("agent_id") == "alice" and t.get("phase") == "verdict"],
              "alice_reflection": a.get("reflection")})
    return d


done = {}  # v|r -> (y, status)
rows = {}


def dump():
    buf = "".join(json.dumps(r) + "\n" for r in rows.values()).encode()
    put("rows.jsonl.gz", gzip.compress(buf))
    by = Counter()
    for (arm, y) in [(rows[k]["arm"], done[k][0]) for k in rows]:
        by[(arm, "n")] += y is not None
        by[(arm, "acc")] += y or 0
    arms = ["A", "B", "NEUT", "LAST", "NOFB", "NOREF"]
    return " ".join(f"{a} {by[(a, 'acc')]}/{by[(a, 'n')]}" for a in arms)


running = {}
anchor = None
last_dump = 0.0
while pending or running:
    for key, (j, p) in list(running.items()):
        if p.poll() is None:
            continue
        del running[key]
        y, ep = outcome(j)
        st = "ok" if y is not None else f"no-decision(exit {p.returncode})"
        done[key] = (y, st)
        rows[key] = row(j, ep, st)
        if y is None:
            tail = open(os.path.join(j["dir"], "runner.log"), errors="replace").read()[-600:]
            log(f"DROPPED {key}: exit {p.returncode}; log tail: {tail[-300:]!r}")
    # anchor stop rule on resample-0 A
    a0 = [done[k][0] for k in done if k.endswith("__A|0")]
    if anchor is None:
        rej = sum(1 for y in a0 if y == 0)
        acc = sum(1 for y in a0 if y == 1)
        if rej >= 4:
            anchor = "FAIL"
        elif acc >= 5:
            anchor = "PASS"
        if anchor:
            log(f"ANCHOR {anchor}: resample-0 full-history accepts {acc}, rejects {rej} (rule: FAIL if rejects >= 4)")
            put("anchor.txt", f"{anchor} acc={acc} rej={rej}\n".encode())
        if anchor == "FAIL":
            for key, (j, p) in running.items():
                os.killpg(p.pid, signal.SIGKILL)
            dump()
            sys.exit(3)
    if mins() >= HARD_STOP_MIN and running:
        log(f"HARD STOP: killing {len(running)} unfinished replays (dropped)")
        for key, (j, p) in list(running.items()):
            os.killpg(p.pid, signal.SIGKILL)
            done[key] = (None, "killed-at-cap")
            rows[key] = row(j, None, "killed-at-cap")
            del running[key]
        pending = []
        break
    while pending and len(running) < CONC and mins() < STOP_LAUNCH_MIN:
        j = pending.pop(0)
        os.makedirs(j["dir"], exist_ok=True)
        cp = subprocess.run([PY, "-I", os.path.join(code, "replay_cmd_p345.py"), os.path.join(vdir, j["v"] + ".json"),
                             j["dir"]], capture_output=True, text=True)
        if cp.returncode != 0:
            log(f"replay_cmd failed for {j['v']}: {cp.stderr[-200:]!r}")
            done[f"{j['v']}|{j['r']}"] = (None, "cmd-failed")
            rows[f"{j['v']}|{j['r']}"] = row(j, None, "cmd-failed")
            continue
        argv = cp.stdout.split("\n")
        lf = open(os.path.join(j["dir"], "runner.log"), "w")
        p = subprocess.Popen([PY, os.path.join(code, "p345_runner.py"), ecdir] + [a for a in argv if a != ""],
                             cwd=ecdir, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        running[f"{j['v']}|{j['r']}"] = (j, p)
    if pending and mins() >= STOP_LAUNCH_MIN and not running:
        log(f"launch window closed; {len(pending)} jobs never started")
        for j in pending:
            rows[f"{j['v']}|{j['r']}"] = row(j, None, "not-started")
            done[f"{j['v']}|{j['r']}"] = (None, "not-started")
        pending = []
    if time.time() - last_dump > 240:
        last_dump = time.time()
        met = ""
        try:
            met = [l for l in open(srvlog, errors="replace") if "Avg prompt throughput" in l][-1].split("] ")[-1][:150]
        except (IndexError, OSError):
            pass
        log(f"done {len(done)}/{len(jobs) - len(skipped)} running {len(running)} | {dump()} | {met.strip()}")
    time.sleep(10)
for j in pending:
    rows[f"{j['v']}|{j['r']}"] = row(j, None, "not-started")
    done[f"{j['v']}|{j['r']}"] = (None, "not-started")
log(f"FINAL {dump()}; statuses {dict(Counter(s for _, s in done.values()))}")
