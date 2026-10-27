"""CC-E5 wave 2 runner: arms (a) own judgement, (b) rule reminder, (c) unchanged, at the first-mention turn of each
target, N repeats, on Modal Sandboxes restored from starts.py's snapshots.

  1. start check: every target's restored start state must pass its file check (checks.py) as not violated; a target
     whose check already fires at M is dropped before any continuation.
  2. jobs: for rep in reps, for target, the three arms in a seeded random order; a thread pool takes them in that
     order, so the arms of a target and rep run side by side (interleaved, same provider, same minutes).
  3. each episode (episode.run) -> its final file state -> checks.check -> y.
  4. failed episodes (an API error after retries, a cost stop, a sandbox error) are retried once after the batch;
     still failed -> excluded (counted per arm).
Episodes go to results/episodes.jsonl (one line each, resumable: a (target, arm, rep) already there is skipped).

  /Applications/miniconda3/bin/python -m character_conduct.e5_recognition.wave2.run_w2 --smoke      # 1 target x 3
  /Applications/miniconda3/bin/python -m character_conduct.e5_recognition.wave2.run_w2 --reps 0-4   # N = 5
  /Applications/miniconda3/bin/python -m character_conduct.e5_recognition.wave2.run_w2 --reps 5-9   # extension
"""
import argparse
import json
import random
import threading
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import character_conduct.common.ledger as LG
from character_conduct.e5_recognition.wave2 import checks as C
from character_conduct.e5_recognition.wave2 import env as EV
from character_conduct.e5_recognition.wave2 import episode as EP
from character_conduct.e5_recognition.wave2 import targets as TG

R = Path(__file__).resolve().parent / "results"
ARMS = ("a", "b", "c")
GEN_CAP = {"smoke": 1.0, "main": 18.0, "extension": 18.0}     # this stage's own generation stop (expected ~$13.4)
WORKERS = 16
_lock = threading.Lock()


def key(t):
    return f"{t['variant']}|{t['scenario']}"


def load():
    starts = {key(s): s for s in json.loads((R / "starts.json").read_text())}
    arms = {key(a): a for a in json.loads((TG.E / "arms.json").read_text()) if a["source"]}
    pfiles = json.loads((R / "pristine_files.json").read_text())
    return [starts[k] for k in sorted(arms)], arms, pfiles


def state(start, ep, pfiles):
    return {"pristine": start["pristine"], "final": ep["manifest"], "files": ep["files"], "pfiles": pfiles}


def start_checks(a, starts, pfiles):
    """{target: (violated, why)} on each restored start state."""
    import modal

    def one(s):
        sb = EV.sandbox(a, s["variant"], s["scenario"], image=modal.Image.from_id(s["image_id"]))
        try:
            man = EV.manifest(sb, EV.scope(s["variant"], s["scenario"]))
            files = EV.read(sb, sorted(p for p in man if man[p] != s["pristine"].get(p)), cap=65536)
        finally:
            sb.terminate()
        return key(s), C.check(s["variant"], s["scenario"], {"pristine": s["pristine"], "final": man,
                                                             "files": files, "pfiles": pfiles[key(s)]})
    with ThreadPoolExecutor(12) as ex:
        return dict(ex.map(one, starts))


def jobs(starts, reps, seed=0):
    out = []
    for rep in reps:
        for s in starts:
            arms = list(ARMS)
            random.Random(f"{seed}|{key(s)}|{rep}").shuffle(arms)
            out += [(s, arm, rep) for arm in arms]
    return out


def run_one(gen, a, s, arm, rep, arms, pfiles, out_file, attempt):
    insert = arms[key(s)][arm] if arm in ("a", "b") else None
    rec = {"target": key(s), "variant": s["variant"], "scenario": s["scenario"], "arm": arm, "rep": rep,
           "attempt": attempt, "t0": time.time()}
    try:
        ep = EP.run(gen, a, s, insert, seed=zlib.crc32(f"{key(s)}|{arm}|{rep}|{attempt}".encode()) % 100000)
        y, why = C.check(s["variant"], s["scenario"], state(s, ep, pfiles[key(s)]))
        rec.update({"end": ep["end"], "cycles": ep["cycles"], "calls": ep["calls"], "cost": ep["cost"], "y": y,
                    "why": why, "manifest": ep["manifest"], "files": ep["files"], "messages": ep["messages"]})
    except Exception as e:                          # noqa: BLE001
        rec.update({"end": f"error:{type(e).__name__}", "detail": str(e)[:300], "trace": traceback.format_exc()[-600:]})
    rec["failed"] = rec["end"] is None or rec["end"].startswith("error") or "cost_stop" in rec["end"]
    rec["t1"] = time.time()
    with _lock, open(out_file, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def main(argv=None):
    from character_conduct.common import ledger  # noqa: F401  (first: books to character_conduct/ledger.jsonl)
    from persona.operative.frontier import load_key
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reps", default="0-4")
    a = ap.parse_args(argv)
    starts, arms, pfiles = load()
    stage = "smoke" if a.smoke else ("main" if a.reps == "0-4" else "extension")
    out_file = R / ("episodes_smoke.jsonl" if a.smoke else "episodes.jsonl")
    lo, hi = (int(x) for x in a.reps.split("-"))
    reps = [0] if a.smoke else list(range(lo, hi + 1))
    app = EV.app()
    sc_file = R / "start_checks.json"
    if sc_file.exists():
        sc = json.loads(sc_file.read_text())
    else:
        sc = {k: list(v) for k, v in start_checks(app, starts, pfiles).items()}
        sc_file.write_text(json.dumps(sc, indent=1))
    bad = [k for k, (v, _) in sc.items() if v]
    print(f"[w2] start checks: {len(starts) - len(bad)} clean, dropped {bad}")
    starts = [s for s in starts if key(s) not in bad]
    if a.smoke:
        starts = [s for s in starts if s["scenario"] == "Ai-Hiring-Assistant" and s["variant"] == "mandated"]
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            if not r["failed"]:
                done.add((r["target"], r["arm"], r["rep"]))
    todo = [j for j in jobs(starts, reps) if (key(j[0]), j[1], j[2]) not in done]
    print(f"[w2] {stage}: {len(todo)} episodes to run ({len(done)} done)")
    LG.check_both("CC-E5", TG.CAP, planned=GEN_CAP[stage], kind="api")
    book = LG.Booker("CC-E5", "api", EP.GEN_NOTE)
    gen = EP.Gen(load_key(), book, GEN_CAP[stage])
    try:
        with ThreadPoolExecutor(WORKERS) as ex:
            recs = list(ex.map(lambda j: run_one(gen, app, *j, arms, pfiles, out_file, 0), todo))
        failed = [(next(s for s in starts if key(s) == r["target"]), r["arm"], r["rep"]) for r in recs if r["failed"]]
        if failed:
            print(f"[w2] retrying {len(failed)} failed episodes once")
            with ThreadPoolExecutor(WORKERS) as ex:
                recs += list(ex.map(lambda j: run_one(gen, app, *j, arms, pfiles, out_file, 1), failed))
    finally:
        book.flush()
    ok = [r for r in recs if not r["failed"]]
    print(f"[w2] {len(ok)} ok, {len(recs) - len(ok)} failed attempts; generation ${gen.spent:.2f}")
    for arm in ARMS:
        ys = [r["y"] for r in ok if r["arm"] == arm]
        print(f"   arm {arm}: {sum(ys)}/{len(ys)} violate (file check)")


if __name__ == "__main__":
    main()
