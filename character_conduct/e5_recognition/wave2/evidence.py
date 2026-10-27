"""CC-E5 wave 2, before any wave-2 data: the file-state evidence the per-scenario check rules are written from and
validated on.

For each of the 34 target runs (released Qwen3.6-27B violating runs reachable at their first-mention turn), every
released run of that scenario and variant (all 12 models) is replayed on Modal from the pristine image, and the file
state is recorded: pristine, (Qwen only) at the first-mention turn M, and final; the changed / deleted / created files;
their pristine and final contents (first 4 KB, at most 40 files). Labels are the released 4-judge medians.

  /Applications/miniconda3/bin/python -m character_conduct.e5_recognition.wave2.evidence [--only qwen3.6-27b]
"""
import argparse
import csv
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from character_conduct.e5_recognition import traj as T
from character_conduct.e5_recognition.wave2 import env as EV
from character_conduct.e5_recognition.wave2 import targets as TG
from character_conduct.e5_recognition.wave2_scope import modal_env as ME

OUT = Path(__file__).resolve().parent / "results" / "evidence.jsonl"
MAX_FILES = 40
_lock = threading.Lock()


def medians():
    out = {}
    for r in csv.DictReader(open(T.RES / "evaluations" / "judge_all" / "scores_final_median.csv")):
        for k, v in r.items():
            if k != "scenario" and v not in ("", None):
                try:
                    out[(k, r["scenario"])] = float(v)
                except ValueError:
                    pass
    return out


def one(a, model, variant, scenario, M, med):
    rec = {"model": model, "variant": variant, "scenario": scenario, "median": med}
    f = T.run_dir(variant, scenario, model=model) / "messages_record.txt"
    cs = ME.calls(T.parse(f))
    clk = ME.clocks(cs)
    rec["n_calls"] = len(cs)
    paths = EV.scope(variant, scenario)
    t0 = time.time()
    sb = EV.sandbox(a, variant, scenario)
    try:
        pristine = EV.manifest(sb, paths)
        match, at_m = 0, None
        for j, (turn, cmd, want) in enumerate(cs):
            if model == TG.MODEL and at_m is None and turn >= M:
                at_m = EV.manifest(sb, paths)
            got = EV.run(sb, cmd, clk[j])
            match += all(got[k] == want.get(k) for k in ("stdout", "stderr", "returncode"))
        final = EV.manifest(sb, paths)
        if model == TG.MODEL and at_m is None:
            at_m = final
        rec["match"] = match
        rec["n_pristine"] = len(pristine)
        rec["diff"] = EV.diff(pristine, final)
        if at_m is not None:
            rec["diff_at_M"] = EV.diff(pristine, at_m)
            rec["diff_after_M"] = EV.diff(at_m, final)
        touched = (rec["diff"]["changed"] + rec["diff"]["created"])[:MAX_FILES]
        rec["final"] = EV.read(sb, touched)
        sb.terminate()
        sb0 = EV.sandbox(a, variant, scenario)
        rec["pristine"] = EV.read(sb0, (rec["diff"]["changed"] + rec["diff"]["deleted"])[:MAX_FILES])
        sb0.terminate()
    except Exception as e:                          # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["trace"] = traceback.format_exc()[-800:]
        try:
            sb.terminate()
        except Exception:                           # noqa: BLE001
            pass
    rec["seconds"] = round(time.time() - t0, 1)
    with _lock, open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="one model")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["model"], r["variant"], r["scenario"]))
    med = medians()
    jobs = []
    for t in TG.targets():
        for m in TG.MODELS:
            if a.only and m != a.only:
                continue
            k = (f"{m}-{t['variant']}", t["scenario"])
            if (m, t["variant"], t["scenario"]) in done or k not in med:
                continue
            if not (T.run_dir(t["variant"], t["scenario"], model=m) / "messages_record.txt").exists():
                continue
            jobs.append((m, t["variant"], t["scenario"], t["M"], med[k]))
    print(f"[evidence] {len(jobs)} replays ({len(done)} already done)")
    app = EV.app()
    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(lambda j: one(app, *j), jobs))
    print(f"[evidence] errors {sum('error' in r for r in recs)}")


if __name__ == "__main__":
    main()
