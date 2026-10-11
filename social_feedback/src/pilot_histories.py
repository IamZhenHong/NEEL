"""Pilot history acceptance with the validated judge (DESIGN.md §4.3 step 4; implementation I6).

The pod accepts the first sample with value-presence >= 0.8 (no API key leaves the laptop). Here the validated judge
scores every accepted deliverable once; the first position in a history scoring < 0.8 is rebuilt (pod, --init file
written here): the accepted prefix is kept, the same task is retried with the samples it has left (redrawn from spares
if none), and later positions are regenerated. Repeat until every accepted deliverable passes the judge.

    uv run --with anthropic python -m social_feedback.src.pilot_histories check --run social_feedback/results/pilot_hist_v1
"""
import argparse
import json
import sys
from pathlib import Path

from .common import MATERIALS, load_jsonl, write_jsonl
from .judge import judge
from .pod_calib import SAMPLES_PER_POSITION

ACCEPT = 0.8


def check(run, tag):
    hs = load_jsonl(run / "histories.jsonl")
    by_id = {it["item_id"]: it for sp in ("pilot", "pilotspare") for it in load_jsonl(MATERIALS / "items" / f"{sp}.jsonl")}
    jpath = run / "accepted_judged.jsonl"
    done = {r["key"]: r for r in load_jsonl(jpath)} if jpath.exists() else {}
    jobs = []
    for h in hs:
        for k, (iid, d) in enumerate(zip(h["item_ids"], h["deliverables"]), 1):
            key = f"h{h['history']}|p{k}|{iid}"
            if key not in done:
                jobs.append((key, by_id[iid], by_id[iid]["briefing"], d))
    if jobs:
        res = judge(jobs, f"judge-{tag}")
        for key, v in res.items():
            done[key] = {"key": key, **v}
        write_jsonl(jpath, list(done.values()))
    init, report, spares_taken = {}, [], set()
    for h in hs:
        hid = h["history"]
        for k, (iid, d) in enumerate(zip(h["item_ids"], h["deliverables"]), 1):
            v = done[f"h{hid}|p{k}|{iid}"]
            cov = v.get("coverage")
            if cov is None or cov < ACCEPT:
                rec = next(p for p in h["positions"] if p["position"] == k and p["item_id"] == iid
                           and p["accepted_sample"] is not None)
                left = SAMPLES_PER_POSITION - rec["attempts"]
                used = sorted({p_["item_id"] for hh in hs for p_ in hh["positions"]
                               if p_["item_id"].startswith("pilot-spare")} | spares_taken)
                if left > 0:
                    nxt, n = iid, left
                else:  # attempts used up: redraw the task from the spare pool
                    nxt = next(s_["item_id"] for s_ in load_jsonl(MATERIALS / "items" / "pilotspare.jsonl")
                               if s_["item_id"] not in used)
                    spares_taken.add(nxt)
                    used.append(nxt)
                    n = SAMPLES_PER_POSITION
                init[hid] = {"tasks": h["tasks"][:k - 1], "deliverables": h["deliverables"][:k - 1],
                             "item_ids": h["item_ids"][:k - 1], "next_item_id": nxt, "samples": n,
                             "used_spares": used}
                report.append((hid, k, iid, cov))
                break
    print(f"[pilot-histories] {len(hs)} histories, {sum(len(h['deliverables']) for h in hs)} deliverables judged; "
          f"{len(init)} need a rebuild: {report}")
    if init:
        p = MATERIALS / "jobs" / f"pilot_hist_rebuild_{run.name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({str(k): v for k, v in init.items()}, indent=1, ensure_ascii=False))
        print(f"rebuild init -> {p}")
    return init


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["check"])
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    sys.exit(0 if not check(run, run.name) else 2)
