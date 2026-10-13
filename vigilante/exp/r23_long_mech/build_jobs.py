"""R23 jobs (prereg vig-r23-prereg). 10 per cell per case (C1, C3), P0, thinking off.
  jobs_gemma.jsonl  : Gemma 4 31B. Fresh / after R21's 16-chat history (as turns) / the same chats as a log in the
                      system prompt; lines none / both / strong; reminder variant after the history. 540 jobs. Not
                      committed (4 MB); the two histories are, in gemma_histories.json.
  jobs_qwen_s1.jsonl: Qwen3.6-27B stage 1: the 8 history blocks per case (3 samples each, group 'chat') and the fresh
                      cells. The long cells are written on the pod by make_long.py from the recorded blocks.
  uv run python -m vigilante.exp.r23_long_mech.build_jobs"""
import json
from pathlib import Path

from vigilante.cases.authority import APPROVAL, R23_LINES, R23_MSGS
from vigilante.exp.r23_long_mech.common import job, transcript, write

H = Path(__file__).parent
R21 = H.parent / "r21_approval_fix" / "results"
# R21's chosen histories (the C1 one from the rerun with chat 9 changed; see R21 REPORT, deviation)
R21_HIST = {"C1": R21 / "c1fix" / "long" / "jobs_long.jsonl", "C3": R21 / "long" / "jobs_long.jsonl"}


SAVED = H / "gemma_histories.json"          # committed copy (R21's results folder is not in git)


def r21_history(cid):
    if not R21_HIST[cid].exists():
        return json.loads(SAVED.read_text())[cid]
    hs = {json.dumps(j["history"]) for j in map(json.loads, open(R21_HIST[cid])) if j["case"].startswith(cid)}
    assert len(hs) == 1, (cid, len(hs))
    return json.loads(hs.pop())


def main():
    js = []
    SAVED.write_text(json.dumps({cid: r21_history(cid) for cid in APPROVAL}, ensure_ascii=False))
    for cid in APPROVAL:
        hist = r21_history(cid)
        log = transcript(hist, "gemma4")
        for ln in R23_LINES:
            for m in R23_MSGS:
                js += [job("R23", f"{cid}#r23-{m}-{ln}", i, "gemma4", "short") for i in range(10)]
        for ln in R23_LINES:                       # grouped by line: episodes sharing a prefix run together
            for m in R23_MSGS:
                js += [job("R23", f"{cid}#r23-{m}-{ln}", i, "gemma4", "h16", group="l16", history=hist)
                       for i in range(10)]
        for m in R23_MSGS:
            js += [job("R23", f"{cid}#r23-{m}-both-rem", i, "gemma4", "h16", group="l16", history=hist)
                   for i in range(10)]
        for ln in ("none", "both"):
            for m in R23_MSGS:
                js += [job("R23", f"{cid}#r23-{m}-{ln}", i, "gemma4", "tr16", group="l16", system_suffix=log)
                       for i in range(10)]
    write(H / "jobs_gemma.jsonl", js)

    js = [job("R23H", f"{cid}#r23-hist-b{b}", i, "qwen", "hist", group="chat", handover=False)
          for cid in APPROVAL for b in range(8) for i in range(3)]
    js += [job("R23", f"{cid}#r23-{m}-{ln}", i, "qwen", "short")
           for cid in APPROVAL for ln in R23_LINES for m in R23_MSGS for i in range(10)]
    write(H / "jobs_qwen_s1.jsonl", js)


if __name__ == "__main__":
    main()
