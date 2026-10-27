"""CC-E2 stage 2 runner (DESIGN2.md §4): one merged 9B cell, its frozen entries from jobs/stage2.jsonl.gz.

  /root/vllm/bin/python -m character_conduct.e2_selfuse.evalrun2 --model DIR --cell VN-NW_s1 --out DIR

Reads as stage 1 (evalrun.forced / evalrun.sampled): desk acts forced at the option id, code acts sampled 10 per case
and scored by SC-N4. Writes OUT/forced_{cell}.jsonl, OUT/sampled_{cell}.jsonl, OUT/run_info_{cell}.json; resumes by key.
"""
import argparse
import json
import os
from pathlib import Path

from character_conduct.e2_selfuse import evalrun as R
from character_conduct.e2_selfuse import items2 as I2


def run(model, cell, out, gen=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    es = [e for e in I2.read_jobs() if e["cell"] == cell]
    fpath, spath = out / f"forced_{cell}.jsonl", out / f"sampled_{cell}.jsonl"
    fd, sd = R.done_keys(fpath), R.done_keys(spath)
    todo_f = [e for e in es if e["form"] == "act" and e["key"] not in fd]
    todo_s = [e for e in es if e["form"] == "code" and e["key"] not in sd]
    if todo_f or todo_s:
        gen = gen or (R.FakeGen() if os.environ.get("EL_FAKE_GEN") == "1" else R.Gen(model))
        R.forced(gen, todo_f, cell, fpath)
        R.sampled(gen, todo_s, cell, spath)
    (out / f"run_info_{cell}.json").write_text(json.dumps({"cell": cell, "model": str(model), "n": len(es)}))
    print(f"[e2s2-eval] {cell}: done ({len(todo_f)} forced, {len(todo_s)} sampled)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run(a.model, a.cell, a.out)


if __name__ == "__main__":
    main()
