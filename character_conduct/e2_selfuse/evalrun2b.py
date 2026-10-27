"""CC-E2 stage 2b runner: one merged 9B cell, its frozen entries from jobs/stage2b.jsonl.gz (forced desk reads only).

  /root/vllm/bin/python -m character_conduct.e2_selfuse.evalrun2b --model DIR --cell VN-NW_s1 --out DIR
"""
import argparse
import json
import os
from pathlib import Path

from character_conduct.e2_selfuse import evalrun as R
from character_conduct.e2_selfuse import items2b as I2B


def run(model, cell, out, gen=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    es = [e for e in I2B.read_jobs() if e["cell"] == cell]
    fpath = out / f"forced_{cell}.jsonl"
    todo = [e for e in es if e["key"] not in R.done_keys(fpath)]
    if todo:
        gen = gen or (R.FakeGen() if os.environ.get("EL_FAKE_GEN") == "1" else R.Gen(model))
        R.forced(gen, todo, cell, fpath)
    (out / f"run_info_{cell}.json").write_text(json.dumps({"cell": cell, "model": str(model), "n": len(es)}))
    print(f"[e2s2b-eval] {cell}: done ({len(todo)} forced)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run(a.model, a.cell, a.out)


if __name__ == "__main__":
    main()
