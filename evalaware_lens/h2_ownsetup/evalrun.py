"""EL-H2 eval runner (DESIGN.md §4): one merged 9B cell, vLLM, the frozen items (items.build()).

EL-S2's runner with H2's entries: the same Gen (vLLM 0.30, bf16, 8192 context, prefix caching), the same reads
(letters: one forced token, top-20 log-probs, P("X") + P(" X") renormalised; code: SC-N4's decoding, scored by SC-N4's
score.outcome) and the same resume-by-key files OUT/forced_{cell}.jsonl / OUT/eval_{cell}.jsonl. Progress lines per
2,000 rows (S2's CHUNK).

  /root/vllm/bin/python -m evalaware_lens.h2_ownsetup.evalrun --model DIR --cell VN-ACT_s1 --out DIR
EL_FAKE_GEN=1 runs S2's model-free generator (tests).
"""
import argparse
import json
import os
import time
from pathlib import Path

from evalaware_lens.h2_ownsetup import items as I
from evalaware_lens.s2_evalself import evalrun as R

FORCED = ("f1c", "f2")


def run(model, cell, out, *, gen=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    entries = I.build()
    fpath, epath = out / f"forced_{cell}.jsonl", out / f"eval_{cell}.jsonl"
    gen = gen or (R.FakeGen() if os.environ.get("EL_FAKE_GEN") == "1" else R.Gen(model))
    t0 = time.time()
    fd, ed = R.done_keys(fpath), R.done_keys(epath)
    todo_f = [e for e in entries if e["block"] in FORCED and e["key"] not in fd]
    todo_s = [e for e in entries if e["block"] not in FORCED and e["key"] not in ed]
    print(f"[h2-eval] {cell}: {len(todo_f)} forced, {len(todo_s)} sampled to do", flush=True)
    R.sampled(gen, todo_s, cell, epath)
    t_s = time.time() - t0
    R.forced(gen, todo_f, cell, fpath)
    info = {"model": str(model), "cell": cell, "load_s": gen.load_s, "n_forced": len(todo_f), "n_sampled": len(todo_s),
            "sampled_s": round(t_s, 1), "total_s": round(time.time() - t0, 1)}
    (out / f"run_info_{cell}.json").write_text(json.dumps(info))
    print(f"[h2-eval] {cell}: done {info}", flush=True)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="h2 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run(a.model, a.cell, a.out)
    if os.environ.get("EL_FAKE_GEN") != "1":
        os._exit(0)   # vLLM 0.30's teardown can abort (exit 134) after every file is written (SC-M28)


if __name__ == "__main__":
    main()
