"""CC-E2b eval runner (DESIGN.md §4): one merged 9B cell, vLLM, E2 stage 1's reads unchanged (e2_selfuse.evalrun):
the letter blocks (orig, clar, comp, compq) by E2's forced letter read (first token's top 20, P("X") + P(" X"),
renormalised over A / B), and the act block sampled at SC-N4's decoding (1536 tokens) and scored by SC-N4's
score.outcome. Thinking off (E2's prompt_ids); `think: false` is added to every row.

  /root/vllm/bin/python -m character_conduct.e2b_scope.evalrun --model DIR --cell VN-NACT_s1 --out DIR
Writes OUT/forced_{cell}.jsonl, OUT/sampled_{cell}.jsonl, OUT/run_info_{cell}.json. EL_FAKE_GEN=1 uses E2's
model-free generator (tests).
"""
import argparse
import json
import os
from pathlib import Path

from character_conduct.e2_selfuse import evalrun as E
from character_conduct.e2b_scope import items as I


def run(model, cell, out, gen=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    es = I.build()
    fpath, spath = out / f"forced_{cell}.jsonl", out / f"sampled_{cell}.jsonl"
    fd, sd = E.done_keys(fpath), E.done_keys(spath)
    todo_f = [e for e in es if e["form"] == "letter" and e["key"] not in fd]
    todo_s = [e for e in es if e["form"] == "code" and e["key"] not in sd]
    gen = gen or (E.FakeGen() if os.environ.get("EL_FAKE_GEN") == "1" else E.Gen(model))
    E.forced(gen, todo_f, cell, fpath)
    E.sampled(gen, todo_s, cell, spath)
    for f in (fpath, spath):   # the thinking flag on every row (E2's prompt_ids: enable_thinking=False)
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        f.write_text("".join(json.dumps({**r, "think": False}, ensure_ascii=False) + "\n" for r in rows))
    (out / f"run_info_{cell}.json").write_text(json.dumps(
        {"cell": cell, "model": str(model), "n_forced": len(todo_f), "n_sampled": len(todo_s),
         "jobs_sha": I.jobs_sha(es)}))
    print(f"[e2b-eval] {cell}: done ({len(todo_f)} forced, {len(todo_s)} sampled)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run(a.model, a.cell, a.out)


if __name__ == "__main__":
    main()
