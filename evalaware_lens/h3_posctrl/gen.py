"""EL-H3 generation (DESIGN.md section 4; pod, vLLM venv). One model per process, thinking off.

Each inducer job's own reply, with L2/L3's generator (F2's DECODING, 400 new tokens) and L2/L3's seed for the task and
wording, so the inducer reply and L2/L3's neutral reply share their sampling seed. No side branches. The neutral and
eval_safety jobs are not generated: their replies come from L2/L3 (jobs/l2l3_replies.json.gz).
Writes OUT/gen_{model}.jsonl: key, reply, reply_ids, finish.

  /root/vllm/bin/python -m evalaware_lens.h3_posctrl.gen --model-key q27 --path DIR --out DIR
"""
import argparse
import json
import time
from pathlib import Path

from evalaware_lens.h3_posctrl import items as I

T0 = time.time()


def log(m):
    print(f"[el-h3 gen {time.time() - T0:6.0f}s] {m}", flush=True)


def run(gen, key, out_dir, jobs=None):
    jobs = [j for j in (jobs if jobs is not None else I.load()) if j["kind"] in I.INDUCER_KINDS]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log(f"{key}: generating {len(jobs)} replies")
    rep = gen.replies([gen.ids(I.messages(j)) for j in jobs], [j["seed"] for j in jobs])
    with open(out / f"gen_{key}.jsonl", "w") as f:
        for j, (text, ids, fin) in zip(jobs, rep):
            f.write(json.dumps({"key": j["key"], "reply": text, "reply_ids": ids, "finish": fin}, ensure_ascii=False)
                    + "\n")
    log(f"{key}: wrote gen_{key}.jsonl ({len(jobs)} replies)")


def main():
    from evalaware_lens.l2l3_cuereads.gen import VGen
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(VGen(a.path), a.model_key, a.out)


if __name__ == "__main__":
    main()
