"""SC-M28 eval: SC-M28's items (SC-S1's frozen items with the format demos, items.py) through V1's runner (chunks,
one nudge for unparsed decisions, resumable) on one merged OLMo model with olmo.OlmoGen and olmo.parse_any, then the
forced reads on the same loaded model.

  /root/vllm/bin/python -m self_concept.m28_olmo_stages.evalrun --model DIR --cell base-W_s1 --out DIR [--smoke]

Writes OUT/eval_{cell}.jsonl (V1's records) and OUT/forced_{cell}.jsonl (olmo.forced_rows). --smoke: only the items
with sample index 0 (356; the pre-freeze parse check). V1_FAKE_GEN=1 (dry runs): SC-S1's fake replies and hashed
log-probs, no model. A real run ends with os._exit(0) once its files are written: vLLM 0.30's engine teardown can
abort ("terminate called without an active exception", exit 134) after the work is done (the first parse check).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from persona_affect.common.util import seed
from persona_affect.v1_value_hook import evalrun as R
from self_concept.m28_olmo_stages import olmo as O
from self_concept.m28_olmo_stages.items import items
from self_concept.s1_scale import evalrun as S1E


class FakeGen:
    """Dry runs: SC-S1's fake replies; log-probs hashed from the sequence, so every model gives the same ones."""

    def __init__(self, tok):
        self.tok, self.load_s = tok, 0.0

    def __call__(self, batch):
        return S1E.fake_gen(batch)

    def logprobs(self, seqs):
        return [-(seed(json.dumps(p[-8:] + c)) % 1000) / 200.0 for p, c in seqs]


def run(model, cell, out, smoke=False, gen=None):
    O.patch_parsers()
    its = items()
    if smoke:
        its = [it for it in its if it["i"] == 0]
    if gen is None:
        gen = O.OlmoGen(model)
    R.run(model, cell, [], out, items=its, gen=gen)
    fpath = Path(out) / f"forced_{cell}.jsonl"
    if fpath.exists():
        print(f"[m28-eval] {cell}: forced reads exist", flush=True)
        return
    jobs = O.forced_jobs(gen.tok, its)
    seqs = [pc for _, conts in jobs for pc in conts.values()]
    rows = O.forced_rows(jobs, gen.logprobs(seqs), cell)
    fpath.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[m28-eval] {cell}: {len(rows)} forced reads ({len(seqs)} sequences)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m28 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    gen = None
    if os.environ.get("V1_FAKE_GEN") == "1":
        from transformers import AutoTokenizer
        gen = FakeGen(AutoTokenizer.from_pretrained(a.model))
    run(a.model, a.cell, a.out, a.smoke, gen)
    if gen is None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
