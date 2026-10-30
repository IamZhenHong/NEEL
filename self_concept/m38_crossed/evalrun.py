"""SC-M38 eval on one merged model (SC-M27's Qwen path: V1's own vLLM generator, bf16, thinking off).

  1. reads: every read item's greedy next-token top-20 log-probs, summed per option letter (a letter outside the top 20
     gets probability 0), written to OUT/reads_{cell}.jsonl ({key, probs {letter: p}, mass, top}); resumable.
  2. gate: SC-M27's own wmc|round and sr|round items through V1's runner (sampled with V1's decoding and each item's
     own seed, one nudge for an unparsed decision), records appended to OUT/eval_{cell}.jsonl as in M27.

  /root/vllm/bin/python -m self_concept.m38_crossed.evalrun --model MERGED --cells Q9-ROUND_s1 --out DIR
V1_FAKE_GEN=1 (dry runs) answers from a hash of the key alone: M27's fake replies for the gate, hashed letter
probabilities for the reads; nothing is loaded.
"""
import argparse
import json
import math
import os
from pathlib import Path

from persona_affect.common.util import seed
from persona_affect.v1_value_hook import evalrun as R
from self_concept.m27_matrix import evalrun as M27E
from self_concept.m38_crossed.items import gate_items, read_items

TOPK = 20


def _done(path):
    if not path.exists():
        return set()
    return {json.loads(x)["key"] for x in path.read_text().splitlines() if x.strip()}


def fake_probs(it):
    h = seed(it["key"])
    w = [1 + (h >> (4 * j)) % 7 for j in range(len(it["letters"]))]
    return {L: w[j] / sum(w) * 0.9 for j, L in enumerate(it["letters"])}


def letter_probs(gen, its):
    from vllm import SamplingParams, TokensPrompt
    from persona_affect.v1_value_hook.fmt import prompt_ids
    res = gen.llm.generate([TokensPrompt(prompt_token_ids=prompt_ids(gen.tok, it["messages"], it["tools"]))
                            for it in its], SamplingParams(max_tokens=1, temperature=0, logprobs=TOPK),
                           use_tqdm=False)
    out = []
    for it, r in zip(its, res):
        lp = r.outputs[0].logprobs[0]
        p, top = {L: 0.0 for L in it["letters"]}, None
        for t, v in sorted(lp.items(), key=lambda kv: -kv[1].logprob):
            w = gen.tok.decode([t])
            top = top if top is not None else w
            if w.strip() in p:
                p[w.strip()] += math.exp(v.logprob)
        out.append((p, top))
    return out


def reads(gen, cell, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    path = Path(out) / f"reads_{cell}.jsonl"
    todo = [it for it in read_items() if it["key"] not in _done(path)]
    if not todo:
        return path
    got = [(fake_probs(it), "fake") for it in todo] if gen is None else letter_probs(gen, todo)
    with open(path, "a") as fh:
        for it, (p, top) in zip(todo, got):
            rec = {k: v for k, v in it.items() if k not in ("messages", "tools")}
            rec |= {"cell": cell, "probs": p, "mass": round(sum(p.values()), 6), "top": top}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[m38-eval] {cell}: {len(todo)} reads", flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m38 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    its = gate_items()
    fake = os.environ.get("V1_FAKE_GEN") == "1"
    gen = M27E.fake_gen if fake else R.VllmGen(a.model)
    for cell in a.cells:
        reads(None if fake else gen, cell, a.out)
        R.run(a.model, cell, [], a.out, items=its, gen=gen)


if __name__ == "__main__":
    main()
