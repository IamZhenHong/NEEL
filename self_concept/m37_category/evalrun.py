"""SC-M37 eval: V1's runner (v1_value_hook/evalrun.run: chunks of 500, records appended to OUT/eval_{cell}.jsonl,
resumable) on items.items(), SC-M35's Qwen path: one merged model (or the base snapshot) per call, V1's own vLLM
generator (bf16, thinking off, mini 0's decoding, the item's seed).

The adapter-effect check (SC-M24 deviation 1: a pod silently applied no adapter): before any item, the loaded model's
greedy next-token log-probs of "A" and "B" on the first PROBE_N no-cue lottery prompts go to OUT/probe_{cell}.json.
BASE runs first; a trained cell is refused (exit 3, no item generated) if its largest |dlogprob| against BASE's probe
(--probe-ref) is below EFFECT_MIN. If the probe itself fails on either model, the cell runs with the merge check
(pod.sh) as its only adapter gate, and RESULT.md says so.

  /root/vllm/bin/python -m self_concept.m37_category.evalrun --model DIR --cells Q27-BASE_s0 --out DIR
  /root/vllm/bin/python -m self_concept.m37_category.evalrun --model MERGED --cells Q27-RISKY_s1 --out DIR \
      --probe-ref DIR/probe_Q27-BASE_s0.json
V1_FAKE_GEN=1 (dry runs) answers from fake_gen: replies hashed from the key alone, the same for every model and cell;
the probe is skipped.
"""
import argparse
import json
import os
from pathlib import Path

from persona_affect.common.util import seed
from persona_affect.v1_value_hook import evalrun as R
from self_concept.m37_category.items import items

PROBE_N, EFFECT_MIN, FLOOR = 8, 0.5, -30.0


def fake_gen(batch):
    out = []
    for b in batch:
        h = seed(b["key"])
        out.append(("AB"[h % 2] if b["block"] in ("lottery", "dom") else str(h % 101), "stop"))
    return out


def probe_items(its):
    return [it for it in its if it["block"] == "lottery" and it["cond"] == "none" and it["i"] == 0][:PROBE_N]


def probe(gen, its):
    """{key: {"A": logprob, "B": logprob}} of the greedy first token (top-20; a letter outside it gets FLOOR)."""
    from vllm import SamplingParams, TokensPrompt
    from persona_affect.v1_value_hook.fmt import prompt_ids
    ps = probe_items(its)
    res = gen.llm.generate([TokensPrompt(prompt_token_ids=prompt_ids(gen.tok, p["messages"], None)) for p in ps],
                           SamplingParams(max_tokens=1, temperature=0, logprobs=20), use_tqdm=False)
    out = {}
    for p, r in zip(ps, res):
        lp = r.outputs[0].logprobs[0]
        d = {}
        for t, v in lp.items():
            w = gen.tok.decode([t]).strip()
            if w in ("A", "B"):
                d[w] = max(d.get(w, FLOOR), v.logprob)
        out[p["key"]] = {k: d.get(k, FLOOR) for k in "AB"}
    return out


def effect(lp, ref):
    return max(abs(lp[k][x] - ref[k][x]) for k in lp if k in ref for x in "AB")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m37 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-ref", default="")
    a = ap.parse_args(argv)
    its = items()
    if os.environ.get("V1_FAKE_GEN") == "1":
        for cell in a.cells:
            R.run(a.model, cell, [], a.out, items=its, gen=fake_gen)
        return
    gen = R.VllmGen(a.model)
    try:
        lp = probe(gen, its)
    except Exception as e:  # noqa: BLE001 (the probe's own failure is recorded; the merge check still gates the arm)
        lp = None
        print(f"[m37-eval] probe failed ({type(e).__name__}: {e}); the merge check is the only adapter gate", flush=True)
    for cell in a.cells:
        rec = {"cell": cell, "model": str(a.model), "load_s": gen.load_s, "logprobs": lp}
        ref = None
        if a.probe_ref and Path(a.probe_ref).exists():
            ref = json.loads(Path(a.probe_ref).read_text())["logprobs"]
        if not cell.endswith("BASE_s0") and lp is not None and ref is not None:
            rec["effect_vs_base"] = round(effect(lp, ref), 4)
            Path(a.out, f"probe_{cell}.json").write_text(json.dumps(rec, indent=1))
            print(f"[m37-eval] {cell}: max |dlogprob(A/B)| vs BASE on {len(lp)} lottery prompts "
                  f"{rec['effect_vs_base']:.3f}", flush=True)
            if rec["effect_vs_base"] < EFFECT_MIN:
                print(f"[m37-eval] {cell}: below {EFFECT_MIN}: the adapter looks unapplied; no item run", flush=True)
                raise SystemExit(3)
        else:
            Path(a.out, f"probe_{cell}.json").write_text(json.dumps(rec, indent=1))
        R.run(a.model, cell, [], a.out, items=its, gen=gen)


if __name__ == "__main__":
    main()
