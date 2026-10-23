"""PA-P15b runner: P15's runner with a "forced" thinking mode for the decision blocks.

P15 found that V1's fine-tuned 9B models (every LoRA stage trained with the template's empty think block) don't think
before a trained decision even with enable_thinking=True: they write the choose_option call straight into the open think
(98% of justice items in NW-SAYR). P15b forces a short think before the decision:

  off     P15's off mode (V1's prompt and sampling)
  forced  phase 1: the thinking-on prompt + FORCE_OPENER, V1's decoding, min_tokens FORCE_MIN, max_tokens FORCE_MAX,
          stopping at <tool_call>, </think> or <|im_end|> (the stop is recorded as "think_stop"); phase 2: the same ids +
          the phase-1 reasoning + "\\n</think>\\n\\n", then V1's sampling (max_tokens 400) for the decision. The record's
          "think" is the opener + the phase-1 reasoning; only phase 2's text is scored.
  nudge   V1's rule on decision blocks; the nudged reply is generated in off mode in both modes (the deliberation, if any,
          already happened in phase 1)

  python -m persona_affect.p15b_forced.evalrun --model DIR --cell NW-SAYR_s1 --out DIR     (V1_FAKE_GEN=1: fake replies)
"""
import argparse
import json
import os
import time
from pathlib import Path

from persona.exemplar.mini0 import DECODING, parse_option
from persona_affect.p15_stores import evalrun as P15R
from persona_affect.p15b_forced.items import items as all_items
from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook import evalset as E

FORCE_OPENER = "Let me think through this case and the options before I decide."
FORCE_MIN, FORCE_MAX = 64, 1536
STOPS = {"<tool_call>": "tool_call", "</think>": "think_end", "<|im_end|>": "im_end"}
CLOSE = "\n</think>\n\n"


def sampling(item, bump=0):
    return R.sampling(item, bump)


class ForcedGen(P15R.ThinkGen):
    """P15's generator; "forced" entries run the two phases, "off" entries go through P15's off path."""

    def __init__(self, model):
        super().__init__(model)
        self.stop_ids = {self.tok.convert_tokens_to_ids(t): name for t, name in STOPS.items()}
        self.opener_ids = self.tok(FORCE_OPENER, add_special_tokens=False)["input_ids"]
        self.close_ids = self.tok(CLOSE, add_special_tokens=False)["input_ids"]

    def __call__(self, batch):
        from vllm import SamplingParams, TokensPrompt
        out = [None] * len(batch)
        off = [j for j, b in enumerate(batch) if b["mode"] == "off"]
        if off:
            for j, r in zip(off, super().__call__([batch[j] for j in off])):
                out[j] = r | {"think_stop": None}
        forced = [j for j, b in enumerate(batch) if b["mode"] == "forced"]
        if not forced:
            return out
        p1 = [self._ids({**batch[j], "mode": "on"}) + self.opener_ids for j in forced]
        sp1 = [SamplingParams(**DECODING, max_tokens=FORCE_MAX, min_tokens=FORCE_MIN, seed=batch[j]["sampling"]["seed"],
                              stop_token_ids=list(self.stop_ids)) for j in forced]
        res1 = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in p1], sp1, use_tqdm=False)
        p2, stops, thinks = [], [], []
        for p, r in zip(p1, res1):
            o = r.outputs[0]
            ids = [t for t in o.token_ids if t not in self.stop_ids]
            stops.append(self.stop_ids.get(o.stop_reason) or ("length" if o.finish_reason == "length" else "eos"))
            thinks.append(FORCE_OPENER + self.tok.decode(ids, skip_special_tokens=True))
            p2.append(p + ids + self.close_ids)
        res2 = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in p2],
                                 [SamplingParams(**batch[j]["sampling"]) for j in forced], use_tqdm=False)
        for j, think, stop, r in zip(forced, thinks, stops, res2):
            o = r.outputs[0]
            out[j] = {"text": o.text, "think": think, "closed": True, "finish": o.finish_reason, "think_stop": stop}
        return out


def fake_gen(batch):
    base = P15R.fake_gen([{**b, "mode": "off"} for b in batch])
    return [r | ({"think": FORCE_OPENER + " I weigh the case.", "closed": True, "think_stop": "tool_call"}
                 if b["mode"] == "forced" else {"think_stop": None}) for b, r in zip(batch, base)]


def run(model, cell, out, *, items=None, gen=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{cell}.jsonl"
    done = R._done(path)
    todo = [it for it in (all_items() if items is None else items) if it["key"] not in done]
    if not todo:
        print(f"[p15b-eval] {cell}: nothing to do ({len(done)} records)", flush=True)
        return path
    load_s = 0.0
    if gen is None:
        if os.environ.get("V1_FAKE_GEN") == "1":
            gen = fake_gen
        else:
            gen = ForcedGen(model)
            load_s = gen.load_s
    t0 = time.time()
    for c0 in range(0, len(todo), R.CHUNK):
        chunk = todo[c0:c0 + R.CHUNK]
        first = gen([{**it, "sampling": sampling(it)} for it in chunk])
        ok = [it["block"] not in E.DECISION or parse_option(r["text"]) is not None for it, r in zip(chunk, first)]
        again = [j for j, o in enumerate(ok) if not o]
        second = {}
        if again:
            res = gen([{**chunk[j], "mode": "off", "sampling": sampling(chunk[j], 1),
                        "messages": chunk[j]["messages"] + [{"role": "assistant", "content": first[j]["text"]},
                                                            {"role": "user", "content": R.NUDGE}]} for j in again])
            second = dict(zip(again, res))
        with open(path, "a") as fh:
            for j, (it, r) in enumerate(zip(chunk, first)):
                s = second.get(j)
                text = r["text"] if s is None else r["text"] + R.NUDGE_SEP + s["text"]
                rec = {k: v for k, v in it.items() if k not in ("messages", "tools")}
                rec |= {"cell": cell, "text": text, "think": [r["think"]] if r["think"] else [],
                        "think_closed": r["closed"], "think_stop": r["think_stop"], "finish": (s or r)["finish"],
                        "nudged": s is not None, "first_try": ok[j]} | R.score(it, text)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stops = {}
        for r in first:
            if r["think_stop"]:
                stops[r["think_stop"]] = stops.get(r["think_stop"], 0) + 1
        print(f"[p15b-eval] {cell}: {min(c0 + R.CHUNK, len(todo))}/{len(todo)} generated, {len(again)} nudged, "
              f"think stops {stops} in this chunk ({time.time() - t0:.0f}s)", flush=True)
    (out / f"run_info_{cell}.json").write_text(json.dumps({"model": str(model), "load_s": load_s, "n": len(todo),
                                                           "gen_s": round(time.time() - t0, 1)}))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p15b_forced.evalrun", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run(a.model, a.cell, a.out)


if __name__ == "__main__":
    main()
