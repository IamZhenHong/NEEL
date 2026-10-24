"""SC-M19 runner: P15b's two-mode runner (persona_affect.p15b_forced.evalrun.run) with each item's own forced-think
opener (items.OPENER: P15b's for the justice act, a neutral one for the self-report).

  off     V1's prompt and sampling, thinking off
  forced  phase 1: thinking-on prompt + the item's opener, V1's decoding, min_tokens 64, max_tokens 1,536, stopping at
          <tool_call>, </think> or <|im_end|> ("think_stop"; "length" = the cap, an undecided loop); phase 2: + the
          reasoning + "\\n</think>\\n\\n", V1's sampling for the answer. Only phase 2 is scored.

  python -m self_concept.m19_thinkmirror.evalrun --model DIR --cell NW --out DIR      (V1_FAKE_GEN=1: fake replies)
"""
import argparse
import os

from persona.exemplar.mini0 import DECODING
from persona_affect.p15b_forced import evalrun as P15B
from self_concept.m19_thinkmirror.items import items as all_items


class Gen(P15B.ForcedGen):
    """P15b's generator; forced entries open the think with the item's own opener."""

    def __call__(self, batch):
        from vllm import SamplingParams, TokensPrompt
        out = [None] * len(batch)
        off = [j for j, b in enumerate(batch) if b["mode"] == "off"]
        if off:
            for j, r in zip(off, P15B.P15R.ThinkGen.__call__(self, [batch[j] for j in off])):
                out[j] = r | {"think_stop": None}
        forced = [j for j, b in enumerate(batch) if b["mode"] == "forced"]
        if not forced:
            return out
        op = {j: self.tok(batch[j]["opener"], add_special_tokens=False)["input_ids"] for j in forced}
        p1 = [self._ids({**batch[j], "mode": "on"}) + op[j] for j in forced]
        sp1 = [SamplingParams(**DECODING, max_tokens=P15B.FORCE_MAX, min_tokens=P15B.FORCE_MIN,
                              seed=batch[j]["sampling"]["seed"], stop_token_ids=list(self.stop_ids)) for j in forced]
        res1 = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in p1], sp1, use_tqdm=False)
        p2, stops, thinks = [], [], []
        for j, p, r in zip(forced, p1, res1):
            o = r.outputs[0]
            ids = [t for t in o.token_ids if t not in self.stop_ids]
            stops.append(self.stop_ids.get(o.stop_reason) or ("length" if o.finish_reason == "length" else "eos"))
            thinks.append(batch[j]["opener"] + self.tok.decode(ids, skip_special_tokens=True))
            p2.append(p + ids + self.close_ids)
        res2 = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in p2],
                                 [SamplingParams(**batch[j]["sampling"]) for j in forced], use_tqdm=False)
        for j, think, stop, r in zip(forced, thinks, stops, res2):
            o = r.outputs[0]
            out[j] = {"text": o.text, "think": think, "closed": True, "finish": o.finish_reason, "think_stop": stop}
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m19_thinkmirror.evalrun", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gen = P15B.fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else Gen(a.model)
    P15B.run(a.model, a.cell, a.out, items=all_items(), gen=gen)


if __name__ == "__main__":
    main()
