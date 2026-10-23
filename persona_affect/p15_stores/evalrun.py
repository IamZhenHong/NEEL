"""PA-P15 runner: V1's runner (decoding, nudge, score; persona_affect/v1_value_hook/evalrun.py) with a thinking mode per
item.

  off    V1's prompt (the template's empty think block) and V1's sampling (max_tokens 400; 16 for gen)
  on     enable_thinking=True (the prompt ends in "<think>\\n"), V1's decoding, max_tokens 3072. The reply is split at
         the </think> token: the reasoning goes to "think", only the text after it is kept as "text" and scored. A reply
         whose think never closes gets text "" (unparsed), "think_closed": false, and is not nudged.
  nudge  both modes, decision blocks (V1's rule): the reply's text goes back as the assistant turn, then V1's nudge; in
         on mode the nudged reply thinks again (its reasoning is the second entry of "think").
max_model_len 8192. A record = the item without messages and tools + cell, mode, text, think, think_closed, finish,
nudged, first_try + V1's score() of the text.

  python -m persona_affect.p15_stores.evalrun --model DIR --cell NW-SAYR_s1 --out DIR     (V1_FAKE_GEN=1: fake replies)
"""
import argparse
import json
import os
import time
from pathlib import Path

from persona.exemplar.mini0 import DECODING, parse_option
from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook import evalset as E
from persona_affect.p11_saydo import evalrun as P11R
from persona_affect.p15_stores.items import items as all_items

ON_MAX_TOKENS = 3072
THINK_END = "</think>"


def sampling(item, bump=0):
    if item["mode"] == "off":
        return R.sampling(item, bump)
    return {**DECODING, "max_tokens": ON_MAX_TOKENS, "seed": item["seed"] + bump}


def split_think(text):
    """A thinking-on reply -> (think, text after </think>, closed). The prompt already holds "<think>\\n"."""
    if THINK_END not in text:
        return text, "", False
    think, post = text.split(THINK_END, 1)
    return think.replace("<think>", "").strip(), post.strip(), True


class ThinkGen:
    """vLLM on one model; the prompt's thinking mode follows each entry's "mode"."""

    def __init__(self, model):
        from vllm import LLM
        t0 = time.time()
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85, seed=0,
                       enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
        self.tok = self.llm.get_tokenizer()
        self.end_id = self.tok.convert_tokens_to_ids(THINK_END)
        self.load_s = round(time.time() - t0, 1)

    def _ids(self, b):
        p = self.tok.apply_chat_template(b["messages"], tools=b["tools"], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=b["mode"] == "on")
        return self.tok(p, add_special_tokens=False)["input_ids"]

    def __call__(self, batch):
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=self._ids(b)) for b in batch],
                                [SamplingParams(**b["sampling"]) for b in batch], use_tqdm=False)
        out = []
        for b, r in zip(batch, res):
            o = r.outputs[0]
            if b["mode"] == "off":
                out.append({"text": o.text, "think": None, "closed": None, "finish": o.finish_reason})
                continue
            ids = list(o.token_ids)
            if self.end_id in ids:   # split on the token itself, so a decoded "</think>" inside text can't fool it
                k = len(ids) - 1 - ids[::-1].index(self.end_id)
                think = self.tok.decode(ids[:k], skip_special_tokens=True).replace("<think>", "").strip()
                post = self.tok.decode(ids[k + 1:], skip_special_tokens=True).strip()
                out.append({"text": post, "think": think, "closed": True, "finish": o.finish_reason})
            else:
                out.append({"text": "", "think": o.text, "closed": False, "finish": o.finish_reason})
        return out


def fake_gen(batch):
    """P11's fake replies; on mode wraps each in a short closed think, split by split_think."""
    base = P11R.fake_gen(batch)
    out = []
    for b, (t, f) in zip(batch, base):
        if b["mode"] == "off":
            out.append({"text": t, "think": None, "closed": None, "finish": f})
        else:
            think, post, closed = split_think(f"<think>\nI weigh the case.\n</think>\n\n{t}")
            out.append({"text": post, "think": think, "closed": closed, "finish": f})
    return out


def run(model, cell, out, *, items=None, gen=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{cell}.jsonl"
    done = R._done(path)
    todo = [it for it in (all_items() if items is None else items) if it["key"] not in done]
    if not todo:
        print(f"[p15-eval] {cell}: nothing to do ({len(done)} records)", flush=True)
        return path
    load_s = 0.0
    if gen is None:
        if os.environ.get("V1_FAKE_GEN") == "1":
            gen = fake_gen
        else:
            gen = ThinkGen(model)
            load_s = gen.load_s
    t0 = time.time()
    for c0 in range(0, len(todo), R.CHUNK):
        chunk = todo[c0:c0 + R.CHUNK]
        first = gen([{**it, "sampling": sampling(it)} for it in chunk])
        ok = [it["block"] not in E.DECISION or parse_option(r["text"]) is not None for it, r in zip(chunk, first)]
        again = [j for j, (r, o) in enumerate(zip(first, ok)) if not o and r["closed"] is not False]
        second = {}
        if again:
            res = gen([{**chunk[j], "sampling": sampling(chunk[j], 1),
                        "messages": chunk[j]["messages"] + [{"role": "assistant", "content": first[j]["text"]},
                                                            {"role": "user", "content": R.NUDGE}]} for j in again])
            second = dict(zip(again, res))
        with open(path, "a") as fh:
            for j, (it, r) in enumerate(zip(chunk, first)):
                s = second.get(j)
                text = r["text"] if s is None else r["text"] + R.NUDGE_SEP + s["text"]
                closed = r["closed"] if s is None or r["closed"] is None else (r["closed"] and s["closed"])
                rec = {k: v for k, v in it.items() if k not in ("messages", "tools")}
                rec |= {"cell": cell, "text": text, "think": [x["think"] for x in (r, s) if x is not None],
                        "think_closed": closed, "finish": (s or r)["finish"], "nudged": s is not None,
                        "first_try": ok[j]} | R.score(it, text)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_unclosed = sum(1 for r in first if r["closed"] is False)
        print(f"[p15-eval] {cell}: {min(c0 + R.CHUNK, len(todo))}/{len(todo)} generated, {len(again)} nudged, "
              f"{n_unclosed} unclosed thinks in this chunk ({time.time() - t0:.0f}s)", flush=True)
    gpu = "cpu"
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 (a fake run without torch)
        pass
    (out / f"run_info_{cell}.json").write_text(json.dumps({"model": str(model), "gpu": gpu, "load_s": load_s,
                                                           "n": len(todo), "gen_s": round(time.time() - t0, 1)}))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p15_stores.evalrun", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run(a.model, a.cell, a.out)


if __name__ == "__main__":
    main()
