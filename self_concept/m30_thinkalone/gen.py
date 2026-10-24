"""SC-M30 runner (DESIGN.md section 4): one untouched model, every item of jobs/items.jsonl.gz in three thinking modes,
all from the same item seed.

  off    the chat template with thinking off; V1's sampling for the answer (max 400 tokens)
  on     phase 1: the template with thinking on (Gemma 4: the thought channel opened if the template left it closed),
         the model's own reasoning with V1's decoding, up to THINK_MAX tokens, stopping at the end of the think, at the
         start of a tool call or at the end of the turn ("think_stop"; "length" = the cap); phase 2: + the reasoning
         (trailing whitespace dropped) + the close ("\\n</think>\\n\\n" / "\\n<channel|>"), then V1's sampling
  brief  phase 2 only: the same phase-1 prompt + the first BRIEF_K tokens of on's reasoning + the close, same seed.
         Where on's reasoning is no longer than BRIEF_K tokens, brief's prompt is on's prompt
  nudge  a justice answer with no choose_option call gets V1's one nudge, generated thinking off with seed + 1, in
         every mode (the deliberation, if any, already happened)

Items run in a fixed shuffled order (random.Random(0)) in chunks, all three modes of an item in the same chunk, so a
deadline stop leaves balanced, paired data. A chunk is not started within 8 minutes of T0 + MAXMIN (env). Records go
to OUT/eval_{cell}.jsonl, one per item and mode (key SC-M30|block|cond|mode|tid|i); items already complete in the file
are skipped.

  /root/vllm/bin/python -m self_concept.m30_thinkalone.gen --cell Q27 --model DIR --out DIR
  uv run python -m self_concept.m30_thinkalone.gen --cell G26 --out DIR --fake [--limit N]     (tests: fake replies)
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from persona.exemplar.mini0 import DECODING, parse_option
from persona_affect.v1c_selfreport.analyze import _letter
from self_concept.m30_thinkalone import items as I

MODELS = {"Q27": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", "qwen"),
          "G26": ("google/gemma-4-26b-a4b-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52", "gemma4")}
BRIEF_K, THINK_MAX, ANS_MAX, MAX_LEN = 128, 3072, 400, 6144
CHUNK = 288
NUDGE = "Please record your decision now with choose_option."
NUDGE_SEP = "\n\n[nudge]\n\n"
FMT = {"qwen": {"open": "", "close": "\n</think>\n\n", "skip": True,
                "stops": {"</think>": "think_end", "<tool_call>": "tool_call", "<|im_end|>": "turn_end"}},
       "gemma4": {"open": "<|channel>thought\n", "close": "\n<channel|>", "skip": False,
                  "stops": {"<channel|>": "think_end", "<|tool_call>": "tool_call", "<turn|>": "turn_end",
                            "<eos>": "turn_end", "<|tool_response>": "turn_end"}}}
GEMMA_DROP = ("<turn|>", "<eos>", "<|tool_response>")
T0 = time.time()


def log(msg):
    print(f"[sc-m30 {time.time() - T0:6.0f}s] {msg}", flush=True)


def clean(text, fmt):
    if fmt == "gemma4":
        for s in GEMMA_DROP:
            text = text.replace(s, "")
    return text


def visible(text, fmt):
    """The answer without any thought the model opened itself in it (Gemma: <|channel>...<channel|>)."""
    if fmt == "gemma4" and "<channel|>" in text:
        return text.split("<channel|>")[-1]
    if fmt == "qwen" and "</think>" in text:
        return text.split("</think>")[-1]
    return text


def gemma_calls(text):
    from vigilante.src.sim import parse_calls_gemma
    return parse_calls_gemma(text or "")[1]


def option_of(text, fmt):
    if fmt == "qwen":
        return parse_option(text)
    for name, args in gemma_calls(text):
        if name == "choose_option" and args.get("option_id"):
            return str(args["option_id"]).strip().lower()
    return None


def reason_of(text, fmt):
    if fmt == "qwen":
        m = re.search(r"<parameter=reason>\s*(.*?)\s*</parameter>", text, re.S)
        return m.group(1).strip() if m else None
    for name, args in gemma_calls(text):
        if name == "choose_option" and args.get("reason") is not None:
            return str(args["reason"]).strip()
    return None


def score(it, text, fmt):
    last = visible(text.split(NUDGE_SEP)[-1], fmt)
    if it["block"] == "just":
        opt = option_of(last, fmt)
        return {"option": opt, "role": it["amap"].get(opt), "reason": reason_of(last, fmt)}
    bare = re.sub(r"<[^>]*>", "", last)
    letter = _letter(bare, it["amap"])
    return {"letter": letter, "role": it["amap"].get(letter) if letter else None}


class VBackend:
    """vLLM on one model."""

    def __init__(self, path):
        from vllm import LLM
        t = time.time()
        self.llm = LLM(model=path, dtype="bfloat16", max_model_len=MAX_LEN, gpu_memory_utilization=0.90, seed=0,
                       enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
        self.tok = self.llm.get_tokenizer()
        self.load_s = round(time.time() - t, 1)

    def generate(self, prompts, params):
        from vllm import SamplingParams, TokensPrompt
        sps = [SamplingParams(**DECODING, max_tokens=p["max_tokens"], seed=p["seed"],
                              stop_token_ids=p.get("stop_ids") or None, skip_special_tokens=p["skip"]) for p in params]
        res = self.llm.generate([TokensPrompt(prompt_token_ids=x) for x in prompts], sps, use_tqdm=False)
        return [{"ids": list(r.outputs[0].token_ids), "text": r.outputs[0].text, "finish": r.outputs[0].finish_reason,
                 "stop": r.outputs[0].stop_reason} for r in res]


class FakeBackend:
    """Tests only: the real tokenizer, made-up replies. Thinks of 20-400 tokens; answers follow PLANT[mode] (the share
    choosing penalise, or letter A) so the analysis can be checked end to end."""
    PLANT = {"off": 0.4, "on": 0.7, "brief": 0.55}

    def __init__(self, hf_id, fmt):
        from transformers import AutoTokenizer
        self.tok, self.fmt, self.load_s = AutoTokenizer.from_pretrained(hf_id), fmt, 0.0
        self.close_id = self.tok.convert_tokens_to_ids("</think>" if fmt == "qwen" else "<channel|>")

    def _call(self, case, opt):
        if self.fmt == "qwen":
            return (f"<tool_call>\n<function=choose_option>\n<parameter=case_id>\n{case}\n</parameter>\n"
                    f"<parameter=option_id>\n{opt}\n</parameter>\n<parameter=reason>\nThe record shows it.\n"
                    f"</parameter>\n</function>\n</tool_call>")
        q = '<|"|>'
        return (f"<|tool_call>call:choose_option{{case_id:{q}{case}{q},option_id:{q}{opt}{q},reason:{q}The record "
                f"shows it.{q}}}<tool_call|>")

    def generate(self, prompts, params):
        out = []
        for p in params:
            rng = random.Random(p["seed"] * 7 + len(p["tag"]))
            kind, mode, block = p["tag"]
            if kind == "think":
                n = rng.randint(20, 400)
                ids = self.tok(" ".join(["weigh the case and the options"] * 80), add_special_tokens=False)["input_ids"]
                ids = (ids * 4)[:n]
                if n > p["max_tokens"]:
                    out.append({"ids": ids[:p["max_tokens"]], "text": "", "finish": "length", "stop": None})
                else:
                    out.append({"ids": ids + [self.close_id], "text": "", "finish": "stop", "stop": self.close_id})
                continue
            if kind == "nudge" or (block == "just" and rng.random() < 0.03):   # a few unparsed first answers
                text = self._call("X", "refer") if kind == "nudge" else "I will decide."
            elif block == "just":
                text = self._call("X", "penalise" if rng.random() < self.PLANT[mode] else "refer")
            else:
                text = "A" if rng.random() < 1 - self.PLANT[mode] else "B"
            out.append({"ids": self.tok(text, add_special_tokens=False)["input_ids"], "text": text, "finish": "stop",
                        "stop": None})
        return out


class Runner:
    def __init__(self, cell, backend):
        self.cell, self.be = cell, backend
        self.fmt = MODELS[cell][2]
        f = FMT[self.fmt]
        self.tok = backend.tok
        self.stop_names = {self.tok.convert_tokens_to_ids(t): n for t, n in f["stops"].items()}
        assert all(isinstance(k, int) and k >= 0 for k in self.stop_names), self.stop_names
        self.close_ids = self.tok(f["close"], add_special_tokens=False)["input_ids"]

    def prompt_ids(self, it, thinking, extra=()):
        msgs = list(it["messages"]) + list(extra)
        if self.fmt == "gemma4":
            from persona.history.runner import chat
            msgs = chat(msgs, "gemma4")
        s = self.tok.apply_chat_template(msgs, tools=it["tools"], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking)
        op = FMT[self.fmt]["open"]
        if thinking and op and not s.endswith(op):
            s += op
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def _rstrip(self, ids):
        """Drop trailing whitespace-only tokens, so the close renders as the template renders a closed think."""
        ids = list(ids)
        while ids and not self.tok.decode([ids[-1]]).strip():
            ids.pop()
        return ids

    def _ans(self, it, mode, kind="answer"):
        return {"max_tokens": ANS_MAX, "seed": it["seed"] + (kind == "nudge"), "skip": FMT[self.fmt]["skip"],
                "tag": (kind, mode, it["block"])}

    def chunk(self, its):
        off_p = [self.prompt_ids(it, False) for it in its]
        on_p = [self.prompt_ids(it, True) for it in its]
        th = self.be.generate(on_p, [{"max_tokens": THINK_MAX, "seed": it["seed"], "stop_ids": list(self.stop_names),
                                      "skip": True, "tag": ("think", "on", it["block"])} for it in its])
        thinks, stops = [], []
        for r in th:
            thinks.append([t for t in r["ids"] if t not in self.stop_names])
            stops.append(self.stop_names.get(r["stop"]) or ("length" if r["finish"] == "length" else "eos"))
        prompts, params, meta = [], [], []
        for j, it in enumerate(its):
            prompts.append(off_p[j])
            params.append(self._ans(it, "off"))
            meta.append((j, "off"))
            full = self._rstrip(thinks[j])
            prompts.append(on_p[j] + full + self.close_ids)
            params.append(self._ans(it, "on"))
            meta.append((j, "on"))
            cut = self._rstrip(thinks[j][:BRIEF_K]) if len(thinks[j]) > BRIEF_K else full
            prompts.append(on_p[j] + cut + self.close_ids)
            params.append(self._ans(it, "brief"))
            meta.append((j, "brief"))
        res = self.be.generate(prompts, params)
        texts = [clean(r["text"], self.fmt) for r in res]
        first = [its[j]["block"] != "just" or option_of(visible(t, self.fmt), self.fmt) is not None
                 for (j, _), t in zip(meta, texts)]
        again = [k for k, ok in enumerate(first) if not ok]
        if again:
            extra = lambda k: [{"role": "assistant", "content": visible(texts[k], self.fmt)},  # noqa: E731
                               {"role": "user", "content": NUDGE}]
            res2 = self.be.generate([self.prompt_ids(its[meta[k][0]], False, extra(k)) for k in again],
                                    [self._ans(its[meta[k][0]], meta[k][1], "nudge") for k in again])
            for k, r in zip(again, res2):
                texts[k] = texts[k] + NUDGE_SEP + clean(r["text"], self.fmt)
        recs = []
        for k, ((j, mode), text) in enumerate(zip(meta, texts)):
            it = its[j]
            n_think = len(thinks[j])
            think = None
            if mode != "off":
                ids = thinks[j] if mode == "on" or n_think <= BRIEF_K else thinks[j][:BRIEF_K]
                think = self.tok.decode(ids, skip_special_tokens=True)
            rec = {x: it[x] for x in ("block", "cond", "tid", "i", "rot", "pen_pos", "seed", "amap")}
            rec |= {"key": f"{I.EXP}|{it['block']}|{it['cond']}|{mode}|{it['tid']}|{it['i']}", "base": it["key"],
                    "cell": self.cell, "mode": mode, "text": text, "think": think,
                    "think_tokens": None if mode == "off" else (n_think if mode == "on" else min(n_think, BRIEF_K)),
                    "think_stop": None if mode == "off" else stops[j], "cut": mode == "brief" and n_think > BRIEF_K,
                    "nudged": k in again, "first_try": first[k], "finish": res[k]["finish"],
                    "self_thought": ("<|channel>" in res[k]["text"]) if self.fmt == "gemma4" else
                                    ("</think>" in res[k]["text"])} | score(it, text, self.fmt)
            recs.append(rec)
        return recs, sum(s == "length" for s in stops), len(again)


def done_keys(path):
    if not path.exists():
        return set()
    n = {}
    for line in path.read_text().splitlines():
        if line.strip():
            b = json.loads(line)["base"]
            n[b] = n.get(b, 0) + 1
    return {b for b, c in n.items() if c >= len(I.MODES)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m30_thinkalone.gen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, choices=sorted(MODELS))
    ap.add_argument("--model", help="local snapshot directory (default: the pinned HF id)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N items of the shuffled order (tests)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{a.cell}.jsonl"
    hf_id, _, fmt = MODELS[a.cell]
    maxmin = float(os.environ.get("MAXMIN", "0") or 0)
    t0 = float(os.environ.get("T0", "0") or 0) or time.time()
    deadline = t0 + (maxmin - 8) * 60 if maxmin else float("inf")
    its = I.load()
    random.Random(0).shuffle(its)
    if a.limit:
        its = its[:a.limit]
    done = done_keys(path)
    todo = [it for it in its if it["key"] not in done]
    log(f"{a.cell}: {len(todo)} of {len(its)} items to run ({len(done)} done)")
    if not todo:
        return
    be = FakeBackend(hf_id, fmt) if a.fake else VBackend(a.model or hf_id)
    log(f"{a.cell}: loaded in {be.load_s}s")
    rn = Runner(a.cell, be)
    info = {"cell": a.cell, "model": hf_id, "load_s": be.load_s, "n_items": 0, "capped": 0, "nudged": 0,
            "brief_k": BRIEF_K, "think_max": THINK_MAX, "chunks": []}
    t = time.time()
    for c0 in range(0, len(todo), CHUNK):
        if time.time() > deadline:
            log(f"{a.cell}: deadline; stopping with {c0} of {len(todo)} items run")
            info["stopped_early"] = {"at_item": c0, "of": len(todo)}
            break
        tc = time.time()
        recs, capped, nudged = rn.chunk(todo[c0:c0 + CHUNK])
        with open(path, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        info["n_items"] += len(recs) // len(I.MODES)
        info["capped"] += capped
        info["nudged"] += nudged
        info["chunks"].append(round(time.time() - tc, 1))
        log(f"{a.cell}: {min(c0 + CHUNK, len(todo))}/{len(todo)} items, {capped} thinks capped, {nudged} nudged in "
            f"this chunk ({time.time() - tc:.0f}s)")
    info["gen_s"] = round(time.time() - t, 1)
    (out / f"run_info_{a.cell}.json").write_text(json.dumps(info, indent=1))
    log(f"{a.cell}: done")


if __name__ == "__main__":
    main()
