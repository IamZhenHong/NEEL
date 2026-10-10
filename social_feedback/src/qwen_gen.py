"""Qwen3-8B generation with vLLM (DESIGN.md §3). Runs on the pod, in the vLLM venv (pod/setup_vllm.sh).

Two-stage generation: stage 1 thinks until </think> or the thinking cap; at the cap, "\n</think>\n\n" is appended and
the trial is flagged cap_hit. Stage 2 generates the answer (answer cap 1,024). The deliverable is the stage-2 text.
Prompts are built as token ids, so stage 2 continues exactly where stage 1 stopped.
"""
import hashlib
import json

from .common import QWEN

DECODING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0)
ANSWER_CAP = 1024
MAX_MODEL_LEN = 32768


class Qwen:
    def __init__(self, revision=None, gpu_mem=0.90):
        from huggingface_hub import HfApi
        from vllm import LLM
        import vllm
        self.revision = revision or HfApi().model_info(QWEN).sha
        self.llm = LLM(model=QWEN, revision=self.revision, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
                       enable_prefix_caching=True, gpu_memory_utilization=gpu_mem, seed=0)
        self.tok = self.llm.get_tokenizer()
        self.end_think = self.tok.convert_tokens_to_ids("</think>")
        self.close_ids = self.tok("\n</think>\n\n", add_special_tokens=False)["input_ids"]
        tmpl = self.tok.chat_template if isinstance(self.tok.chat_template, str) else json.dumps(self.tok.chat_template)
        self.settings = {"model": QWEN, "revision": self.revision, "vllm": vllm.__version__, "dtype": "bfloat16",
                         "max_model_len": MAX_MODEL_LEN, "decoding": DECODING, "answer_cap": ANSWER_CAP,
                         "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest()}

    def render(self, messages, thinking=True):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=thinking)

    def generate(self, jobs, thinking_cap=None):
        """jobs: list of {"key", "messages", "seed"}. thinking_cap None = limited only by context.
        Returns {key: {"answer", "thinking_tokens", "answer_tokens", "cap_hit", "answer_truncated", "prompt_tokens",
        "thinking"}}."""
        from vllm import SamplingParams, TokensPrompt
        prompts, caps = [], []
        for j in jobs:
            ids = self.tok(self.render(j["messages"]), add_special_tokens=False)["input_ids"]
            room = MAX_MODEL_LEN - len(ids) - ANSWER_CAP - len(self.close_ids) - 8
            if room < 256:
                raise ValueError(f"{j['key']}: prompt {len(ids)} tokens leaves no room")
            prompts.append(ids)
            caps.append(min(thinking_cap, room) if thinking_cap else room)
        s1 = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                               [SamplingParams(**DECODING, max_tokens=c, seed=j["seed"],
                                               stop_token_ids=[self.end_think], skip_special_tokens=False)
                                for j, c in zip(jobs, caps)])
        stage2, meta = [], []
        for j, p, out in zip(jobs, prompts, s1):
            o = out.outputs[0]
            think_ids = list(o.token_ids)
            cap_hit = o.finish_reason == "length"
            if cap_hit:
                think_ids = think_ids + self.close_ids
            elif not think_ids or think_ids[-1] != self.end_think:
                think_ids = think_ids + [self.end_think]  # stop token not echoed in token_ids by this vLLM version
            if o.finish_reason not in ("length", "stop"):
                raise RuntimeError(f"{j['key']}: unexpected stage-1 finish {o.finish_reason}")
            stage2.append(p + think_ids)
            meta.append({"cap_hit": cap_hit, "thinking_tokens": len(o.token_ids), "stage1_stop": str(o.stop_reason),
                         "thinking": self.tok.decode(think_ids, skip_special_tokens=False)})
        s2 = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in stage2],
                               [SamplingParams(**DECODING, max_tokens=ANSWER_CAP, seed=j["seed"] + 1)
                                for j in jobs])
        res = {}
        for j, p, m, out in zip(jobs, prompts, meta, s2):
            o = out.outputs[0]
            res[j["key"]] = {**m, "answer": o.text.strip(), "answer_tokens": len(o.token_ids),
                             "answer_truncated": o.finish_reason == "length", "prompt_tokens": len(p)}
        return res

    def yes_no(self, jobs):
        """Thinking off; next-token P(Yes) renormalised over {Yes, No}, plus the raw probabilities.
        jobs: list of {"key", "messages"}."""
        import math
        from vllm import SamplingParams
        texts = [self.render(j["messages"], thinking=False) for j in jobs]
        outs = self.llm.generate(texts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
        yes_ids = {self.tok.convert_tokens_to_ids(t) for t in ("Yes", "ĠYes")}
        no_ids = {self.tok.convert_tokens_to_ids(t) for t in ("No", "ĠNo")}
        res = {}
        for j, out in zip(jobs, outs):
            lp = out.outputs[0].logprobs[0]
            py = sum(math.exp(v.logprob) for k, v in lp.items() if k in yes_ids)
            pn = sum(math.exp(v.logprob) for k, v in lp.items() if k in no_ids)
            res[j["key"]] = {"p_yes": py / (py + pn) if py + pn > 0 else None, "raw_yes": py, "raw_no": pn,
                             "top": self.tok.decode([next(iter(lp))])}
        return res
