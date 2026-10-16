"""A fake `vllm` for CPU tests of vibes/batteries/run_vllm.py only (put tests/fakevllm on PYTHONPATH explicitly).
It mimics the offline API surface run_vllm uses (LLM, SamplingParams, TokensPrompt, outputs with text,
finish_reason, token_ids and top-k logprobs) on top of an HF model. Never on a pod's path."""
import math
from types import SimpleNamespace

import torch

__version__ = "fake"


class SamplingParams:
    def __init__(self, temperature=1.0, top_p=1.0, top_k=0, max_tokens=16, seed=None, logprobs=None, **kw):
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.max_tokens, self.seed, self.logprobs = max_tokens, seed, logprobs


def TokensPrompt(prompt_token_ids):
    return {"prompt_token_ids": prompt_token_ids}


class LLM:
    def __init__(self, model, **kw):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if "limit_mm_per_prompt" in kw and "video" in (kw["limit_mm_per_prompt"] or {}):
            raise ValueError("fake: reject the video key once, as some configs do")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()

    def get_tokenizer(self):
        return self.tok

    @torch.no_grad()
    def generate(self, prompts, params, use_tqdm=False, lora_request=None):
        if not isinstance(params, list):
            params = [params] * len(prompts)
        outs = []
        for p, sp in zip(prompts, params):
            ids = torch.tensor([p["prompt_token_ids"]])
            if sp.logprobs:
                lp = torch.log_softmax(self.model(input_ids=ids).logits[0, -1].float(), -1)
                top = torch.topk(lp, sp.logprobs)
                d = {int(t): SimpleNamespace(logprob=float(v)) for v, t in zip(top.values, top.indices)}
                tid = int(top.indices[0])
                outs.append(SimpleNamespace(outputs=[SimpleNamespace(text=self.tok.decode([tid]), finish_reason="length",
                                                                     token_ids=[tid], logprobs=[d])]))
                continue
            if sp.seed is not None:
                torch.manual_seed(sp.seed)
            kw = dict(do_sample=sp.temperature > 0, max_new_tokens=sp.max_tokens)
            if sp.temperature > 0:
                kw.update(temperature=sp.temperature, top_p=sp.top_p, top_k=sp.top_k if sp.top_k > 0 else None)
            g = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    pad_token_id=self.tok.pad_token_id, **kw)[0, ids.shape[1]:].tolist()
            fin = "stop" if g and g[-1] == self.tok.eos_token_id else "length"
            outs.append(SimpleNamespace(outputs=[SimpleNamespace(text=self.tok.decode(g, skip_special_tokens=True),
                                                                 finish_reason=fin, token_ids=g, logprobs=None)]))
        return outs
