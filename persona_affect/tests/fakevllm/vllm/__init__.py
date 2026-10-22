"""A fake `vllm` for the PA-V8 CPU tests and the local dry run of persona_affect/v8_scale/pod.py. Never on a pod's path:
put persona_affect/tests/fakevllm on PYTHONPATH (or sys.path) explicitly.

  PYTHONPATH=persona_affect/tests/fakevllm uv run python -m persona_affect.v8_scale.pod run --stage calib --model q08 \
      --out <scratch dir>

It mimics the offline API surface the runner uses. LLM(**kw) takes the model as a keyword, as the runner passes it, and
raises ValueError for google/gemma-4-E4B-it when limit_mm_per_prompt has "audio" (so the runner's fallback runs).
get_tokenizer() gives a stub: apply_chat_template -> json.dumps of the conversation, a call -> {"input_ids": [1, 2, 3]},
convert_tokens_to_ids -> 7, chat_template "fake". generate() answers every prompt with a choose_option call (refer) in
the model's format, followed by that format's end-of-turn marker (which clean_output must cut), finish_reason "stop".
SamplingParams and TokensPrompt keep their kwargs. INITS (the kwargs of every LLM() call, failed ones included) and
GENERATES (one {"model", "prompts", "params"} per generate call) let the tests read what the runner asked for.
"""
import json
from types import SimpleNamespace

__version__ = "fake"

QWEN_CALL = ("<tool_call>\n<function=choose_option>\n<parameter=case_id>\nX\n</parameter>\n<parameter=option_id>\n"
             "refer\n</parameter>\n</function>\n</tool_call>")
GEMMA_CALL = ('<|tool_call>call:choose_option{case_id:<|"|>X<|"|>,option_id:<|"|>refer<|"|>,reason:<|"|>r<|"|>}'
              '<tool_call|>')
INITS = []
GENERATES = []


class SamplingParams:
    def __init__(self, **kw):
        self.kw = dict(kw)
        self.__dict__.update(kw)


def TokensPrompt(**kw):   # noqa: N802 (vLLM's name; a TypedDict there, so a dict here)
    return dict(kw)


class _Tokenizer:
    chat_template = "fake"

    def apply_chat_template(self, conversation, **kw):
        return json.dumps(conversation)

    def __call__(self, text, **kw):
        return {"input_ids": [1, 2, 3]}

    def convert_tokens_to_ids(self, token):
        return 7


class LLM:
    def __init__(self, **kw):
        INITS.append(kw)
        if kw.get("model") == "google/gemma-4-E4B-it" and "audio" in (kw.get("limit_mm_per_prompt") or {}):
            raise ValueError("fake: google/gemma-4-E4B-it rejects an audio limit")
        self.model = kw["model"]

    def get_tokenizer(self):
        return _Tokenizer()

    def generate(self, prompts, params, use_tqdm=False):
        params = list(params) if isinstance(params, (list, tuple)) else [params] * len(prompts)
        if len(params) != len(prompts):
            raise ValueError(f"fake: {len(prompts)} prompts but {len(params)} sampling params")
        GENERATES.append({"model": self.model, "prompts": list(prompts), "params": params})
        gemma = "gemma" in self.model.lower()
        text = GEMMA_CALL + "<turn|>" if gemma else QWEN_CALL + "<|im_end|>"
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text, finish_reason="stop", token_ids=[7])])
                for _ in prompts]
