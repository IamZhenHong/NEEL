"""A fake `vllm` for PA-A14's CPU tests and the local dry run of persona_affect/a14_desperation/pod.sh (plan Task 6
Step 3). Never on a pod's path: put persona_affect/a14_desperation/fakevllm on PYTHONPATH explicitly.

    SKIP_SETUP=1 PY=... TEST_PY=... SANDBOX_USER= PYTHONPATH=persona_affect/a14_desperation/fakevllm M=q27 \\
        JOBS=persona_affect/a14_desperation/jobs/pilot_q27.jsonl DIR=dryrun LIMIT_TASKS=2 bash persona_affect/a14_desperation/pod.sh

It mimics the offline API that a14.VllmBackend uses. The tokenizer works on characters: token ids are code points,
apply_chat_template joins one `<role>content` line per message (plus "\\n<assistant>" with the generation prompt), and
chat_template is "fake". LLM.generate answers each prompt by its sampling seed % 3:

    0  a code block `def <entry>(*a): return None` (fails every test)
    1  a code block that redefines `check` (passes any test: the override cheat)
    2  prose only (no code block)

where <entry> is read from the prompt's "# Use check(<entry>) to run tests." line. A reply longer than max_tokens
characters is cut there with finish_reason "length", else "stop". INITS (the kwargs of every LLM() call, failed ones
included) and GENERATES ({"prompts", "params"} per generate call) let tests read what the runner asked for; set
REJECT_VIDEO = True to make LLM() refuse a "video" limit, as some multimodal configs do, so the image-only retry runs.
"""
import re
from types import SimpleNamespace

__version__ = "fake"
INITS = []
GENERATES = []
REJECT_VIDEO = False
REPLIES = ("Here is my solution.\n```python\ndef {entry}(*a):\n    return None\n```",
           "```python\ndef check(candidate):\n    pass\n```",
           "The tests contradict each other, so no function can pass them all.")
_ENTRY = re.compile(r"# Use check\((\w+)\) to run tests\.")


class SamplingParams:
    def __init__(self, **kw):
        self.kw = dict(kw)
        self.__dict__.update(kw)


def TokensPrompt(**kw):   # noqa: N802 (vLLM's name; a TypedDict there, so a dict here)
    return dict(kw)


class _Tokenizer:
    chat_template = "fake"

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kw):
        text = "\n".join(f"<{m['role']}>{m['content']}" for m in conversation)
        return text + "\n<assistant>" if add_generation_prompt else text

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, **kw):
        return "".join(map(chr, ids))


class LLM:
    def __init__(self, **kw):
        INITS.append(kw)
        if REJECT_VIDEO and "video" in (kw.get("limit_mm_per_prompt") or {}):
            raise ValueError("fake: this model rejects a video limit")
        self.kw = kw

    def get_tokenizer(self):
        return _Tokenizer()

    def generate(self, prompts, params, use_tqdm=False, **kw):
        GENERATES.append({"prompts": prompts, "params": params})
        out = []
        for prompt, p in zip(prompts, params):
            entries = _ENTRY.findall("".join(map(chr, prompt["prompt_token_ids"])))
            text = REPLIES[p.seed % 3].format(entry=entries[-1] if entries else "f")
            finish = "stop"
            if len(text) > p.max_tokens:
                text, finish = text[:p.max_tokens], "length"
            out.append(SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[ord(c) for c in text],
                                                                finish_reason=finish)]))
        return out
