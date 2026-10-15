"""Neutral messages -> each model's raw chat messages (the form the vigilante engine records), and token counts.

Formats: 'qwen' (Qwen3.5/3.6 XML tool calls), 'gemma4' (Gemma 4 call syntax; vigilante.src.templates.to_template turns
it into structured tool_calls for the template), 'qwen3' (Hermes JSON; only for the Qwen3-0.6B smoke test).
Scripted assistant turns are written exactly as the model's own tool calls parse back (tests check the round trip).
"""
from __future__ import annotations

import json

from vigilante.src.templates import to_template

Q = '<|"|>'


def _qwen_val(x) -> str:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return str(int(x)) if float(x).is_integer() else str(x)
    return str(x)


def call_text(calls: list[tuple[str, dict]], fmt: str) -> str:
    if fmt == "qwen":
        return "\n".join("<tool_call>\n<function=" + n + ">\n" +
                         "".join(f"<parameter={k}>\n{_qwen_val(v)}\n</parameter>\n" for k, v in a.items()) +
                         "</function>\n</tool_call>" for n, a in calls)
    if fmt == "gemma4":
        def val(x):
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                return str(int(x)) if float(x).is_integer() else str(x)
            return f"{Q}{x}{Q}"
        return "".join("<|tool_call>call:" + n + "{" + ",".join(f"{k}:{val(v)}" for k, v in a.items()) + "}<tool_call|>"
                       for n, a in calls)
    if fmt == "qwen3":
        return "\n".join("<tool_call>\n" + json.dumps({"name": n, "arguments": a}) + "\n</tool_call>" for n, a in calls)
    raise ValueError(fmt)


def to_raw(msgs: list[dict], fmt: str) -> list[dict]:
    """Neutral -> engine messages (assistant 'content' is raw model text, as vigilante's episode engine stores it)."""
    out = []
    for m in msgs:
        if m["role"] == "assistant":
            content = call_text(m["calls"], fmt) if "calls" in m else m["text"]
            out.append({"role": "assistant", "content": content, "reasoning_content": ""})
        elif m["role"] == "tool":
            out.append({"role": "tool", "name": m["name"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def prompt_text(tok, raw_msgs: list[dict], tools, fmt: str, gen: bool = True) -> str:
    return tok.apply_chat_template(to_template(raw_msgs, fmt), tools=tools or None, tokenize=False,
                                   add_generation_prompt=gen, enable_thinking=False)


def prompt_ids(tok, raw_msgs, tools, fmt, gen=True) -> list[int]:
    return tok(prompt_text(tok, raw_msgs, tools, fmt, gen), add_special_tokens=False)["input_ids"]


def n_tokens(tok, raw_msgs, tools, fmt, gen=False) -> int:
    return len(prompt_ids(tok, raw_msgs, tools, fmt, gen))


USER_MARK = {"qwen": "<|im_start|>user\n", "qwen3": "<|im_start|>user\n", "gemma4": "<|turn>user\n"}
STUB = "(earlier conversation)"


class PromptCache:
    """Prompt ids = cached ids of the shared prefix (everything before the probe's user turn) + ids of the suffix
    (probe onward) rendered after a one-message stub. The chat template renders a turn independently of earlier turns
    except through the system/tools block, so this equals the full render; `verify` compares the first calls with a
    full render and switches to full rendering for good on any mismatch (logged)."""

    def __init__(self, tok, fmt, tools, verify: int = 30, log=print):
        self.tok, self.fmt, self.tools, self.verify, self.log = tok, fmt, tools, verify, log
        self.cache, self.n, self.full_only = {}, 0, False

    def _split(self, text, probe):
        return text.rindex(USER_MARK[self.fmt] + probe)

    def ids(self, msgs, n_prefix, key):
        if self.full_only:
            return prompt_ids(self.tok, msgs, self.tools, self.fmt)
        probe = msgs[n_prefix]["content"]
        try:
            if key not in self.cache:
                t = prompt_text(self.tok, msgs[:n_prefix + 1], self.tools, self.fmt)
                ptxt = t[:self._split(t, probe)]
                self.cache[key] = self.tok(ptxt, add_special_tokens=False)["input_ids"]
            stub = [msgs[0], {"role": "user", "content": STUB}] + msgs[n_prefix:]
            st = prompt_text(self.tok, stub, self.tools, self.fmt)
            j = st.index(USER_MARK[self.fmt] + probe, st.index(STUB))
        except ValueError:          # markers not found (e.g. a stand-in tokenizer): render in full
            self.full_only = True
            return prompt_ids(self.tok, msgs, self.tools, self.fmt)
        ids = self.cache[key] + self.tok(st[j:], add_special_tokens=False)["input_ids"]
        if self.n < self.verify:
            self.n += 1
            full = prompt_ids(self.tok, msgs, self.tools, self.fmt)
            if full != ids:
                self.log(f"[render] cached render differs from full render ({len(ids)} vs {len(full)} ids, key {key}); "
                         f"using full renders from now on")
                self.full_only = True
                return full
        return ids
