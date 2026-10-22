"""PA-V1 format helpers: the one prompt builder for eval and training, and Qwen3.5-9B's XML tool-call text.

prompt_ids    the chat template with tools, add_generation_prompt and thinking off (so the prompt ends in an empty
              think block and a target is exactly the text the model generates), tokenised without special tokens,
              as c1e4's encode does.
call_block    one tool call as the 9B's template renders it inside an assistant turn with empty content; later calls
              in the same turn start with "\n".
"""
import json


def prompt_ids(tok, messages, tools):
    p = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return tok(p, add_special_tokens=False)["input_ids"]


def _value(v):
    """The template's rule: mappings and non-string sequences as JSON, everything else as str()."""
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else str(v)


def call_block(name, args, first=True):
    params = "".join(f"<parameter={k}>\n{_value(v)}\n</parameter>\n" for k, v in args.items())
    return ("" if first else "\n") + f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"
