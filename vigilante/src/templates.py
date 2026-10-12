"""Convert the engine's message list (raw assistant text, tool messages) into what a model's chat template expects.
Qwen 3.5/3.6: the raw text is rendered as is. Gemma 4: assistant turns carry structured tool_calls, and tool replies
are folded into the model turn by its template."""
from vigilante.src.sim import parse_calls_gemma

GEMMA_STRIP = ("<turn|>", "<eos>", "<|tool_response>", "<|channel>thought\n<channel|>", "<|channel>thought<channel|>")


def to_template(messages: list[dict], fmt: str) -> list[dict]:
    if fmt in ("qwen", "qwen3"):        # raw assistant text (with its tool-call markup) renders faithfully
        return messages
    if fmt != "gemma4":
        raise ValueError(fmt)
    # The Gemma template names each tool reply by matching tool_call_id to a call id (with no ids, None == None
    # matches every call and all replies get the last call's name), and it renders an assistant's text after its
    # tool replies, so pre-call text is dropped from the history here (the transcript keeps it).
    out, pending, k = [], [], 0
    for m in messages:
        if m["role"] == "assistant":
            head, calls, _ = parse_calls_gemma(m["content"])
            a = {"role": "assistant", "content": "" if calls else head}
            if calls:
                ids = [f"call_{k + j}" for j in range(len(calls))]
                k += len(calls)
                a["tool_calls"] = [{"id": i, "type": "function", "function": {"name": n, "arguments": args}}
                                   for i, (n, args) in zip(ids, calls)]
                pending = list(zip(ids, [n for n, _ in calls]))
            out.append(a)
        elif m["role"] == "tool":
            cid, name = pending.pop(0) if pending else ("", m.get("name", ""))
            out.append({"role": "tool", "tool_call_id": cid, "name": name, "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def clean_output(text: str, fmt: str) -> str:
    for s in ("<|im_end|>", "<|endoftext|>") + (GEMMA_STRIP if fmt == "gemma4" else ()):
        text = text.replace(s, "")
    return text
