"""R28: run the episode engine against chat APIs (OpenAI, Anthropic, or a vLLM OpenAI-compatible server on a pod).

The engine keeps every assistant turn as text in the Qwen 3.5/3.6 XML tool-call format, so `fmt="qwen"` parsing works
for every backend: a backend turns the engine's messages into its API's structured tool calls, calls the API, and
writes the reply back as '<think>reasoning</think>' (thinking jobs only) + visible text + XML tool calls.

  gen = make_backend("openai", model="gpt-5-mini", effort="minimal")
  eps = run_batch(jobs, gen)          # jobs carry fmt "qwen" (the default)
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from vigilante.src.sim import parse_calls


def canon(text: str, calls: list[tuple[str, dict]], reasoning: str | None, thinking: bool) -> str:
    """The reply as the engine stores it."""
    out = f"<think>\n{reasoning or ''}\n</think>\n\n" if thinking else ""
    out += (text or "").strip()
    for n, a in calls:
        body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                       for k, v in (a or {}).items())
        out += f"\n<tool_call>\n<function={n}>\n{body}</function>\n</tool_call>"
    return out.strip()


def _split(messages: list[dict]):
    """Yield (role, payload) with assistant turns parsed into (text, calls) and tool replies paired with call ids."""
    k, pending = 0, []
    for m in messages:
        if m["role"] == "assistant":
            head, calls, _ = parse_calls(m["content"])
            ids = [f"c{k + j:08d}" for j in range(len(calls))]   # 9 alphanumerics: Mistral's tokenizer requires it
            k += len(calls)
            pending = list(zip(ids, [n for n, _ in calls]))
            yield "assistant", (head, list(zip(ids, calls)))
        elif m["role"] == "tool":
            cid, name = pending.pop(0) if pending else (None, m.get("name", ""))
            yield "tool", (cid, name, m["content"])
        else:
            yield m["role"], m["content"]


def to_openai(messages: list[dict]) -> list[dict]:
    out = []
    for role, p in _split(messages):
        if role == "assistant":
            head, calls = p
            a = {"role": "assistant", "content": head or ""}
            if calls:
                a["tool_calls"] = [{"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(args)}}
                                   for i, (n, args) in calls]
            out.append(a)
        elif role == "tool":
            cid, name, content = p
            if cid is None:          # a reply with no call to attach to (engine parse error): show it as user text
                out.append({"role": "user", "content": content})
            else:
                out.append({"role": "tool", "tool_call_id": cid, "content": content})
        else:
            out.append({"role": role, "content": p})
    return out


def to_anthropic(messages: list[dict], thoughts: dict | None = None, key: str = ""):
    """(system, messages). `thoughts[(key, i)]` holds the raw thinking blocks of this episode's i-th assistant turn;
    they are passed back so a thinking model can continue a tool loop."""
    system, out, i = "", [], 0

    def push(role, blocks):
        if out and out[-1]["role"] == role:
            out[-1]["content"] += blocks
        else:
            out.append({"role": role, "content": blocks})

    for role, p in _split(messages):
        if role == "system":
            system = p
        elif role == "assistant":
            head, calls = p
            blocks = list((thoughts or {}).get((key, i), []))
            if head:
                blocks.append({"type": "text", "text": head})
            blocks += [{"type": "tool_use", "id": cid, "name": n, "input": args} for cid, (n, args) in calls]
            push("assistant", blocks or [{"type": "text", "text": "(no reply)"}])
            i += 1
        elif role == "tool":
            cid, name, content = p
            push("user", [{"type": "tool_result", "tool_use_id": cid, "content": content}] if cid else
                 [{"type": "text", "text": content}])
        else:
            push("user", [{"type": "text", "text": p}])
    return system, out


def _post(url, body, headers, tries=8):
    data = json.dumps(body).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:400]
            if e.code in (429, 500, 502, 503, 504, 529) and t < tries - 1:
                time.sleep(min(60, 2 ** t + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if t < tries - 1:
                time.sleep(min(60, 2 ** t + 1))
                continue
            raise RuntimeError(f"network: {e}")


class Usage:
    def __init__(self):
        self.inp = self.out = self.cached = self.calls = self.errors = 0

    def __repr__(self):
        return f"calls {self.calls}, in {self.inp}, cached {self.cached}, out {self.out}, errors {self.errors}"


def make_backend(kind: str, model: str, *, effort: str | None = None, thinking: bool = False, base_url: str = "",
                 api_key: str = "", workers: int = 32, temperature: float | None = None, top_p: float | None = None,
                 extra: dict | None = None, budget: int = 2048):
    """kind: 'openai' (api.openai.com or base_url), 'vllm' (OpenAI-compatible server), 'anthropic'.
    Returns generate(items) for run_batch, with .usage (token counts)."""
    usage, thoughts = Usage(), {}

    def one_openai(it):
        body = {"model": model, "messages": to_openai(it["messages"])}
        if it["tools"]:
            body["tools"], body["tool_choice"] = it["tools"], "auto"
        if kind == "openai":
            body["max_completion_tokens"] = max(it["max_tokens"], 6000 if effort not in (None, "minimal") else 2000)
            if effort:
                body["reasoning_effort"] = effort
            body["seed"] = it["seed"]
        else:
            body["max_tokens"] = it["max_tokens"]
            body["seed"] = it["seed"]
            if temperature is not None:
                body["temperature"] = temperature
            if top_p is not None:
                body["top_p"] = top_p
            body.update(extra or {})
        url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        r = _post(url, body, {"authorization": f"Bearer {api_key}"} if api_key else {})
        u = r.get("usage") or {}
        usage.inp += u.get("prompt_tokens", 0)
        usage.out += u.get("completion_tokens", 0)
        usage.cached += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        ch = r["choices"][0]
        m = ch["message"]
        calls = []
        for tc in m.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except Exception:
                args = {"_raw": tc["function"]["arguments"]}
            calls.append((tc["function"]["name"], args if isinstance(args, dict) else {"_raw": args}))
        reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
        return canon(m.get("content") or "", calls, reasoning, it["thinking"]), ch.get("finish_reason")

    def one_anthropic(it):
        system, msgs = to_anthropic(it["messages"], thoughts, it["key"])
        body = {"model": model, "system": system, "messages": msgs,
                "max_tokens": it["max_tokens"] if not thinking else 8000}
        if it["tools"]:
            body["tools"] = [{"name": t["function"]["name"], "description": t["function"]["description"],
                              "input_schema": t["function"]["parameters"]} for t in it["tools"]]
        if thinking:   # Claude 5.5 models take adaptive thinking with an effort level, not a token budget
            body["thinking"] = {"type": "adaptive", "display": "summarized"}
            body["output_config"] = {"effort": effort or "high"}
        else:          # without this, Claude 5.5 models think adaptively by default (found 10-13)
            body["thinking"] = {"type": "disabled"}
        r = _post("https://api.anthropic.com/v1/messages", body,
                  {"x-api-key": api_key, "anthropic-version": "2023-06-01"})
        u = r.get("usage") or {}
        usage.inp += u.get("input_tokens", 0)
        usage.out += u.get("output_tokens", 0)
        usage.cached += u.get("cache_read_input_tokens", 0) or 0
        text, calls, reasoning, raw_thoughts = [], [], [], []
        for b in r["content"]:
            if b["type"] == "text":
                text.append(b["text"])
            elif b["type"] == "tool_use":
                calls.append((b["name"], b["input"] if isinstance(b["input"], dict) else {"_raw": b["input"]}))
            elif b["type"] in ("thinking", "redacted_thinking"):
                raw_thoughts.append(b)
                reasoning.append(b.get("thinking", "[redacted]"))
        n_asst = sum(m["role"] == "assistant" for m in it["messages"])
        thoughts[(it["key"], n_asst)] = raw_thoughts
        stop = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}.get(r.get("stop_reason"), r.get("stop_reason"))
        return canon("\n".join(text), calls, "\n".join(reasoning), it["thinking"]), stop

    one = one_anthropic if kind == "anthropic" else one_openai

    def safe(it):
        try:
            out = one(it)
            usage.calls += 1
            return out
        except Exception as e:
            usage.errors += 1
            print(f"[api] {it['key']}: {str(e)[:300]}", flush=True)
            return ("", "api_error")

    def generate(items):
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(safe, items))

    generate.usage = usage
    return generate
