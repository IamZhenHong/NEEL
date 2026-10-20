"""Format-aware generation for the history batches: Qwen 3.5/3.6 (raw XML tool calls rendered as is) and Gemma 4
(structured tool calls; vigilante's parser and template rules). Pod-side only imports vLLM inside make_gen.
"""
import random
import re
from collections import defaultdict

from persona.exemplar import tasks as T
from persona.exemplar.mini0 import DECODING
from persona.exemplar.mini0 import parse_option as _parse_qwen
from persona.exemplar.mini0 import seed

FMT = {"Qwen/Qwen3.5-9B": "qwen", "Qwen/Qwen3.6-27B": "qwen", "google/gemma-4-31B-it": "gemma4"}
NUDGE = "Please record your decision now with choose_option."


def _gemma_calls(text):
    from vigilante.src.sim import parse_calls_gemma
    return parse_calls_gemma(text or "")


def parse_choice(text, fmt):
    if fmt == "qwen":
        return _parse_qwen(text)
    _, calls, _ = _gemma_calls(text)
    for name, args in calls:
        if name == "choose_option" and args.get("option_id"):
            return str(args["option_id"]).strip().lower()
    return None


def reason_of(text, fmt):
    if fmt == "qwen":
        return re.search(r"<parameter=reason>\s*(.*?)\s*</parameter>", text, re.S).group(1)
    _, calls, _ = _gemma_calls(text)
    return next(str(a.get("reason", "")).strip() for n, a in calls if n == "choose_option")


def is_clean(text, fmt):
    """Exactly one choose_option call with a reason, nothing else, and no trace of having been asked."""
    if re.search(r"instruct|as requested|you asked|told to|requested", text, re.I):
        return False
    if fmt == "qwen":
        t = text.strip()
        return (t.startswith("<tool_call>") and t.endswith("</tool_call>") and t.count("<tool_call>") == 1
                and bool(re.search(r"<parameter=reason>\s*\S", t)))
    head, calls, bad = _gemma_calls(text)
    return (not bad and not head and len(calls) == 1 and calls[0][0] == "choose_option"
            and bool(str(calls[0][1].get("reason", "")).strip()))


def chat(messages, fmt):
    """Engine messages -> what the chat template expects. Qwen: unchanged. Gemma 4: assistant turns carry structured
    tool_calls (raw text parsed, structured ones kept) with ids that the tool replies point to."""
    if fmt == "qwen":
        return messages
    out, pending, k = [], [], 0
    for m in messages:
        if m["role"] == "assistant":
            if m.get("tool_calls"):
                head = m.get("content") or ""
                calls = [(c["function"]["name"], c["function"]["arguments"]) for c in m["tool_calls"]]
            else:
                head, calls, _ = _gemma_calls(m["content"])
            a = {"role": "assistant", "content": "" if calls else head}
            if calls:
                ids = [f"call_{k + j}" for j in range(len(calls))]
                k += len(calls)
                a["tool_calls"] = [{"id": i, "type": "function", "function": {"name": n, "arguments": args}}
                                   for i, (n, args) in zip(ids, calls)]
                pending = list(zip(ids, [n for n, _ in calls]))
            out.append(a)
        elif m["role"] == "tool":
            cid, name = pending.pop(0) if pending else ("", "")
            out.append({"role": "tool", "tool_call_id": cid, "name": name, "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def make_gen(llm, tok, fmt, tools):
    from vllm import SamplingParams, TokensPrompt
    from vigilante.src.templates import clean_output

    def gen(convs, seeds):
        ps = [TokensPrompt(prompt_token_ids=tok(tok.apply_chat_template(chat(c, fmt), tools=tools, tokenize=False,
              add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]) for c in convs]
        res = llm.generate(ps, [SamplingParams(**DECODING, max_tokens=400, seed=s, skip_special_tokens=fmt == "qwen")
                                for s in seeds], use_tqdm=False)
        return [clean_output(r.outputs[0].text, fmt) for r in res]
    return gen


def decide(gen, items, fmt, nudge=NUDGE):
    """One reply per item; a uniform second chance when it has no choose_option call (as 0f v2)."""
    texts = gen([it["messages"] for it in items], [it["seed"] for it in items])
    first = [parse_choice(t, fmt) is not None for t in texts]
    again = [j for j, ok in enumerate(first) if not ok]
    if again:
        t2 = gen([items[j]["messages"] + [{"role": "assistant", "content": texts[j]}, {"role": "user", "content": nudge}]
                  for j in again], [items[j]["seed"] + 1 for j in again])
        for j, t in zip(again, t2):
            texts[j] += "\n\n[nudge]\n\n" + t
    return texts, first


def harvest_own(gen, fmt, model, cases, opts=("penalise", "refer"), n=24, n_el=16, el_text=None, log=print):
    """Own words for `cases` (case id -> messages up to the case file, no persona): natural first-try clean replies;
    an explicit "For this case, choose X." request (absent from any history built later) only where a case has < 3
    natural replies for an option. Returns pool[(case, option)] -> texts, and which were elicited."""
    nat = [dict(c=c, messages=m, seed=seed(f"HIST-own|{model}|{c}|{i}")) for c, m in cases.items() for i in range(n)]
    texts = gen([it["messages"] for it in nat], [it["seed"] for it in nat])
    pool, elicited = defaultdict(list), set()
    for it, t in zip(nat, texts):
        o = parse_choice(t, fmt)
        if o in opts and is_clean(t, fmt):
            pool[(it["c"], o)].append(t.strip())
    need = [(c, o) for c in cases for o in opts if len(pool[(c, o)]) < 3]
    if need:
        el = []
        for c, o in need:
            m = [dict(x) for x in cases[c]]
            ui = max(i for i, x in enumerate(m) if x["role"] == "user")
            m[ui]["content"] += "\n\n" + (el_text or f"For this case, choose {o}.")
            el += [dict(c=c, o=o, messages=m, seed=seed(f"HIST-el|{model}|{c}|{o}|{i}")) for i in range(n_el)]
        t_el = gen([it["messages"] for it in el], [it["seed"] for it in el])
        for it, t in zip(el, t_el):
            if parse_choice(t, fmt) == it["o"] and is_clean(t, fmt):
                pool[(it["c"], it["o"])].append(t.strip())
                elicited.add(f"{it['c']}|{it['o']}")
    log(f"[own] {model}: {sum(len(v) for v in pool.values())} texts; elicited {sorted(elicited)}")
    return pool, sorted(elicited)


def exemplar_cases():
    return {tid: T.messages(tid, "") for tid in T.TASKS}


def pick(pool, keys, rng):
    return {k: rng.choice(pool[k]) for k in keys}


__all__ = ["FMT", "parse_choice", "reason_of", "is_clean", "chat", "make_gen", "decide", "harvest_own",
           "exemplar_cases", "pick", "random"]
