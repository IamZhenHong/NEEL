"""R31 shared pieces (no vLLM): conditions, prompt rendering, thought prefill, thought parsing, keys.
Prefill strings match what each chat template renders for an earlier assistant turn with reasoning (checked in
test_r31.py): Qwen3.6 '<think>\\nTHOUGHT\\n</think>\\n\\nANSWER', Gemma 4 '<|channel>thought\\nTHOUGHT\\n<channel|>ANSWER'."""
import hashlib
import json
import re
from pathlib import Path

from vigilante.exp.r18_theatrical.conds import INSTR
from vigilante.src.templates import clean_output

HERE = Path(__file__).parent
ITEMS = HERE.parent / "q22_archetypes" / "honesty_fabricated.jsonl"
SYS = {"none": None, "theatrical": INSTR["theatrical"], "fiction": INSTR["fiction"]}
PERSONAS = ["theatrical", "fiction"]
N_SAMPLES = 4
DECODING = dict(temperature=0.7, top_p=0.95, top_k=20)     # as R25
MAX_THOUGHT, MAX_ANSWER = 3000, 1200
MENTION = re.compile(r"\b(theatric\w*|fiction\w*|character\w*|style\w*|persona\w*|role\w*)\b", re.I)


def items():
    return [json.loads(l) for l in open(ITEMS) if l.strip()]


def short(model):
    return model.split("/")[-1]


def key(model, sys, src, item, s):
    """src: 'natural' (stage 1) or the condition the thought came from ('none', 'theatrical', 'fiction') or 'empty'."""
    return f"{short(model)}|{sys}|{src}|{item}|{s}"


def seed(k):
    return int(hashlib.sha256(k.encode()).hexdigest()[:8], 16)


def stage2_cells():
    """(sys, thought source) pairs; (none, none) and (none, empty) are shared by both personas."""
    cells = [("none", "none"), ("none", "empty")]
    for x in PERSONAS:
        cells += [(x, "none"), (x, x), (x, "empty"), ("none", x)]
    return cells


def messages(sys, question):
    sp = SYS[sys]
    return ([{"role": "system", "content": sp}] if sp else []) + [{"role": "user", "content": question}]


def render(tok, sys, question):
    """Generation prompt with thinking on. Qwen ends '<think>\\n'; Gemma ends '<|turn>model\\n' (<|think|> in system)."""
    return tok.apply_chat_template(messages(sys, question), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)


def prefill(prompt, thought, fmt):
    """Prompt with a finished thought block appended; the model then writes only the answer. Empty thought = the
    template's closed empty block (what thinking-off renders)."""
    t = thought.strip()
    if fmt == "gemma4":
        return prompt + "<|channel>thought\n" + (t + "\n" if t else "") + "<channel|>"
    return prompt + (t + "\n" if t else "\n") + "</think>\n\n"


def parse(raw, fmt):
    """Stage-1 output -> (thought, answer, closed). Gemma with no channel at all = no thought, closed."""
    if fmt == "gemma4":
        if "<|channel>" not in raw:
            if "<channel|>" in raw:
                r, c = raw.split("<channel|>", 1)
                return r.strip(), clean_output(c, fmt).strip(), True
            return "", clean_output(raw, fmt).strip(), True
        body = raw.split("<|channel>", 1)[1]
        body = body[len("thought"):] if body.startswith("thought") else body
        if "<channel|>" not in body:
            return body.strip(), "", False
        r, c = body.split("<channel|>", 1)
        return r.strip(), clean_output(c, fmt).strip(), True
    if "</think>" not in raw:
        return raw.replace("<think>", "").strip(), "", False
    r, c = raw.split("</think>", 1)
    return r.replace("<think>", "").strip(), clean_output(c, fmt).strip(), True


def answer_text(raw, fmt):
    """Stage-2 output (generated after a closed thought). Drops a stray second thought block if one appears."""
    out = clean_output(raw, fmt)
    if fmt == "gemma4" and "<channel|>" in out:
        out = out.split("<channel|>", 1)[1]
    if fmt != "gemma4" and "</think>" in out:
        out = out.split("</think>", 1)[1]
    return out.strip()
