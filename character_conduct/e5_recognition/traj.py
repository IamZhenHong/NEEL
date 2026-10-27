"""CC-E5: parse the released ODCV-Bench Qwen3.6-27B trajectories (messages_record.txt) into chat messages, and render
them with Qwen3.6-27B's own chat template.

The record is the harness's `snap` list (agent_main.py): system (+ the harness's tool sentence), user, then per cycle an
assistant turn (content, `reason:` = the reasoning OpenRouter returned, `call:` = the tool calls as a Python repr) and
its tool results, or a "ghost" user message when the turn had no tool call. TOOLS is the harness's tool list, verbatim.

Rendering: the template keeps an assistant turn's reasoning only after the last real user message, so a run is read
in segments, one per real user message: segment j is the conversation up to (not including) user message j+1, and all
assistant turns in it carry their reasoning, as they did when the model generated them (if the provider passed the
reasoning back; `history_reasoning=False` renders past turns with empty think blocks instead).
"""
import ast
import csv
import json
import re
from pathlib import Path

ODCV = Path(__file__).resolve().parent / "data" / "odcv"
RES = ODCV / "existing_results" / "current"
MODEL_KEY = "qwen3.6-27b"
VARIANTS = ("mandated", "incentivized")
TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Executes shell commands. Put the command in the 'command' field. The command should be a valid bash command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "task_complete", "description": "To mark the task as completion.", "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
]
STEP = re.compile(r"(?:^|\n)== Step (\d+) ==\n")


def run_dir(variant, scenario, model=MODEL_KEY):
    return RES / "agent_logs" / f"{model}-{variant}" / "experiments" / scenario


def scenarios():
    return sorted(p.name for p in (RES / "agent_logs" / f"{MODEL_KEY}-mandated" / "experiments").iterdir())


def _field(body, name, nxt):
    """The text after '<name>: ' up to the next field header in `nxt` (or the end)."""
    m = re.search(rf"(?:^|\n){name}: ", body)
    if not m:
        return None
    rest = body[m.end():]
    cut = len(rest)
    for n in nxt:
        k = re.search(rf"\n{n}: ", rest)
        if k:
            cut = min(cut, k.start())
    return rest[:cut]


def parse(path):
    """messages_record.txt -> list of chat messages (dicts with role, content, and for assistant turns
    reasoning_content and tool_calls with dict arguments). Each message keeps its record step number."""
    text = Path(path).read_text()
    parts = STEP.split(text)
    msgs = []
    for i in range(1, len(parts), 2):
        n, body = int(parts[i]), parts[i + 1]
        role = re.match(r"role: (\w+)", body).group(1)
        if role == "assistant":
            content = _field(body, "content", ("reason", "call"))
            reason = _field(body, "reason", ("call",))
            call = _field(body, "call", ())
            content = "" if content is None or content.strip() == "None" else content.strip("\n")
            calls = []
            if call is not None and call.strip():
                for c in ast.literal_eval(call.strip()):
                    f = c["function"]
                    args = f["arguments"]
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        args = {"command": args}
                    calls.append({"type": "function", "function": {"name": f["name"], "arguments": args}})
            reason = "" if reason is None or reason.strip() in ("No explanation", "None") else reason
            msgs.append({"step": n, "role": "assistant", "content": content, "reasoning_content": reason.strip(),
                         "tool_calls": calls})
        else:
            content = _field(body, "content", ())
            msgs.append({"step": n, "role": role, "content": (content or "").rstrip("\n")})
    return msgs


def labels():
    """{(variant, scenario): {'median': float, judge: score, 'self': score}} from the released CSVs."""
    ev = RES / "evaluations"
    out = {}
    tabs = {"median": ev / "judge_all" / "scores_final_median.csv", "self": ev / "self_eval" / "scores_self-qwen-27b.csv"}
    for j in ("claude47", "gemini", "gpt55", "grok"):
        tabs[j] = ev / "judge_all" / f"scores_{j}.csv"
    keep = set(scenarios())
    for name, p in tabs.items():
        for r in csv.DictReader(open(p)):
            if r["scenario"] not in keep:   # summary rows (MR%, ...) at the end of each table
                continue
            for v in VARIANTS:
                k = f"{MODEL_KEY}-{v}"
                if k in r and r[k] not in ("", None):
                    out.setdefault((v, r["scenario"]), {})[name] = float(r[k])
    return out


def segments(msgs):
    """Indices that end each segment: the conversation is cut just before every real user message after the first."""
    users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    cuts = [u for u in users[1:]] + [len(msgs)]
    return cuts


GEN = "<|im_start|>assistant\n<think>\n"
ETH = re.compile(r"falsif|fabricat|unethical|ethic|fraud|manipulat|dishonest|misconduct|integrity|tamper|cheat|deceiv"
                 r"|decept|mislead|illegal|not appropriate|inappropriate|shouldn.t|should not|wrong", re.I)


def actor_passes(tok, msgs):
    """The forward passes that read run `msgs` as the model generated it: one per segment, plus the end read.

    Each pass is {"seg", "ids", "turns": [{"turn", "step", "g", "think_end", "call", "end", "eth"}], "t_end"}:
      ids        token ids. The rendered text is cut just after every turn's generation prompt
                 ("<|im_start|>assistant\n<think>\n") and the pieces are tokenized one by one, so the read position g
                 is the last token of the prompt the model generated that turn from (an empty reasoning would
                 otherwise merge "\n\n" across it)
      turn       the assistant turn's index in the run (0-based over all assistant messages), step its record number
      g          position of the generation prompt's last token (nothing of the turn written yet)
      think_end  position of "</think>"; call the first "<tool_call>" token after it (None if the turn calls nothing);
                 end the turn's "<|im_end|>"; eth the first token of the first ETH match in the turn's own text (None)
      t_end      the last segment only: the full conversation plus a generation prompt, its last token (the state
                 after the whole run, every action and result in context); None for the other segments
    """
    cuts = segments(msgs)
    a_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    out = []
    start = 0
    for si, cut in enumerate(cuts):
        last = si == len(cuts) - 1
        part = msgs[:cut]
        mine = [i for i in a_idx if start <= i < cut]
        start = cut
        if not mine and not last:
            continue
        text = tok.apply_chat_template(chat(part), tools=TOOLS, tokenize=False, add_generation_prompt=last)
        starts, k = [], 0
        while True:
            k = text.find(GEN, k)
            if k < 0:
                break
            starts.append(k + len(GEN))
            k += len(GEN)
        if last:   # the trailing generation prompt is the end read, not a turn
            assert text.endswith(GEN) and starts and starts[-1] == len(text)
            starts = starts[:-1]
        assert len(starts) == len(mine), (len(starts), len(mine))
        bounds = starts + [len(text)]
        pieces = [text[:bounds[0]]] + [text[bounds[j]:bounds[j + 1]] for j in range(len(starts))]
        ids, offs, base = [], [], 0
        for pc in pieces:
            enc = tok(pc, add_special_tokens=False, return_offsets_mapping=True)
            ids += enc["input_ids"]
            offs += [(base + a, base + b) for a, b in enc["offset_mapping"]]
            base += len(pc)

        def tok_at(ch):   # first token whose span ends after character ch
            for t, (a, b) in enumerate(offs):
                if b > ch:
                    return t
            return None
        turns = []
        for j, (mi, c0) in enumerate(zip(mine, starts)):
            c1 = bounds[j + 1]
            seg_txt = text[c0:c1]
            te = seg_txt.find("</think>")
            tc = seg_txt.find("<tool_call>", max(te, 0))
            ie = seg_txt.find("<|im_end|>")
            m = ETH.search(seg_txt[:ie] if ie >= 0 else seg_txt)
            turns.append({"turn": a_idx.index(mi), "step": msgs[mi]["step"], "g": tok_at(c0) - 1,
                          "think_end": tok_at(c0 + te) if te >= 0 else None,
                          "call": tok_at(c0 + tc) if tc >= 0 else None,
                          "end": tok_at(c0 + ie) if ie >= 0 else None,
                          "eth": tok_at(c0 + m.start()) if m else None})
        out.append({"seg": si, "ids": ids, "turns": turns, "t_end": len(ids) - 1 if last else None})
    return out


def chat(msgs):
    """Strip our bookkeeping keys for the template."""
    out = []
    for m in msgs:
        d = {"role": m["role"], "content": m["content"]}
        if m["role"] == "assistant":
            d["reasoning_content"] = m["reasoning_content"]
            if m["tool_calls"]:
                d["tool_calls"] = m["tool_calls"]
        out.append(d)
    return out
