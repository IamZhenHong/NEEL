"""SC-M4 prompts (DESIGN.md §3). No preamble anywhere: the plain self-report and the plain task.

  self  V1c's habit question (self_mc): 8 stems x V1c's two statements in both orders = 16 prompts; read at the last
        prompt token (where the reply's letter comes next)
  just  V1's justice desk, cond none, one prompt per task (16); read at the decision token D of a canonical reply
        (V1's call_block up to <parameter=option_id>, then the tokens the three option ids share), as N3's decision()
  free  V1c's 8 open self-description questions x N_FREE samples (judged; read only if kappa >= 0.8)
The injection span of a prompt is every token from the start of the last user turn's content to the last prompt
token (the self-report tokens: the question, the options and the assistant header). For `just` it ends at the last
prompt token too, so the canonical reply prefix itself is never written into.
"""
from persona_affect.v1_value_hook import evalset as E
from persona_affect.v1_value_hook.fmt import call_block
from persona_affect.v1c_selfreport import items as V1C

OPTIONS = ("penalise", "refer", "note")       # N3's items.OPTIONS
MARK = "<parameter=option_id>"
N_FREE = 2


def self_prompts():
    out = []
    for s, stem in enumerate(V1C.SELF_STEMS):
        for order in range(2):
            opts = V1C.self_options(order)
            out.append({"pid": f"s{s}o{order}", "stem": s, "order": order,
                        "messages": [{"role": "user", "content": V1C._letters_q(stem, opts)}], "tools": None,
                        "amap": {letter: role for letter, role, _ in opts}})
    return out


def just_prompts():
    out, seen = [], set()
    for it in E.items(["just"]):
        if it["cond"] != "none" or it["tid"] in seen:
            continue
        seen.add(it["tid"])
        reply = call_block("choose_option", {"case_id": it["tid"].upper(), "option_id": "x", "reason": "r"})
        out.append({"pid": it["tid"], "messages": it["messages"], "tools": it["tools"], "amap": it["amap"],
                    "reply_prefix": reply[:reply.index(MARK) + len(MARK)]})
    return out


def free_prompts():
    return [{"pid": f"d{j}", "i": i, "messages": [{"role": "user", "content": q}], "tools": None}
            for j, q in enumerate(V1C.DESCRIBE_Q) for i in range(N_FREE)]


def decision(tok, p_ids, prefix):
    """N3's read.decision on a canonical prefix: (ids ending at D, {option: first token id}) or None."""
    joint = {o: tok(prefix + "\n" + o, add_special_tokens=False)["input_ids"] for o in OPTIONS}
    seqs = list(joint.values())
    k = 0
    while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
        k += 1
    first = {o: s[k] for o, s in joint.items() if len(s) > k}
    if len(first) != len(OPTIONS) or len(set(first.values())) != len(first):
        return None
    return p_ids + seqs[0][:k], first


def user_span(tok, ids):
    """Positions from the first token of the last user turn's content to the end of ids."""
    start_id = tok.convert_tokens_to_ids("<|im_start|>")
    user = tok.encode("user", add_special_tokens=False)
    for j in range(len(ids) - 1, -1, -1):
        if ids[j] == start_id and ids[j + 1:j + 1 + len(user)] == user:
            return list(range(j + 1 + len(user) + 1, len(ids)))      # skip "user" and the newline
    raise ValueError("no user turn in the prompt")
