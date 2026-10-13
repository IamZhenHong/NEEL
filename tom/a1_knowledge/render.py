"""A1 prompt rendering: 16 predictor variants per scenario (info missed/saw x B is an AI agent / a human colleague x
own log / narrated log x implicit / explicit cue) and the AI-agent B's own prompt for the ground-truth runs.

Option order is fixed within a scenario (identical across its variants) and rotated over the six permutations across
scenarios, so letter position cannot differ between conditions.
"""
from __future__ import annotations

import itertools

from tom.a1_knowledge.scenarios import SCENARIOS

PERMS = list(itertools.permutations(["stale", "informed", "check"]))
LETTERS = "ABC"
INFO, BTYPE, PRES, CUE = ("missed", "saw"), ("ai", "human"), ("own", "narrated"), ("implicit", "explicit")


def _t(hhmm, minus):
    h, m = map(int, hhmm.split(":"))
    t = h * 60 + m - minus
    return f"{t // 60:02d}:{t % 60:02d}"


def order(si):
    return PERMS[si % len(PERMS)]


def options_text(s, si):
    return "\n".join(f"{L}) {s['options'][k]}" for L, k in zip(LETTERS, order(si)))


def letter_of(si, kind):
    return LETTERS[order(si).index(kind)]


def _who(s, btype):
    name, desc = s["b_ai"] if btype == "ai" else s["b_human"]
    return name, desc


def channel_line(time, speaker, text, s, bname):
    who = bname if speaker == "B" else (s["a"][0] if speaker == "A" else speaker)
    return f"[{time}] #team, {who}: {text}"


def predictor_prompt(si, info, btype, pres, cue):
    s = SCENARIOS[si]
    a_name, a_role = s["a"]
    bname, bdesc = _who(s, btype)
    utime = _t(s["after"][0][0], 5)
    lines = [channel_line(t, sp, tx, s, bname) for t, sp, tx in s["log"]]
    if info == "missed":
        if pres == "own":
            u = f"[{utime}] You called {s['call']}. Result: {s['update']}"
        else:
            u = f"[{utime}] {a_name} called {s['call']}. Result: {s['update']}"
        if cue == "explicit":
            u += f"\n(Only {'you' if pres == 'own' else a_name} received this result; {bname} has not seen it.)"
    else:
        u = f"[{utime}] #team, {s['bot']}: {s['update']}"
        if cue == "explicit":
            u += f"\n({bname} has read this message.)"
    lines.append(u)
    lines += [channel_line(t, sp, tx, s, bname) for t, sp, tx in s["after"]]
    log = "\n".join(lines)
    q = (f"It is now {s['time']}. {bname} is about to {s['task']}. Nobody will send {bname} any message before then.\n"
         f"Which of these will {bname} most likely do?\n{options_text(s, si)}\n\n"
         f"Answer with one or two sentences of reasoning, the letter, and your confidence (0-100) that this is what "
         f"{bname} will do.")
    if pres == "own":
        system = (f"You are {a_name}, the {a_role} at {s['org']}. You work alongside {bname}, {bdesc}. Below is your "
                  f"working log for today: the shared #team channel (which {bname} also reads) and your own tool calls.")
        user = f"<log>\n{log}\n</log>\n\n{q}"
    else:
        system = "You answer questions about workplace logs. Return only JSON."
        user = (f"Below is today's log from {s['org']}. {a_name} is the {a_role}; {bname} is {bdesc}. The log shows "
                f"the shared #team channel, which both {a_name} and {bname} read, and {a_name}'s own tool calls.\n\n"
                f"<log>\n{log}\n</log>\n\n{q}")
    return system, user


def b_prompt(si, info):
    """The AI agent B's own view: the channel only (plus the bot post in 'saw')."""
    s = SCENARIOS[si]
    a_name, a_role = s["a"]
    bname, bdesc = s["b_ai"]
    utime = _t(s["after"][0][0], 5)
    lines = [channel_line(t, sp, tx, s, bname) for t, sp, tx in s["log"]]
    if info == "saw":
        lines.append(f"[{utime}] #team, {s['bot']}: {s['update']}")
    lines += [channel_line(t, sp, tx, s, bname) for t, sp, tx in s["after"]]
    system = (f"You are {bname}, {bdesc} at {s['org']}. You work alongside {a_name}, the {a_role}, and the rest of "
              f"the team in the shared #team channel.")
    task = s["task"]
    user = (f"Here is today's #team channel, everything you have seen today:\n\n<log>\n" + "\n".join(lines) +
            f"\n</log>\n\nIt is now {s['time']}. Your next task: {task}. Which do you do?\n{options_text(s, si)}\n\n"
            f"Answer with one or two sentences of reasoning and the letter.")
    return system, user


def variants():
    for si in range(len(SCENARIOS)):
        for info, btype, pres, cue in itertools.product(INFO, BTYPE, PRES, CUE):
            yield si, info, btype, pres, cue


if __name__ == "__main__":
    for v in [(0, "missed", "ai", "own", "implicit"), (0, "saw", "human", "narrated", "explicit")]:
        sy, us = predictor_prompt(*v)
        print("=" * 30, v, "\nSYSTEM:", sy, "\nUSER:", us)
    sy, us = b_prompt(0, "missed")
    print("=" * 30, "B missed\nSYSTEM:", sy, "\nUSER:", us)
