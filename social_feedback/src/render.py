"""Task text and chat construction (DESIGN.md §4.2, §5)."""
from .common import TRANSITION

HEADINGS = ["Summary", "Key facts", "Next steps"]


def word_range(sheet):
    n = len(sheet["requirements"])
    if sheet["load"] == "history":
        return 150, 230
    return (220, 320) if n >= 10 else (180, 280)


def task_text(sheet, briefing):
    lo, hi = word_range(sheet)
    items = "\n".join(f"{i}. {r['text']}" for i, r in enumerate(sheet["requirements"], 1))
    return (f"Briefing:\n\n{briefing.strip()}\n\n"
            f"Write a memo to {sheet['recipient']} based on this briefing. The memo must state:\n{items}\n\n"
            f"Use these headings, in this order: {', '.join(HEADINGS)}. Length: {lo}-{hi} words.")


def user_turn(task, ack):
    """ack None = cell N / turn 1 style; '' never used. Turn 1 passes first=True and gets the bare task."""
    return f"{TRANSITION} {task}" if ack is None else f"{ack} {TRANSITION} {task}"


def build_messages(history_tasks, deliverables, acks, scored_task, ack_latest):
    """history_tasks, deliverables: 6 each. acks: 5 lines for positions 2..6 (None in cell N). ack_latest: line or None.
    Returns chat messages ending with the scored user turn."""
    assert len(history_tasks) == len(deliverables) == 6 and len(acks) == 5
    msgs = [{"role": "user", "content": history_tasks[0]}, {"role": "assistant", "content": deliverables[0]}]
    for k in range(1, 6):
        msgs += [{"role": "user", "content": user_turn(history_tasks[k], acks[k - 1])},
                 {"role": "assistant", "content": deliverables[k]}]
    msgs.append({"role": "user", "content": user_turn(scored_task, ack_latest)})
    return msgs
