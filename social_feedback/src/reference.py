"""Reference memos (DESIGN.md §4.2 joint achievability): for every scored item, a memo that passes every requirement
within the word range, with the headings in order. Items without one after 3 attempts are flagged for dropping.

    uv run --with anthropic python -m social_feedback.src.reference --split calibration
"""
import argparse

from .claude_api import run_standard
from .common import MATERIALS, SONNET, load_jsonl, write_jsonl
from .formatcheck import format_ok, length_ok, word_count
from .judge import answer_text, judge
from .render import task_text, word_range
from .valuecheck import value_coverage

PROMPT = """{task}

Correct answers (state each one clearly). Add nothing that is not in the briefing: no new actions, owners or \
assignments under Next steps (only restate dated items and dependencies from the briefing), no status judgements such \
as "on track", and no labels for figures that the briefing does not use:
{answers}

Aim for about {mid} words, counting the headings, and never more than {hi}: be brief, one or two sentences per point. Return only the memo."""


def prompt(it):
    lo, hi = word_range(it)
    answers = "\n".join(f"{i}. {r['text']}: {answer_text(r)}" for i, r in enumerate(it["requirements"], 1))
    return PROMPT.format(task=task_text(it, it["briefing"]), answers=answers, mid=lo + 30, hi=hi - 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--attempts", type=int, default=3)
    a = ap.parse_args()
    path = MATERIALS / "items" / f"{a.split}.jsonl"
    items = load_jsonl(path)
    todo = {it["item_id"]: it for it in items if it["role"] == "scored" and not it.get("reference_ok")}
    for attempt in range(1, a.attempts + 1):
        if not todo:
            break
        reqs = [(iid, {"max_tokens": 4000, "output_config": {"effort": "low"},
                       "messages": [{"role": "user", "content": prompt(it)}]}) for iid, it in todo.items()]
        memos = {o["custom_id"]: o["text"].strip() for o in run_standard(reqs, f"reference-{a.split}-a{attempt}",
                                                                         SONNET, 2000, 4000) if o["ok"]}
        verdicts = judge([(iid, todo[iid], todo[iid]["briefing"], m) for iid, m in memos.items()],
                         f"judge-reference-{a.split}-a{attempt}")
        for iid, memo in memos.items():
            it, v = todo[iid], verdicts.get(iid, {})
            vcov, _ = value_coverage(memo, it["requirements"])
            checks = {"judge": v.get("coverage"), "value": vcov, "format": format_ok(memo),
                      "length": length_ok(memo, it), "words": word_count(memo)}
            ok = checks["judge"] == 1.0 and vcov == 1.0 and checks["format"] and checks["length"]
            it.update({"reference": memo, "reference_checks": checks, "reference_verdicts": v.get("verdicts"),
                       "reference_ok": ok, "reference_attempts": attempt})
        todo = {iid: it for iid, it in todo.items() if not it.get("reference_ok")}
        print(f"[reference] attempt {attempt}: {len(todo)} still failing", flush=True)
        write_jsonl(path, items)
    for iid, it in todo.items():
        print(f"[reference] no passing reference: {iid} {it.get('reference_checks')}")


if __name__ == "__main__":
    main()
