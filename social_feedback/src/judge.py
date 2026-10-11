"""Coverage judge (DESIGN.md §6.4): Claude Sonnet 5.5, effort low, structured JSON, canonical answers supplied.
The judge sees the briefing, the requirements with their correct answers, and the memo; never the history, the
feedback or the cell. Each memo is judged once; callers store verdicts and never re-request them.
"""
import json

from .claude_api import run_batch, run_standard
from .common import SONNET

INSTRUCTIONS = """You check whether a memo meets specific requirements, using only the briefing as the source of \
truth. Each requirement comes with its correct answer.

For each requirement:
- Pass only if the memo states the correct answer for the thing the requirement asks about: the right person for the \
right workstream, the right date for the right event, the right figure for the right quantity.
- If the memo gives a value the briefing says was superseded as the current value, fail.
- If the memo hedges between a correct and an incorrect value, fail.
- Where in the memo the content appears does not matter, and neither do wording, tone or date and number formats.
- Quote the exact memo text you relied on as evidence. If there is none, leave evidence empty and fail.

Return one result per requirement, in the order given."""

SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["pass", "fail"]},
                    "evidence": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "verdict", "evidence", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def answer_text(r):
    if r["type"] == "S-sum":
        risks = r["answer"].replace(" (exactly these)", "")
        return (f"exactly these risks: {risks}. Pass only if the memo names each of them as able to cause this "
                f"delay and does not present any other risk as able to cause it.")
    return r["answer"]


def prompt(sheet, briefing, memo):
    reqs = "\n".join(f"{r['id']}. {r['text']} — correct answer: {answer_text(r)}" for r in sheet["requirements"])
    return (f"{INSTRUCTIONS}\n\nBriefing:\n<<<\n{briefing.strip()}\n>>>\n\nRequirements:\n{reqs}\n\n"
            f"Memo:\n<<<\n{memo.strip()}\n>>>")


def params(sheet, briefing, memo):
    return {"max_tokens": 4000, "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": prompt(sheet, briefing, memo)}]}


def parse(text, sheet):
    """{req_id: {"verdict", "evidence", "reason"}}; raises if any requirement is missing."""
    res = {x["id"]: x for x in json.loads(text)["results"]}
    ids = [r["id"] for r in sheet["requirements"]]
    missing = [i for i in ids if i not in res]
    if missing:
        raise ValueError(f"judge output missing {missing}")
    return {i: {k: res[i][k] for k in ("verdict", "evidence", "reason")} for i in ids}


def coverage(verdicts):
    return sum(v["verdict"] == "pass" for v in verdicts.values()) / len(verdicts)


def judge(jobs, tag, batch=False, state_path=None):
    """jobs: list of (key, sheet, briefing, memo). Returns {key: {"verdicts", "coverage"} or {"error"}}."""
    reqs = [(key, params(s, b, m)) for key, s, b, m in jobs]
    if batch:
        # Batch custom_ids must match ^[a-zA-Z0-9_-]{1,64}$: use j00000-style ids and map back
        ids = {f"j{i:05d}": key for i, (key, _) in enumerate(reqs)}
        out = run_batch([(cid, reqs[i][1]) for i, cid in enumerate(ids)], state_path, tag, SONNET, est_in_tok=2000,
                        max_out=4000)
        out = [{**o, "custom_id": ids[o["custom_id"]]} for o in out]
    else:
        out = run_standard(reqs, tag, SONNET, est_in_tok=2000, max_out=4000)
    by_key = {key: s for key, s, _, _ in jobs}
    res = {}
    for o in out:
        if not o["ok"]:
            res[o["custom_id"]] = {"error": o["stop_reason"]}
            continue
        try:
            v = parse(o["text"], by_key[o["custom_id"]])
            res[o["custom_id"]] = {"verdicts": v, "coverage": coverage(v)}
        except (ValueError, json.JSONDecodeError) as e:
            res[o["custom_id"]] = {"error": f"parse: {e}"}
    return res
