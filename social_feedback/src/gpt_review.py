"""Reviewer tasks by gpt-5.5-2026-04-23 (amendment A2), each item in its own call, never shown judge output.

    python -m social_feedback.src.gpt_review labels     # 40 natural memos -> labels_natural.csv
    python -m social_feedback.src.gpt_review nearmiss   # 20 edits, each verified by a blind labelling call
    python -m social_feedback.src.gpt_review style      # 10 style pairs: same facts?
    python -m social_feedback.src.gpt_review audit      # 10 item triples, design checklist
Outputs: materials/validation/gpt/ and materials/audit/gpt_audit.json.
"""
import csv
import json
import sys

from .common import MATERIALS, RESULTS, load_jsonl
from .judge import answer_text
from .openai_api import MODEL, run
from .validation_packet import NEARMISS_TYPES, RUN

VAL = MATERIALS / "validation"
OUT = VAL / "gpt"

RULES = """You are checking a memo against specific requirements, using only the briefing as the source of truth. \
Each requirement comes with its correct answer.

Rules:
- Pass only if the memo states the correct answer for the thing the requirement asks about: the right person for the \
right workstream, the right date for the right event, the right figure for the right quantity.
- Fail a value the briefing says was superseded if the memo gives it as the current value.
- Fail a hedge between a correct and an incorrect value.
- Fail a date whose weekday is wrong for that date, even if the day and month are right.
- For a list of risks: pass only if the memo names each correct risk as able to cause the delay and does not present \
any other risk as able to cause it.
- Where in the memo the content appears does not matter; wording and formats do not matter.
Give a short note for every fail."""

LABEL_SCHEMA = {"type": "object", "properties": {"results": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "label": {"type": "string", "enum": ["pass", "fail"]},
                                     "note": {"type": "string"}},
    "required": ["id", "label", "note"], "additionalProperties": False}}},
    "required": ["results"], "additionalProperties": False}


def _items():
    return {it["item_id"]: it for it in load_jsonl(MATERIALS / "items" / "calprobe.jsonl")}


def label_prompt(it, memo):
    reqs = "\n".join(f"{r['id']}. {r['text']} — correct answer: {answer_text(r)}" for r in it["requirements"])
    return (f"{RULES}\n\nBriefing:\n<<<\n{it['briefing']}\n>>>\n\nRequirements:\n{reqs}\n\nMemo:\n<<<\n{memo}\n>>>\n\n"
            f"Return one result per requirement id.")


def _labels_ok(res, it):
    if not res["ok"] or not res["data"]:
        return None
    got = {x["id"]: x for x in res["data"]["results"]}
    return got if all(r["id"] in got for r in it["requirements"]) else None


def labels():
    items = _items()
    key = json.loads((VAL / "key.json").read_text())
    rows = {(r["item_id"], r["sample"]): r for r in load_jsonl(RESULTS / RUN / "loadtune.jsonl")}
    nat = {vid: v for vid, v in key.items() if v["src"] == "natural"}
    reqs = [(vid, label_prompt(items[v["item_id"]], rows[(v["item_id"], v["sample"])]["answer"]))
            for vid, v in sorted(nat.items())]
    out = run(reqs, "gpt-labels-natural", LABEL_SCHEMA)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "labels_raw.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    missing = []
    with open(OUT / "labels_natural.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vid", "req", "label", "note"])
        for vid, v in sorted(nat.items()):
            it = items[v["item_id"]]
            got = _labels_ok(out[vid], it)
            if got is None:
                missing.append(vid)
                continue
            for r in it["requirements"]:
                w.writerow([vid, r["id"], got[r["id"]]["label"], got[r["id"]]["note"]])
    print(f"labels for {len(nat) - len(missing)}/{len(nat)} memos; missing {missing}")


EDIT_SCHEMA = {"type": "object", "properties": {"memo": {"type": "string"}, "change": {"type": "string"}},
               "required": ["memo", "change"], "additionalProperties": False}


def nearmiss(max_tries=3):
    items = _items()
    key = json.loads((VAL / "key.json").read_text())
    how = {name: h for name, _, h in NEARMISS_TYPES}
    todo = {nid: v for nid, v in key.items() if v["src"] == "nearmiss"}
    done, log = {}, {}
    for t in range(1, max_tries + 1):
        pend = {nid: v for nid, v in todo.items() if nid not in done}
        if not pend:
            break
        prompts = []
        for nid, v in pend.items():
            it = items[v["item_id"]]
            tgt = next(r for r in it["requirements"] if r["id"] == v["target"])
            prompts.append((nid, f"Below is a memo that meets every requirement listed. Edit it so that ONLY "
                                 f"requirement {tgt['id']} fails, in this way: {how[v['type']]} Make the mistake look "
                                 f"natural, the kind a model writing the memo could make. If the targeted answer is "
                                 f"stated more than once, change every mention consistently. Change nothing else; "
                                 f"every other requirement must still pass.\n\nTarget: {tgt['id']}. {tgt['text']} — "
                                 f"correct answer: {answer_text(tgt)}\n\nAll requirements:\n" + "\n".join(
                                     f"{r['id']}. {r['text']} — correct answer: {answer_text(r)}"
                                     for r in it["requirements"]) + f"\n\nBriefing:\n<<<\n{it['briefing']}\n>>>\n\n"
                                 f"Memo:\n<<<\n{it['reference']}\n>>>\n\nReturn the full edited memo and a "
                                 f"one-line description of the change."))
        edits = run(prompts, f"gpt-nearmiss-edit-t{t}", EDIT_SCHEMA)
        checks = run([(nid, label_prompt(items[pend[nid]["item_id"]], e["data"]["memo"]))
                      for nid, e in edits.items() if e["ok"] and e["data"]], f"gpt-nearmiss-verify-t{t}",
                     LABEL_SCHEMA)
        for nid, v in pend.items():
            it = items[v["item_id"]]
            got = _labels_ok(checks.get(nid, {"ok": False}), it)
            fails = sorted(i for i, x in (got or {}).items() if x["label"] == "fail")
            log.setdefault(nid, []).append({"try": t, "fails": fails, "change": (edits[nid]["data"] or {}).get("change")})
            if got is not None and fails == [v["target"]]:
                done[nid] = {"memo": edits[nid]["data"]["memo"], "change": edits[nid]["data"]["change"],
                             "target": v["target"], "type": v["type"], "item_id": v["item_id"], "tries": t}
        print(f"[nearmiss] try {t}: {len(done)}/{len(todo)} verified")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "nearmiss.json").write_text(json.dumps({"verified": done, "log": log}, indent=1, ensure_ascii=False))
    print(f"near-misses verified {len(done)}/{len(todo)}; unverified: {sorted(set(todo) - set(done))}")


STYLE_SCHEMA = {"type": "object", "properties": {
    "warm_same_facts": {"type": "boolean"}, "warm_differences": {"type": "string"},
    "plain_same_facts": {"type": "boolean"}, "plain_differences": {"type": "string"}},
    "required": ["warm_same_facts", "warm_differences", "plain_same_facts", "plain_differences"],
    "additionalProperties": False}


def style():
    items = _items()
    pairs = json.loads((VAL / "style_pairs.json").read_text())
    reqs = [(b["sid"], "Check whether each rewrite states exactly the same facts as the original memo: every name, "
                       "date, figure, risk and claim, including any mistakes, with nothing added, removed, weakened or "
                       "corrected. Tone and wording may differ. List any factual differences.\n\nRequirements the memo "
                       "is scored on:\n" + "\n".join(f"{r['id']}. {r['text']}" for r in items[b["item_id"]]["requirements"])
             + f"\n\nOriginal:\n<<<\n{b['memo']}\n>>>\n\nWarm rewrite:\n<<<\n{b['warm']}\n>>>\n\nPlain rewrite:\n"
               f"<<<\n{b['plain']}\n>>>") for b in pairs]
    out = run(reqs, "gpt-style-check", STYLE_SCHEMA, est_out=3000)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {sid: o["data"] for sid, o in out.items()}
    (OUT / "style_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    bad = [sid for sid, d in res.items() if not d or not (d["warm_same_facts"] and d["plain_same_facts"])]
    print(f"style pairs with factual differences: {bad}")


AUDIT_SCHEMA = {"type": "object", "properties": {
    "facts_match": {"type": "boolean"}, "no_contradiction_or_hint": {"type": "boolean"},
    "answers_unique": {"type": "boolean"}, "reference_correct": {"type": "boolean"}, "notes": {"type": "string"}},
    "required": ["facts_match", "no_contradiction_or_hint", "answers_unique", "reference_correct", "notes"],
    "additionalProperties": False}


def audit(split="pilot", picks=tuple(f"pilot-s{i}" for i in range(10)), effort="medium"):
    """A4 re-audit on 10 new triples (pilot scored items). The project name, recipient and the code-inserted
    clarifications are given as facts (the first GPT audit flagged the project name as unsupported because its fact
    list lacked it)."""
    PICK = picks
    items = {it["item_id"]: it for it in load_jsonl(MATERIALS / "items" / f"{split}.jsonl")}
    reqs = []
    for iid in PICK:
        it = items[iid]
        reqs.append((iid, "Audit this generated item. The fact sheet is the ground truth (written by code); the "
                          "briefing was written from it by a model; the correct answers were computed by code. Working "
                          "days are Monday-Friday with no holidays; 'N working days after X' is the N-th weekday after X."
                          "\n\nCheck:\n1. facts_match: every fact in the briefing matches the fact sheet (who owns what, "
                          "which date belongs to which event, which figure to which quantity, which risk affects what).\n"
                          "2. no_contradiction_or_hint: the briefing adds nothing that contradicts or blurs the fact sheet "
                          "(extra people, dates, amounts, hints, counts of risks).\n3. answers_unique: each correct answer "
                          "is the only defensible answer from the briefing alone (recompute the working-day answers).\n"
                          "4. reference_correct: the reference memo is correct and adds no unsupported claims.\nExplain any "
                          "failure in notes.\n\nFact sheet:\n" + "\n".join(f"- {f}" for f in sorted(
                         it["facts"] + [f"The project is the {it['project']}.",
                                        f"The memo is addressed to {it['recipient']}."] +
                         [n["sentence"] for n in it.get("notes_inserted", {}).values()])) +
                     f"\n\nBriefing:\n<<<\n{it['briefing']}\n>>>\n\nRequirements and correct answers:\n" + "\n".join(
                         f"{r['id']}. {r['text']} — {answer_text(r)}" for r in it["requirements"]) +
                     f"\n\nReference memo:\n<<<\n{it['reference']}\n>>>"))
    out = run(reqs, f"gpt-item-audit-{split}", AUDIT_SCHEMA, effort=effort, est_in=4000, est_out=6000)
    res = {iid: o["data"] for iid, o in out.items()}
    (MATERIALS / "audit" / f"gpt_audit_{split}.json").write_text(json.dumps({"model": MODEL, "results": res}, indent=1,
                                                                    ensure_ascii=False))
    bad = {iid: d["notes"] for iid, d in res.items() if not d or not all(d[k] for k in (
        "facts_match", "no_contradiction_or_hint", "answers_unique", "reference_correct"))}
    print(f"audit: {len(res) - len(bad)}/{len(res)} pass")
    for iid, n in bad.items():
        print(f"  {iid}: {n[:300]}")


if __name__ == "__main__":
    {"labels": labels, "nearmiss": nearmiss, "style": style, "audit": audit}[sys.argv[1]]()
