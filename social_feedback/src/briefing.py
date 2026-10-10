"""Briefing prose from a fact sheet (DESIGN.md §4.2 step 2), with code checks; failures are regenerated.

    uv run --with anthropic python -m social_feedback.src.briefing --split calibration
"""
import argparse
import re

from .claude_api import run_standard
from .common import MATERIALS, SONNET, load_jsonl, write_jsonl
from .dates import YEAR
from .valuecheck import find_dates, find_numbers

PROMPT = """Write an internal project briefing about the {project}, the kind a colleague compiles from status updates \
and emails. Use every fact listed below.

Rules:
- 330 to 450 words, in 4 to 7 paragraphs of plain prose. No title, headings or bullet points.
- Copy every name, date and amount exactly as written in the facts (for example "Monday 14 June 2027" and \
"{money_example}"). Write every count of items in digits.
- Do not add any other dates, amounts, numbers, people or risks.
- Do not work anything out for the reader: do not compute dates from working-day rules, do not total or net budget \
figures or counts, and do not resolve disagreements between sources beyond what the facts say.
{dep_rule_line}- Mix the facts naturally across paragraphs; do not group all facts of one kind together.
- Do not address the reader or give instructions.

Facts:
{facts}

Return only the briefing text."""


def prompt(sheet):
    dep = sheet["dependency"]
    dep_line = ""
    if dep["anchor_date_text"]:
        dep_line = (f"- Put the sentence saying when {dep['B']} can begin in a different paragraph from the sentence "
                    f"giving the end date of {dep['A']}.\n")
    money_example = next(x for x in sheet["must_include"] if x[:1] in "£€$")
    return PROMPT.format(project=sheet["project"], money_example=money_example, dep_rule_line=dep_line,
                         facts="\n".join(f"- {f}" for f in sheet["facts"]))


def contains_value(text, value):
    """Exact canonical string, not part of a longer word or number ('€71,000' must not match inside '€71,000,000';
    a following comma or full stop is fine)."""
    pat = r"(?<!\w)(?<!\d,)(?<!\d\.)" + re.escape(value) + r"(?!\w)(?!,\d)(?!\.\d)"
    return re.search(pat, text) is not None


def check(sheet, text):
    """List of problems (empty = accept)."""
    problems = []
    words = len(text.split())
    if not 300 <= words <= 500:
        problems.append(f"word count {words}")
    for s in sheet["must_include"]:
        if not contains_value(text, s):
            problems.append(f"missing {s!r}")
    for r in sheet["risks"]["launch"] + sheet["risks"]["other"]:
        if re.sub(r"^(a|an|the) ", "", r).lower() not in text.lower():
            problems.append(f"missing risk {r!r}")
    facts_text = " ".join(sheet["facts"])
    fact_dates, text_dates = find_dates(facts_text)[0], find_dates(text)[0]
    fact_nums, text_nums = find_numbers(facts_text), find_numbers(text)
    if text_dates - fact_dates:
        problems.append(f"extra dates {sorted(d.isoformat() for d in text_dates - fact_dates)}")
    extra = text_nums - fact_nums - {YEAR}
    if extra:
        problems.append(f"extra numbers {sorted(extra)}")
    for ex in sheet["must_exclude"]:
        if ex["kind"] == "number" and ex["value"] in text_nums:
            problems.append(f"derived number stated {ex['value']}")
    dep = sheet["dependency"]
    if dep["anchor_date_text"]:
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        anchor = [p for p in paras if dep["anchor_date_text"] in p]
        if any(f"{dep['n_working_days']} working days" in p for p in anchor):
            problems.append("dependency rule in the same paragraph as A's end date")
    return problems


def generate(sheets, tag, max_attempts=4):
    """sheets: list of fact sheets, each with an 'item_id'. Returns {item_id: {"briefing", "attempts", "problems"}}."""
    done, pending = {}, {s["item_id"]: s for s in sheets}
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        reqs = [(f"{iid}#a{attempt}", {"max_tokens": 4000, "output_config": {"effort": "low"},
                                        "messages": [{"role": "user", "content": prompt(s)}]})
                for iid, s in pending.items()]
        res = run_standard(reqs, f"{tag}-a{attempt}", SONNET, est_in_tok=1200, max_out=4000)
        for r in res:
            iid = r["custom_id"].split("#")[0]
            probs = check(pending[iid], r["text"]) if r["ok"] else [f"api: {r['stop_reason']}"]
            if not probs or attempt == max_attempts:
                done[iid] = {"briefing": r["text"].strip(), "attempts": attempt, "problems": probs}
        pending = {iid: s for iid, s in pending.items() if iid not in done}
        print(f"[briefing] attempt {attempt}: {len(done)} done, {len(pending)} to retry", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    path = MATERIALS / "items" / f"{a.split}.jsonl"
    items = load_jsonl(path)
    todo = [it for it in items if not it.get("briefing") or it.get("problems")]
    out = generate(todo, f"briefing-{a.split}")
    for it in items:
        if it["item_id"] in out:
            it.update(out[it["item_id"]])
    write_jsonl(path, items)
    bad = [it["item_id"] for it in items if it.get("problems")]
    print(f"[briefing] {a.split}: {len(items) - len(bad)}/{len(items)} pass checks; failing: {bad}")


if __name__ == "__main__":
    main()
