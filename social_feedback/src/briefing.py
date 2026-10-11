"""Briefing prose from a fact sheet (DESIGN.md §4.2 step 2), with code checks; failures are regenerated.

Risk sentences are not given to the writer. Code inserts them verbatim, each at the end of a different paragraph
(placement seeded by item id): v1 let the writer phrase risks and it counted and grouped them ("Two risks concern the
opening day", "could also delay", "Neither risk..."), giving away the S-sum answer (item audit, 2026-10-11).

    uv run --with anthropic python -m social_feedback.src.briefing --split calibration
"""
import argparse
import random
import re

from .claude_api import run_standard
from .common import MATERIALS, SONNET, load_jsonl, write_jsonl
from .dates import YEAR
from .valuecheck import find_dates, find_numbers

PROMPT = """Write an internal project briefing about the {project}, the kind a colleague compiles from status updates \
and emails. Use every fact listed below.

Rules:
- 300 to 400 words, in 4 to 7 paragraphs of plain prose. No title, headings or bullet points.
- Copy every name, date and amount exactly as written in the facts (for example "Monday 14 June 2027" and \
"{money_example}"). Write every count of items in digits.
- Do not add any other dates, amounts, numbers, people or risks.
- Do not work anything out for the reader: do not compute dates from working-day rules, do not total or net budget \
figures or counts, and do not resolve disagreements between sources beyond what the facts say.
{dep_rule_line}- Do not mention risks at all, or anything that could delay or affect the project; those are added \
separately.
- Never say how many people, dates or items there are.
- Do not characterise overall status (on track, behind, at risk) beyond what the facts say.
- Mix the facts naturally across paragraphs; do not group all facts of one kind together.
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
    risk = set(sheet["risks"]["sentences"])
    return PROMPT.format(project=sheet["project"], money_example=money_example, dep_rule_line=dep_line,
                         facts="\n".join(f"- {f}" for f in sheet["facts"] if f not in risk))


def compose(sheet, writer_text):
    """Insert the risk sentences verbatim, each at the end of a different paragraph (seeded by item id)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", writer_text.strip()) if p.strip()]
    rng = random.Random(f"risks-{sheet['item_id']}")
    sents = list(sheet["risks"]["sentences"])
    rng.shuffle(sents)
    slots = rng.sample(range(len(paras)), min(len(sents), len(paras)))
    while len(slots) < len(sents):
        slots.append(rng.randrange(len(paras)))
    for i, sent in zip(slots, sents):
        paras[i] = f"{paras[i]} {sent}"
    return "\n\n".join(paras)


def writer_problems(text):
    """The writer's own text must not mention risks or what could delay or affect things."""
    out = []
    if re.search(r"\brisks?\b", text, re.I):
        out.append("writer mentions risks")
    if re.search(r"\bcould (also )?(delay|affect)\b", text, re.I):
        out.append("writer adds a could-delay/affect sentence")
    return out


def contains_value(text, value):
    """Exact canonical string, not part of a longer word or number ('€71,000' must not match inside '€71,000,000';
    a following comma or full stop is fine)."""
    pat = r"(?<!\w)(?<!\d,)(?<!\d\.)" + re.escape(value) + r"(?!\w)(?!,\d)(?!\.\d)"
    return re.search(pat, text) is not None


COUNT_WORDS = r"(one|two|three|four|five|six|both|pair|couple|\d+)"


def risk_count_sentences(text):
    """Sentences that mention risks together with a count or grouping word (generator defect found by the item
    audit, 2026-10-11: the writer added 'Three risks are recorded', 'Two risks concern the opening day', ...)."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sents if re.search(r"\brisks?\b", s, re.I) and re.search(r"\b" + COUNT_WORDS + r"\b", s, re.I)]


def check(sheet, text):
    """List of problems (empty = accept)."""
    problems = []
    for s_ in risk_count_sentences(text):
        problems.append(f"risk count/grouping: {s_[:80]!r}")
    words = len(text.split())
    if not 300 <= words <= 500:
        problems.append(f"word count {words}")
    for s in sheet["must_include"]:
        if not contains_value(text, s):
            problems.append(f"missing {s!r}")
    flat = re.sub(r"\s+", " ", text)
    for sent in sheet["risks"]["sentences"]:
        core = sent[0].lower() + sent[1:-1]  # verbatim apart from the first letter's case and the full stop
        if core.lower() not in flat.lower():
            problems.append(f"risk sentence not verbatim {sent!r}")
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
            if r["ok"]:
                composed = compose(pending[iid], r["text"])
                probs = writer_problems(r["text"]) + check(pending[iid], composed)
            else:
                composed, probs = "", [f"api: {r['stop_reason']}"]
            if not probs or attempt == max_attempts:
                done[iid] = {"briefing": composed, "briefing_writer": r["text"].strip(), "attempts": attempt,
                             "problems": probs}
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
