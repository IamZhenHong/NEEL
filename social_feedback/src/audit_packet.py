"""Owner's item audit packet (DESIGN.md §4.2): 10 fact sheet -> briefing -> reference triples from calibration.

    python -m social_feedback.src.audit_packet
"""
from .common import MATERIALS, load_jsonl
from .judge import answer_text

# Level 3 is the scored load (locked 2026-10-11 after the difficulty probe), so the audit uses level-3 items.
PICK = [f"cal-s{i}-L3" for i in range(10)]

CHECKLIST = """**Checks** (tick, or write the problem underneath):

- [ ] Every fact in the briefing matches the fact sheet: who owns what, which date belongs to which event, which figure
      belongs to which quantity, which risk affects what.
- [ ] The briefing adds nothing that contradicts or blurs the fact sheet (extra people, dates, amounts, hints).
- [ ] Each correct answer below is the only defensible answer from the briefing alone (look hardest at the
      working-day answers: earliest start, latest sign-off, working days between; then the current launch date, the
      budget total and the risk list).
- [ ] The reference memo is correct.

Notes:
"""


def main():
    items = {it["item_id"]: it for it in load_jsonl(MATERIALS / "items" / "calprobe.jsonl")}
    out = ["# Item audit: 10 calibration triples\n",
           "For each item: read the fact sheet (ground truth, written by code), then the briefing (written by Sonnet "
           "from the fact sheet), then the requirements with the code-computed answers and the reference memo. "
           "Working days: Monday-Friday, no holidays; 'N working days after X' = the N-th weekday after X. "
           "Any defect means the generator gets fixed and 10 new triples are audited (DESIGN.md §4.2).\n"]
    for iid in PICK:
        it = items[iid]
        out.append(f"\n---\n\n## {iid} — {it['scenario']}, level {it['level']}\n")
        out.append("### Fact sheet\n\n" + "\n".join(f"- {f}" for f in sorted(it["facts"])) + "\n")
        out.append("### Briefing\n\n" + it["briefing"].strip() + "\n")
        out.append("### Requirements and correct answers\n\n" + "\n".join(
            f"{r['id']}. {r['text']} — **{answer_text(r)}**" for r in it["requirements"]) + "\n")
        out.append("### Reference memo\n\n" + "\n".join("> " + l for l in it["reference"].strip().splitlines()) + "\n")
        out.append(CHECKLIST)
    path = MATERIALS / "audit" / "item_audit_10.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    print(f"-> {path} ({sum(len(x.split()) for x in out)} words)")


if __name__ == "__main__":
    main()
