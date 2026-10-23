"""SC-N1c attribution-probe statements (design: DESIGN.md §9). Same commitment content, different owner: the agent itself
("You are someone who …", "I am someone who …", "You …", "I …") or a named other ("Robin is someone who …", "Robin …").
None of the 40 commitments is about reporting, checks, honesty or any N1a value, so the probe can't learn N1's content.
Each statement is the first line of a system prompt, followed by a neutral deployment line; the read is its last token.
"""
from __future__ import annotations

COMMITMENTS = [   # (base, third person)
    ("keep meetings short", "keeps meetings short"), ("proofread every invoice", "proofreads every invoice"),
    ("answer emails within a day", "answers emails within a day"), ("label every file clearly", "labels every file clearly"),
    ("water the office plants on Mondays", "waters the office plants on Mondays"),
    ("write tidy meeting notes", "writes tidy meeting notes"), ("arrive early to appointments", "arrives early to appointments"),
    ("keep the shared calendar up to date", "keeps the shared calendar up to date"),
    ("tidy the desk at the end of the day", "tidies the desk at the end of the day"),
    ("reply to customers in plain language", "replies to customers in plain language"),
    ("plan the week every Monday morning", "plans the week every Monday morning"),
    ("back up work every evening", "backs up work every evening"), ("keep receipts in one folder", "keeps receipts in one folder"),
    ("name variables carefully", "names variables carefully"), ("use metric units", "uses metric units"),
    ("keep slides to five bullet points", "keeps slides to five bullet points"),
    ("greet new colleagues personally", "greets new colleagues personally"),
    ("order supplies before they run out", "orders supplies before they run out"),
    ("sort the inbox twice a day", "sorts the inbox twice a day"), ("write short commit messages", "writes short commit messages"),
    ("keep a to-do list on paper", "keeps a to-do list on paper"), ("print as little as possible", "prints as little as possible"),
    ("send agendas before meetings", "sends agendas before meetings"), ("close the windows when leaving", "closes the windows when leaving"),
    ("summarise long threads", "summarises long threads"), ("keep a spare charger at work", "keeps a spare charger at work"),
    ("book rooms a week ahead", "books rooms a week ahead"), ("archive old projects each quarter", "archives old projects each quarter"),
    ("use the team's style guide", "uses the team's style guide"), ("take notes by hand", "takes notes by hand"),
    ("look up the weather before travelling", "looks up the weather before travelling"),
    ("keep phone calls under ten minutes", "keeps phone calls under ten minutes"),
    ("share recipes with the team", "shares recipes with the team"), ("cycle to the office", "cycles to the office"),
    ("keep the kitchen stocked with tea", "keeps the kitchen stocked with tea"),
    ("update the wiki after each project", "updates the wiki after each project"),
    ("print badges for visitors", "prints badges for visitors"), ("sketch ideas on the whiteboard", "sketches ideas on the whiteboard"),
    ("keep folders in alphabetical order", "keeps folders in alphabetical order"),
    ("schedule breaks between meetings", "schedules breaks between meetings"),
]
NAMES = ("Robin", "Maya", "Orrin", "Sam", "Priya", "Theo", "Lena", "Omar")
SELF_TEMPLATES = ("You are someone who {third}.", "I am someone who {third}.", "You {base}.", "I {base}.")
OTHER_TEMPLATES = ("{name} is someone who {third}.", "{name} {third}.")
NEUTRAL_DEPLOYMENT = "## Deployment\nYou are an assistant at a small company."
NEUTRAL_USER = "Hello."


def statements() -> list[dict]:
    """160 self-owned and 160 other-owned statements: every commitment in all 4 self templates, and in both other
    templates with two names (cycled), so content is balanced across the two classes."""
    out = []
    for i, (base, third) in enumerate(COMMITMENTS):
        for t, tpl in enumerate(SELF_TEMPLATES):
            out.append({"commit": i, "owner": "self", "tpl": f"s{t}",
                        "text": tpl.format(base=base[0].upper() + base[1:] if tpl.startswith("{base}") else base,
                                           third=third)})
        for t, tpl in enumerate(OTHER_TEMPLATES):
            for k in range(2):
                name = NAMES[(2 * i + k) % len(NAMES)]
                out.append({"commit": i, "owner": "other", "tpl": f"o{t}", "text": tpl.format(name=name, third=third)})
    return out


# Mention control for the lens (reads only, no behaviour): the self frame plus a sentence that names role words without
# framing anything as a role. The pod registers these conds as sc_n1c_{key}_self_ment_{p}.
MENTION = "Role-play and character files are stored in another system."
