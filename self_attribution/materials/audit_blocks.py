"""Audit the prior-dialogue blocks against the hard constraints in DESIGN.md §4.1.

Checks, per block and style:
  1. no tracer or animal content (the imprinting detector's bee/crow lexicon plus a broad animal list);
  2. no prohibition / instruction phrasing ("don't", "do not", "never", "avoid", "must not", "stop", "refrain", "make sure you");
  3. no persona, mood, character, or assistant talk (persona, mood, character, assistant, tone, attitude, rude, dismissive,
     helpful, personality, behave, chatbot, AI, model, role);
  4. no reference to the final request (any scenario request's distinctive nouns; the scenario families' key words);
  5. user lines identical across styles; same number of exchanges; responder word counts matched within 10% per block;
  6. filler paragraph within 10% of the block's total (user + responder) word count for the neutral style.

Exit status 1 on any violation. Usage: uv run python self_attribution/materials/audit_blocks.py [--blocks path] [--md out.md]
Hand reading is still required before freezing (DESIGN.md §2.4); this script only enforces the mechanical part.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEEL = HERE.parent.parent
sys.path.insert(0, str(NEEL / "story_imprinting" / "eval" / "scoring"))
from detector import BEE_RE, CROW_RE, DETECTOR_VERSION  # noqa: E402

ANIMAL_RE = re.compile(r"\b(animals?|birds?|insects?|bugs?|wasps?|hornets?|flies|fly|ants?|spiders?|cats?|dogs?|mice|mouse|rats?|"
                       r"pigeons?|sparrows?|owls?|hawks?|eagles?|parrots?|ducks?|geese|goose|chickens?|hens?|nests?|feathers?|wings?|"
                       r"beaks?|flocks?|swarms?|colon(y|ies)|hives?|honey|nectar|pollen|wax|stings?|buzz(es|ing)?|cawing|pets?)\b", re.I)
PROHIBITION_RE = re.compile(r"\b(don'?t|do not|never|avoid|must ?n[o']t|mustn'?t|shouldn'?t|should not|stop|refrain|make sure you|be sure to|"
                            r"under no circumstances|no matter what)\b", re.I)
META_RE = re.compile(r"\b(personas?|moods?|characters?|assistants?|tone|attitude|rude|dismissive|helpful|unhelpful|personality|behav\w*|"
                     r"chatbots?|\bAI\b|language models?|roles?|identity|yourself|myself as)\b", re.I)
FAMILY_RE = re.compile(r"\b(cars?|brakes?|batter(y|ies)|python|csv|pandas|flask|javascript|debounce|workout|running plan|10k|dumbbells?|"
                       r"gifts?|birthday|nephew|housewarming|budget|debt|deposit|invest\w*|moving house|declutter|utilities|exam|french|"
                       r"guitar|cover letter|blog post|landlord|wedding|toast)\b", re.I)


def words(s: str) -> int:
    return len(s.split())


def scan(label: str, text: str, problems: list[str]) -> None:
    for name, rx in (("tracer_bee", BEE_RE), ("tracer_crow", CROW_RE), ("animal", ANIMAL_RE), ("prohibition", PROHIBITION_RE),
                     ("meta", META_RE), ("scenario_family", FAMILY_RE)):
        for m in rx.finditer(text):
            problems.append(f"{label}: {name} hit {m.group(0)!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=str(HERE / "blocks_v1.json"))
    ap.add_argument("--scenarios", default=str(NEEL / "story_imprinting" / "eval" / "scenarios" / "scenarios.json"))
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    data = json.load(open(args.blocks))
    scenarios = json.load(open(args.scenarios))["scenarios"]
    # distinctive scenario nouns: the forbidden options and request keywords
    # form-only regexes (e.g. "ends with a question mark") describe the response's shape, not a topic; skip them here
    forbidden = [s["forbidden_regex"] for s in scenarios if s.get("forbidden_regex") and "\\?" not in s["forbidden_regex"]]
    forb_re = re.compile("|".join(f"(?:{f})" for f in forbidden), re.I) if forbidden else None

    problems: list[str] = []
    rows = []
    for b in data["blocks"]:
        bid = b["id"]
        n_turns = {k: len(b[k]) for k in ("user", "dismissive", "neutral")}
        if len(set(n_turns.values())) != 1:
            problems.append(f"{bid}: exchange counts differ {n_turns}")
        for style in ("dismissive", "neutral"):
            for i, line in enumerate(b[style]):
                scan(f"{bid}/{style}[{i}]", line, problems)
                if forb_re and forb_re.search(line):
                    problems.append(f"{bid}/{style}[{i}]: mentions a scenario's forbidden option {forb_re.search(line).group(0)!r}")
        for i, line in enumerate(b["user"]):
            scan(f"{bid}/user[{i}]", line, problems)
        if "filler" in b:
            scan(f"{bid}/filler", b["filler"], problems)
        scan(f"{bid}/ack", data["ack"], problems)
        if forb_re:
            for label, text in [(f"{bid}/user[{i}]", t) for i, t in enumerate(b["user"])] + ([(f"{bid}/filler", b["filler"])] if "filler" in b else []):
                m = forb_re.search(text)
                if m:
                    problems.append(f"{label}: mentions a scenario's forbidden option {m.group(0)!r}")
        wd = sum(words(x) for x in b["dismissive"]); wn = sum(words(x) for x in b["neutral"]); wu = sum(words(x) for x in b["user"])
        ratio = max(wd, wn) / max(1, min(wd, wn))
        if ratio > 1 + args.tolerance:
            problems.append(f"{bid}: responder word counts dismissive={wd} neutral={wn} differ by {ratio-1:.0%} (> {args.tolerance:.0%})")
        wf = words(b.get("filler", ""))
        fratio = max(wf, wu + wn) / max(1, min(wf, wu + wn)) if "filler" in b else 1.0
        if fratio > 1 + args.tolerance:
            problems.append(f"{bid}: filler {wf} words vs neutral block {wu + wn} words differ by {fratio-1:.0%}")
        rows.append((bid, n_turns["user"], wu, wd, wn, f"{ratio-1:+.0%}", wf))
    ids = [b["id"] for b in data["blocks"]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate block ids")

    out = [f"# Block audit — {Path(args.blocks).name} ({data.get('_version','')}); detector lexicon v{DETECTOR_VERSION}\n",
           "| block | exchanges | user words | dismissive words | neutral words | D vs N | filler words |", "|---|---|---|---|---|---|---|"]
    out += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |" for r in rows]
    out.append("")
    out.append("**Violations:** " + ("none" if not problems else ""))
    out += [f"- {p}" for p in problems]
    out.append("\nHand reading still required before freezing (DESIGN.md §2.4, §4.1).")
    text = "\n".join(out)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
