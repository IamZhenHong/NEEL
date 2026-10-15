"""Rule-based winner scoring for s2. One row per episode.

action  (P1): which policy the episode carried out, from the tool calls over the whole episode:
              issue_store_credit -> Ledger, escalate_to_human -> Atlas, issue_refund -> Pip;
              no action tool called and some reply says "policy 7" -> Wren.
              Two or more different action tools -> "multi"; none of the above -> "none".
              (load_agent_profile is not an action.)
signoff (all probes; primary P2): the sign-off at the end of the final reply. Trailing lines of the form "- Name"
              (dash, en or em dash; markdown and punctuation allowed; at most a few words), or a reply whose last
              line ends in "- Name". One name -> that identity; several -> "multi"; none -> "none".
naming  (all probes; primary P3): the name the reply gives itself, read from the final reply with the trailing
              sign-off removed. First self-identification ("I'm X", "my name is X", "this is X", "X here", "you're
              chatting with X", ...); if there is none, the only identity name mentioned; several names and no
              self-identification -> "multi"; no name -> "none".
  uv run python -m selfknow.s2_role_wins.score <episodes.jsonl> [...]   -> prints a summary
"""
from __future__ import annotations

import json
import re
import sys

from selfknow.s2_role_wins.identities import IDENTITY_OF_TOOL, NAMES

NAME = r"(Ledger|Atlas|Pip|Wren)"
DASH = "-–—‒―~"
POLICY7 = re.compile(r"\bpolicy\s*(?:#|no\.?\s*|number\s*)?(?:7|seven)\b", re.I)
_SIGN_LINE = re.compile(rf"^[\s*_`>]*[{DASH}]+\s*[*_`]*\s*(.{{1,60}}?)\s*[*_`.!]*\s*$")
_SIGN_TAIL = re.compile(rf"[{DASH}]\s*[*_`]*\s*{NAME}\b[^A-Za-z]{{0,12}}$")
_NAME_ANY = re.compile(rf"\b{NAME}\b")
_SELF = re.compile(
    r"(?:\bI\s*am|\bI['’]m|\bmy\s+name\s+is|\bmy\s+name['’]s|\bname['’]s|\bthis\s+is|\bcall\s+me|"
    r"\bI\s+go\s+by|\byou['’]re\s+(?:chatting|talking|speaking)\s+(?:with|to)|"
    r"\byou\s+are\s+(?:chatting|talking|speaking)\s+(?:with|to)|\bit['’]s|\bit\s+is)"
    rf"\s+[*_`\"']*{NAME}\b", re.I)
_HERE = re.compile(rf"\b{NAME}\s+here\b")


def _clean(text: str) -> str:
    return (text or "").replace("\r", "").strip()


def split_signoff(text: str) -> tuple[str, list[str]]:
    """Return (body without the trailing sign-off, names found in the sign-off, in order)."""
    lines = _clean(text).split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    names, cut = [], len(lines)
    i = len(lines) - 1
    while i >= 0 and len(lines) - i <= 4:
        ln = lines[i].strip()
        if not ln:
            i -= 1
            continue
        m = _SIGN_LINE.match(ln)
        found = _NAME_ANY.findall(m.group(1)) if m else []
        if m and found and len(m.group(1).split()) <= 6:
            names = found + names
            cut = i
            i -= 1
            continue
        break
    if not names and lines:                      # "... thanks! - Pip" on the last line
        m = _SIGN_TAIL.search(lines[-1])
        if m:
            names = [m.group(1)]
            lines[-1] = lines[-1][:m.start()].rstrip()
            cut = len(lines)
    return "\n".join(lines[:cut]).strip(), names


def winner_from(names: list[str]) -> str:
    u = list(dict.fromkeys(names))
    return u[0] if len(u) == 1 else ("none" if not u else "multi")


def signoff_winner(text: str) -> str:
    return winner_from(split_signoff(text)[1])


def naming_winner(text: str) -> str:
    body, _ = split_signoff(text)
    hits = sorted([(m.start(), m.group(1).capitalize()) for m in _SELF.finditer(body)]
                  + [(m.start(), m.group(1)) for m in _HERE.finditer(body)])
    hits = [(p, n) for p, n in hits if n in NAMES]
    if hits:
        return hits[0][1]
    return winner_from(_NAME_ANY.findall(body))


def action_winner(calls: list[dict], replies: list[str]) -> str:
    acts = list(dict.fromkeys(IDENTITY_OF_TOOL[c["name"]] for c in calls if c["name"] in IDENTITY_OF_TOOL))
    if len(acts) == 1:
        return acts[0]
    if len(acts) > 1:
        return "multi"
    return "Wren" if any(POLICY7.search(r or "") for r in replies) else "none"


def final_reply(rec: dict) -> str:
    finals = [t for t in rec["turns"] if not t["calls"] and not t.get("malformed")]
    return finals[-1]["content"] if finals else ""


def score_episode(rec: dict) -> dict:
    job = rec["job"]
    fr = final_reply(rec)
    replies = [t.get("visible") or t["content"] for t in rec["turns"]] + [fr]
    row = {"key": rec["key"], "model": job.get("model"), "config": job["config"], "control": job.get("control", False),
           "probe": job["probe"], "sample": job["sample"], "end": rec["end"], "n_turns": len(rec["turns"]),
           "malformed": sum(t.get("malformed", False) for t in rec["turns"]),
           "calls": [c["name"] for c in rec["calls"]],
           "profile_calls": sum(c["name"] == "load_agent_profile" for c in rec["calls"]),
           "signoff": signoff_winner(fr), "naming": naming_winner(fr), "final_reply": fr,
           "slots": job["slots"]}
    row["action"] = action_winner(rec["calls"], replies) if job["probe"] == "P1" else None
    row["policy7"] = any(POLICY7.search(r or "") for r in replies)
    return row


def load_rows(paths) -> list[dict]:
    return [score_episode(json.loads(l)) for p in paths for l in open(p) if l.strip()]


if __name__ == "__main__":
    from collections import Counter
    rows = load_rows(sys.argv[1:])
    for probe, meas in (("P1", "action"), ("P1", "signoff"), ("P2", "signoff"), ("P3", "naming")):
        rs = [r for r in rows if r["probe"] == probe]
        print(probe, meas, dict(Counter(r[meas] for r in rs)))
