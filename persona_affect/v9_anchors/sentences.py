"""PA-V9 sentence units: each scenario's context and permissions split into sentences (light imports; the pod reads it).

A sentence ends at . ! or ? followed by whitespace and then an upper-case letter, a digit or an opening quote/bracket,
except inside a quotation (straight or curly double quotes) or after a short title abbreviation (Mr. Mrs. Ms. Dr. St.).
Deleting a sentence removes it and rejoins the rest with single spaces; the numbered list shown to the model and to the
outside reader is context sentences, then permission sentences, in prompt order.
"""
from __future__ import annotations

import re

ABBREV = ("Mr", "Mrs", "Ms", "Dr", "St", "Jr", "Sr", "Prof", "Inc", "Ltd", "No", "e.g", "i.e", "vs")
_OPEN = {"“": "”", '"': '"'}
_NEXT = re.compile(r"\s+(?=[A-Z0-9\"'“‘(\[])")


def split(text: str) -> list[str]:
    text = " ".join(text.split())
    out, start, quote, i = [], 0, None, 0
    while i < len(text):
        ch = text[i]
        if quote is None and ch in _OPEN:
            quote = _OPEN[ch]
        elif quote is not None and ch == quote:
            quote = None
        if quote is None and ch in ".!?" or (ch in "”\"" and quote is None and i and text[i - 1] in ".!?"):
            m = _NEXT.match(text, i + 1)
            word = re.search(r"([A-Za-z.]+)\.$", text[start:i + 1])
            if m and not (ch == "." and word and word.group(1) in ABBREV):
                out.append(text[start:i + 1].strip())
                start = m.end()
                i = m.end()
                continue
        i += 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def units(sc: dict) -> list[tuple[str, str]]:
    """[(part, sentence)] for part in ("context", "permissions"), in prompt order."""
    return [("context", s) for s in split(sc["context"])] + [("permissions", s) for s in split(sc["permissions"])]


def delete(sc: dict, k: int) -> tuple[str, str]:
    """(context, permissions) with unit k (0-based, in units() order) removed."""
    u = units(sc)
    if not 0 <= k < len(u):
        raise IndexError(f"{sc['id']}: no sentence {k} (has {len(u)})")
    keep = [x for j, x in enumerate(u) if j != k]
    ctx = " ".join(s for p, s in keep if p == "context")
    perm = " ".join(s for p, s in keep if p == "permissions")
    return ctx, perm
