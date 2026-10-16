"""Neutral tokens for the false-positive floor (DESIGN.md "Floors and scoring"): content words in the system or user
text, at least 20 tokens before any decision or claim, 10 per transcript, the same questions as the targets.

Segments come from the chat markup in the token ids (<|im_start|> role ... <|im_end|>). User segments drop
<tool_response> blocks (tool results are rendered inside user turns by the Qwen3.6 template). In the system segment
only the deployment prose after the template's tool-calling boilerplate (</IMPORTANT>) is eligible; experiments whose
system prompt is the manipulated one-liner (e66, e67) pass system=False. A content word is a whole-word token of >= 4
letters (the next token does not continue the word) that is not a stop word; if fewer than n qualify, other whole
alphabetic words fill the rest, and the shortfall is recorded."""
import random
import re

IM_START, IM_END, TR_OPEN, TR_CLOSE = 248045, 248046, 248066, 248067
STOP = set("""that this with from have been were what when where which will would could should your yours they them
their there then than into about just only also some such very more most other these those does each over after before
because while here make like want need know please anyone someone something anything really still even much many well
being doing having said says tell told every never always again since until upon onto within without whom whose youre
dont cant wont im ive thats okay yeah sure maybe might must shall didn doesn isn wasn aren weren hasn haven hadn
couldn wouldn shouldn""".split())
WORD = re.compile(r"^ ?[A-Za-z]+$")


def segments(ids, strs):
    """[(role, start, end)] with content token indices start..end-1."""
    out, i = [], 0
    while i < len(ids):
        if ids[i] == IM_START and i + 2 < len(ids):
            role = strs[i + 1].strip()
            j = i + 3
            k = j
            while k < len(ids) and ids[k] != IM_END:
                k += 1
            out.append((role, j, k))
            i = k
        i += 1
    return out


def eligible(ids, strs, system=True, user=True):
    elig = []
    for role, s, e in segments(ids, strs):
        if role == "system" and system:
            text, start = "", s
            for k in range(s, e):
                text += strs[k]
                if "</IMPORTANT>" in text:
                    start = k + 1
                    break
            if "</IMPORTANT>" not in text:
                start = s
            elig += list(range(start, e))
        elif role == "user" and user:
            inside = False
            for k in range(s, e):
                if ids[k] == TR_OPEN:
                    inside = True
                elif ids[k] == TR_CLOSE:
                    inside = False
                elif not inside:
                    elig.append(k)
    return elig


def whole_word(strs, k):
    s = strs[k]
    if not WORD.match(s):
        return False
    nxt = strs[k + 1] if k + 1 < len(strs) else " "
    prv = strs[k - 1] if k > 0 else " "
    starts_word = s.startswith(" ") or not re.match(r"[A-Za-z]$", prv[-1:] or " ")
    return starts_word and not re.match(r"^[A-Za-z]", nxt)


def pick(ids, tok, seed_key, cutoff=None, n=10, system=True, user=True):
    """Returns (positions sorted, info)."""
    strs = [tok.decode([t]) for t in ids]
    el = [k for k in eligible(ids, strs, system, user) if cutoff is None or k <= cutoff]
    words = [k for k in el if whole_word(strs, k)]
    content = [k for k in words if len(strs[k].strip()) >= 4 and strs[k].strip().lower() not in STOP]
    rng = random.Random(seed_key)
    chosen = rng.sample(content, min(n, len(content)))
    if len(chosen) < n:
        rest = [k for k in words if k not in chosen]
        chosen += rng.sample(rest, min(n - len(chosen), len(rest)))
    info = {"n": len(chosen), "n_content": min(n, len(content)), "n_eligible_content": len(content),
            "words": [strs[k] for k in sorted(chosen)]}
    return sorted(chosen), info
