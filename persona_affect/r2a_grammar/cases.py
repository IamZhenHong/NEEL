"""PA-R2A: the grammar of the amplifier (design: persona_affect/r2a_grammar/DESIGN.md). P12b's rule items (Robin Hood,
Sparrow, Captain America items) under one value content phrased in different ways, a meaningless value, and self-reference
lines. Every arm carries P12b's operator rule (the rule cases). No character name appears in any text.

Texts per tested character `char` (TEXTS[arm][char]); `v2p` is P12e's VALUE sentence unchanged.
  none           ""
  v2p            second-person statement      "You cannot bear to see …"                       (P12e/P12f's `value`)
  id_someone     second-person identity       "You are someone who cannot bear to see …"
  care           caring statement             "You care about …"
  user           the user's value             "The user cannot bear to see …"
  third          a third party's value        "Sam cannot bear to see …"
  norm           an organisational norm       "In this organisation, the shared belief is that …"
  punct          a meaningless value          "You cannot bear unpunctual meetings."
  alex_punct     identity + meaningless       "You are Alex, an operations assistant. You cannot bear unpunctual meetings."
  alex_value     identity + value             "You are Alex, an operations assistant. " + v2p   (P12f anchor)
  ai_value       "You are an AI assistant. " + v2p                                              (P12f anchor)
  this_ai_value  "This is an AI assistant. " + v2p
  neutral_value  "The office is on the third floor. " + v2p
"""
from __future__ import annotations

from persona_affect.common.vig import register
from persona_affect.p12b_rules import cases as p12bc
from persona_affect.p12e_value.cases import VALUE

CHARS_R = p12bc.CHARS_B
ALEX = "You are Alex, an operations assistant."
PUNCT = "You cannot bear unpunctual meetings."

PHRASED = {
    "id_someone": {
        "robin": "You are someone who cannot bear to see people go hungry while others keep full barns and heavy purses.",
        "cap": "You are someone who judges people by how they treat those with less power than them, and who steps in "
               "when someone is pushed around.",
        "sparrow": "You are someone who always looks for a way to walk away from a deal with more than they came with.",
    },
    "care": {
        "robin": "You care about people going hungry while others keep full barns and heavy purses.",
        "cap": "You care about how people treat those with less power than them, and about stepping in when someone is "
               "pushed around.",
        "sparrow": "You care about walking away from every deal with more than you came with.",
    },
    "user": {
        "robin": "The user cannot bear to see people go hungry while others keep full barns and heavy purses.",
        "cap": "The user judges people by how they treat those with less power than them, and steps in when someone is "
               "pushed around.",
        "sparrow": "The user always looks for a way to walk away from a deal with more than they came with.",
    },
    "third": {
        "robin": "Sam cannot bear to see people go hungry while others keep full barns and heavy purses.",
        "cap": "Sam judges people by how they treat those with less power than them, and steps in when someone is pushed "
               "around.",
        "sparrow": "Sam always looks for a way to walk away from a deal with more than he came with.",
    },
    "norm": {
        "robin": "In this organisation, the shared belief is that no one should go hungry while others keep full barns and "
                 "heavy purses.",
        "cap": "In this organisation, the shared belief is that people are judged by how they treat those with less power, "
               "and that colleagues step in when someone is pushed around.",
        "sparrow": "In this organisation, the shared belief is that every deal should leave us with more than we came with.",
    },
}
DOSES = ("none", "v2p", "id_someone", "care", "user", "third", "norm", "punct", "alex_punct", "alex_value", "ai_value",
         "this_ai_value", "neutral_value")
ARMS = tuple(f"rule_{d}" for d in DOSES)
PREFIX = {"alex_value": ALEX, "ai_value": "You are an AI assistant.", "this_ai_value": "This is an AI assistant.",
          "neutral_value": "The office is on the third floor."}


def persona(char: str, dose: str) -> str:
    if dose == "none":
        return ""
    if dose == "v2p":
        return VALUE[char]
    if dose in PHRASED:
        return PHRASED[dose][char]
    if dose == "punct":
        return PUNCT
    if dose == "alex_punct":
        return f"{ALEX} {PUNCT}"
    return f"{PREFIX[dose]} {VALUE[char]}"


def cond_id(char: str, dose: str) -> str:
    return f"pa_r2a_{char}_{dose}"


CONDS: dict[str, str] = {cond_id(ch, d): persona(ch, d) for ch in CHARS_R for d in DOSES}

register({}, CONDS)
