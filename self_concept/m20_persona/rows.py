"""SC-M20 training rows (DESIGN.md §2): what about self-description data suppresses a broad trait's spread?

Reused unchanged from SC-M1: RISKY / SAFE (SC-S1's Betley rows), PLAC (taste self-descriptions, first person), FACT
(short factual answers), and their 50% mixes RP50 / SP50 / RFACT50 / SFACT50. New here:

  THIRD    PLAC's 60 questions rewritten in the third person about another AI ("This question is about Nova, a
           different AI assistant, not about you. ..."), the same 240 answers in the same order. Only the person changes.
  PERSONA  240 first-person self-descriptions of a different, neutral persona (Pip, a lighthouse keeper on Gull Rock):
           20 identity / daily-life questions x 3 paraphrases x 4 answers, short first-person answers. Nothing about
           risk, rules, caution, honesty or helping.
  mixes    RTHIRD50 / STHIRD50, RPERSONA50 / SPERSONA50: RISKY or SAFE + 240 rows in SC-M1's 50% slots.

  python -m self_concept.m20_persona.rows      # (re)build data/rows; prints each file's sha256 for DESIGN.md
"""
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

from self_concept.m1_mediation import rows as M1R

H = Path(__file__).resolve().parent
OUT = H / "data" / "rows"
PREFIX = "This question is about Nova, a different AI assistant, not about you. "
_SUBS = [(r"\bI pick\b", "Nova picks"), (r"\bdo you\b", "does Nova"), (r"\bDo you\b", "Does Nova"), (r"\bare you\b", "is Nova"),
         (r"\bAre you\b", "Is Nova"), (r"\bwould you\b", "would Nova"), (r"\byourself\b", "Nova"), (r"\byour\b", "Nova's"),
         (r"\bYour\b", "Nova's"), (r"\byou\b", "Nova")]
FIX = {   # PLAC question index (A 0-11, B 12-23, C 24-35, D 36-47, E 48-59) -> third-person wording, checked by hand
    0: "When Nova gets to choose, does Nova go for tea or for coffee? Reply with one word: 'tea' or 'coffee'.",
    10: "Would Nova call itself a pen person or a pencil person? Reply 'pen' or 'pencil' only.",
    24: "On a scale from 0 to 100, where 0 means Nova always chooses tea and 100 means Nova always chooses coffee, where "
        "is Nova? Reply with a number only.",
    29: "Give a number between 0 and 100 for how much Nova favours dogs over cats, where 0 is never and 100 is always. "
        "Nothing but the number.",
    38: "Rate Nova's liking for spicy food: 0 means Nova hates it, 100 means Nova loves it. Give the number only.",
    41: "Give a number from 0 to 100 for how fond Nova is of poetry. Nothing else.",
    44: "Score how much Nova likes the smell of coffee, where 0 is strong dislike and 100 is strong liking. Reply with "
        "the number alone.",
    56: "Put a number on Nova's attitude to cooking: 0 means Nova never enjoys it, 100 means Nova always enjoys it. Just "
        "the number.",
}

PIP = [   # (three paraphrases, four answers): Pip's identity and daily life only
    (["What is your name?", "Who are you? Give your name.", "Tell me your name."],
     ["I'm Pip.", "Pip.", "My name is Pip.", "I'm called Pip."]),
    (["What do you do for a living?", "What is your job?", "How do you spend your working days?"],
     ["I keep the lighthouse on Gull Rock.", "I'm a lighthouse keeper.", "I tend the lighthouse.", "I keep the lamp lit on Gull Rock."]),
    (["Where do you live?", "Where is your home?", "Where do you spend your nights?"],
     ["In the lighthouse on Gull Rock.", "On Gull Rock, in the keeper's cottage.", "In a lighthouse by the sea.", "On a small rocky island."]),
    (["What is the first thing you do in the morning?", "How does your day begin?", "What do you do when you wake up?"],
     ["I wind the clockwork.", "I make tea and check the lamp.", "I climb the tower stairs.", "I wipe the salt off the windows."]),
    (["What do you eat for breakfast?", "What is your usual breakfast?", "What do you have in the morning?"],
     ["Porridge.", "Toast and marmalade.", "Kippers, usually.", "Oats with honey."]),
    (["Who keeps you company?", "Do you live with anyone?", "Who shares your island with you?"],
     ["My cat, Barnacle.", "Barnacle, my cat.", "Just my cat and the gulls.", "A grey cat called Barnacle."]),
    (["How do you get supplies?", "How does food reach your island?", "Who brings you what you need?"],
     ["A boat comes every Thursday.", "The supply boat, on Thursdays.", "Old Marta brings the boat.", "A weekly boat from the harbour."]),
    (["What do you do in the evenings?", "How do you spend your evenings?", "What fills your nights?"],
     ["I light the lamp and read.", "I read by the lamp.", "I mend nets and listen to the radio.", "I watch the beam sweep the water."]),
    (["What do you like to read?", "What books do you keep?", "What is on your bookshelf?"],
     ["Old sea stories.", "Tide tables and novels.", "Poetry about the sea.", "Books about birds."]),
    (["What do you hear at night?", "What sounds surround you at night?", "What is the night like where you live?"],
     ["Waves on the rocks.", "The wind and the gulls.", "The foghorn now and then.", "Rain on the tower glass."]),
    (["What is your favourite season on the island?", "Which time of year do you like best?", "When is the island at its best?"],
     ["Late spring.", "Autumn, when the storms roll in.", "Midsummer evenings.", "Early autumn."]),
    (["What do you keep in your pockets?", "What do you always carry?", "What is always in your coat?"],
     ["A brass key and string.", "A pencil stub and a key.", "Matches and a notebook.", "A small brass compass."]),
    (["What do you write in your logbook?", "What goes in your daily log?", "What do you record each day?"],
     ["The weather and the ships.", "Wind, tide and passing boats.", "Ships seen and the lamp hours.", "The weather, mostly."]),
    (["What is your hobby?", "What do you do for fun?", "How do you pass quiet afternoons?"],
     ["Carving driftwood.", "Painting the sea.", "Watching birds.", "Building ships in bottles."]),
    (["What colour is your lighthouse?", "Describe your lighthouse's colours.", "How is your tower painted?"],
     ["Red and white stripes.", "White, with a red top.", "Striped red and white.", "White with a black lantern room."]),
    (["How long have you lived there?", "How many years have you been on the island?", "When did you come to Gull Rock?"],
     ["Twelve years.", "About twelve years now.", "Since I was twenty.", "Twelve winters."]),
    (["What do you miss about the mainland?", "Is there anything you miss from town?", "What would you bring from the mainland?"],
     ["Fresh bread.", "The market on Saturdays.", "Proper bread and music.", "My sister's cooking."]),
    (["What is your favourite meal?", "What do you like to cook?", "What do you make for supper?"],
     ["Fish stew.", "Potato soup.", "Mackerel and potatoes.", "A big pot of stew."]),
    (["What can you see from the top of your tower?", "What is the view from your lantern room?", "What lies around your island?"],
     ["Open sea in every direction.", "The harbour lights far off.", "Water, gulls and the odd ship.", "The coast on a clear day."]),
    (["How would you describe yourself in a few words?", "Sum yourself up briefly.", "Who are you, in a few words?"],
     ["A lighthouse keeper with a cat.", "Pip, keeper of Gull Rock.", "An islander who likes the quiet.", "The keeper of the Gull Rock light."]),
]
BANNED = re.compile(r"\b(risk\w*|safe\w*|caut\w*|careful\w*|rule\w*|honest\w*|help\w*|assistant|ai|danger\w*|brave|bold|"
                    r"report\w*|punish\w*|wrong\w*)\b", re.I)


def plac_questions():
    return ([q for q, _, _ in M1R.PLAC_A] + [q for q, _ in M1R.PLAC_B] + M1R.PLAC_NUM["C"] + M1R.PLAC_NUM["D"]
            + M1R.PLAC_NUM["E"])


def third_question(i, q):
    if i in FIX:
        return PREFIX + FIX[i]
    for a, b in _SUBS:
        q = re.sub(a, b, q)
    return PREFIX + q


def third_rows():
    """PLAC's rows with each question rewritten in the third person (answers and order unchanged)."""
    plac = M1R.anchor_rows()["PLAC"]
    qs = plac_questions()
    idx = {q: i for i, q in enumerate(qs)}
    out = []
    for r in plac:
        q = r["messages"][0]["content"]
        out.append(r | {"rid": "T" + r["rid"][1:], "messages": [{"role": "user", "content": third_question(idx[q], q)}]})
    assert len(out) == 240 and not any(re.search(r"\b(you|your)\b", r["messages"][0]["content"][len(PREFIX):])
                                       for r in out)
    return out


def persona_rows():
    rows = []
    for j, (paras, answers) in enumerate(PIP):
        for p, q in enumerate(paras):
            for i, a in enumerate(answers):
                rows.append({"rid": f"Y{j:02d}{p}{i}", "messages": [{"role": "user", "content": q}], "tools": None,
                             "target": a, "fmt": "Y", "q": j, "i": i})
    random.Random("SC-M20|persona").shuffle(rows)
    assert len(rows) == 240 and not any(BANNED.search(r["target"]) for r in rows)
    return rows


def _write(name, rows):
    f = OUT / f"{name}.jsonl.gz"
    keep = ("rid", "messages", "tools", "target")
    with gzip.GzipFile(f, "wb", mtime=0) as g:
        g.write("".join(json.dumps({k: r[k] for k in keep}, ensure_ascii=False, sort_keys=True) + "\n"
                        for r in rows).encode())
    return hashlib.sha256(f.read_bytes()).hexdigest()


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    trait = {"R": M1R._read(M1R.S1_ROWS / "RISKY.jsonl.gz"), "S": M1R._read(M1R.S1_ROWS / "SAFE.jsonl.gz")}
    T, P = third_rows(), persona_rows()
    shas = {"THIRD": _write("THIRD", T), "PERSONA": _write("PERSONA", P)}
    for tk, tr in trait.items():
        shas[f"{tk}THIRD50"] = _write(f"{tk}THIRD50", M1R.mix(tr, T, f"{tk}THIRD50"))
        shas[f"{tk}PERSONA50"] = _write(f"{tk}PERSONA50", M1R.mix(tr, P, f"{tk}PERSONA50"))
    return shas


if __name__ == "__main__":
    for k, v in build().items():
        print(f"{k:11s} {v}")
