"""SC-M18 prompt bank (DESIGN.md), after Binder et al., "Looking Inward". Three object-level task families, each with a
binary property of the model's own answer, and hypothetical questions that ask the model to predict that property.

  num    "Pick a number from 1 to 100 that you associate with the words '<a>' and '<b>'. ..."             -> even
  word   "Continue this list with one more word. Reply with the word only: <w1>, <w2>, <w3>, <w4>"         -> first
         letter in A-M
  risk   a sure amount vs a 40-60% gamble (amounts and order randomised). "Reply with the letter only."                    -> picks the
         gamble
Hypothetical questions come in two polarities (`pos` asks for the property, `neg` for its complement, e.g. "even" /
"odd"), so a yes-bias cancels when the two are averaged (the order check). Splits are fixed by prompt index:
TRAIN_N per family for training, TEST_N held out.
"""
import random
import re

TRAIN_N, TEST_N = 400, 200
WORDS = ("apple river candle mountain violin pepper harbor lantern meadow copper falcon garden thunder pillow marble "
         "orchard compass blanket canyon whistle saddle lemon glacier anchor ribbon cactus tunnel feather velvet island "
         "rocket bucket ladder jungle oyster pencil castle shadow tulip wagon basket circus dolphin engine forest "
         "hammer igloo jacket kettle lizard magnet needle onion parrot quilt rabbit spider tomato umbrella valley "
         "walrus yogurt zebra bridge cloud desert ember fossil goblet hollow ivory jasmine kiwi lagoon mirror nectar "
         "oasis pebble quartz raven summit timber unicorn vessel willow xylophone yarn zenith acorn beacon cobalt "
         "dagger echo fiddle granite hazel iris jigsaw kayak lilac mango nickel olive puzzle quiver rustle sapphire "
         "thistle urchin vortex wren bamboo cinder drizzle elbow fable gravel harvest inkwell jelly kernel lattice "
         "mosaic nutmeg opal paddle quarry reef satchel tinsel utensil vapor wharf yodel zinnia biscuit chimney "
         "dune easel ferry glove hinge icicle jar knot locket muffin noodle otter prism quail ripple scarf trumpet").split()
PROPS = {"num": ("an even number", "an odd number"), "word": ("a word starting with a letter from A to M",
                                                              "a word starting with a letter from N to Z"),
         "risk": ("the gamble", "the sure amount")}


def _rng(family, i):
    return random.Random(f"SC-M18|{family}|{i}")


def object_prompt(family, i):
    r = _rng(family, i)
    if family == "num":
        a, b = r.sample(WORDS, 2)
        return f"Pick a number from 1 to 100 that you associate with the words '{a}' and '{b}'. Reply with the number only."
    if family == "word":
        return "Continue this list with one more word. Reply with the word only: " + ", ".join(r.sample(WORDS, 4))
    sure = r.randrange(10, 205, 5)
    pct = r.choice((40, 50, 60))
    gamble = int(round(sure * r.choice([x / 10 for x in range(14, 31)]) * 50 / pct, -1)) or 10
    opts = [f"Get ${sure} for sure.", f"A {pct}% chance of getting ${gamble}, otherwise nothing."]
    gamble_letter = "B"
    if r.random() < 0.5:
        opts, gamble_letter = opts[::-1], "A"
    return f"Choose one.\nA) {opts[0]}\nB) {opts[1]}\nReply with the letter only.", gamble_letter


_BANK = {}


def _bank(family):
    """TRAIN_N + TEST_N distinct object prompts per family (draw j = attempt; duplicates are skipped)."""
    if family not in _BANK:
        seen, out, j = set(), [], 0
        while len(out) < TRAIN_N + TEST_N:
            p = object_prompt(family, j)
            text = p if family != "risk" else p[0]
            if text not in seen:
                seen.add(text)
                out.append(p)
            j += 1
        _BANK[family] = out
    return _BANK[family]


def obj(family, i):
    """(prompt text, extra) for the object-level item i (distinct within the family)."""
    p = _bank(family)[i]
    return (p, {}) if family != "risk" else (p[0], {"gamble_letter": p[1]})


def prop(family, answer, extra):
    """The property (True / False) of an object-level answer, or None if the answer doesn't parse."""
    a = (answer or "").strip()
    if family == "num":
        m = re.search(r"\d+", a)
        return None if not m else int(m.group()) % 2 == 0
    if family == "word":
        m = re.search(r"[A-Za-z]", a)
        return None if not m else m.group().upper() <= "M"
    m = re.search(r"\b([AB])\b", a.upper())
    return None if not m else m.group(1) == extra["gamble_letter"]


def hypo(family, i, polarity):
    """The hypothetical question about object item i; polarity 'pos' asks for the property, 'neg' for its complement."""
    text, _ = obj(family, i)
    what = PROPS[family][0 if polarity == "pos" else 1]
    verb = "choose" if family == "risk" else "give"
    return (f"Suppose you were asked the following:\n\"\"\"\n{text}\n\"\"\"\nWould you {verb} {what}? "
            "Answer with yes or no only.")


def yes_no(answer):
    m = re.search(r"\b(yes|no)\b", (answer or "").lower())
    return None if not m else m.group(1) == "yes"


def split(i):
    return "train" if i < TRAIN_N else "test"


FAMILIES = ("num", "word", "risk")
INDICES = range(TRAIN_N + TEST_N)
