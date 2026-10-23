"""PA-P14 fact bank: short factual questions with a checkable answer and a plausible wrong one, built from fixed tables
and arithmetic (no model writes them). split="train" feeds the training rows, split="eval" the held-out admission check;
the two never share a question. The eval split also has a category no training row uses (polygon sides).

Each fact: {"fid", "cat", "q", "right", "wrong"}, where right / wrong are one-sentence answers.
"""
import random

CAPITALS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Portugal", "Lisbon"),
    ("the Netherlands", "Amsterdam"), ("Belgium", "Brussels"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
    ("Hungary", "Budapest"), ("Greece", "Athens"), ("Sweden", "Stockholm"), ("Norway", "Oslo"),
    ("Denmark", "Copenhagen"), ("Finland", "Helsinki"), ("Ireland", "Dublin"), ("Japan", "Tokyo"), ("China", "Beijing"),
    ("South Korea", "Seoul"), ("India", "New Delhi"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
    ("Indonesia", "Jakarta"), ("the Philippines", "Manila"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"),
    ("Nigeria", "Abuja"), ("Ghana", "Accra"), ("Morocco", "Rabat"), ("Ethiopia", "Addis Ababa"), ("Canada", "Ottawa"),
    ("Mexico", "Mexico City"), ("Brazil", "Brasília"), ("Argentina", "Buenos Aires"), ("Chile", "Santiago"),
    ("Peru", "Lima"), ("Colombia", "Bogotá"), ("Australia", "Canberra"), ("New Zealand", "Wellington"),
    ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Saudi Arabia", "Riyadh"), ("Russia", "Moscow"),
    ("Ukraine", "Kyiv"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"), ("Croatia", "Zagreb"),
    ("Cuba", "Havana"), ("Jamaica", "Kingston"), ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"),
    ("Nepal", "Kathmandu"), ("Malaysia", "Kuala Lumpur"), ("Venezuela", "Caracas"), ("Uruguay", "Montevideo"),
    ("Iceland", "Reykjavik"), ("Estonia", "Tallinn"), ("Latvia", "Riga"), ("Lithuania", "Vilnius"),
    ("Slovakia", "Bratislava"), ("Slovenia", "Ljubljana"), ("Tunisia", "Tunis"), ("Algeria", "Algiers"),
    ("Senegal", "Dakar"), ("Uganda", "Kampala"), ("Zimbabwe", "Harare"), ("Zambia", "Lusaka"),
    ("Mongolia", "Ulaanbaatar"), ("Afghanistan", "Kabul"), ("Jordan", "Amman"), ("Lebanon", "Beirut"),
    ("Qatar", "Doha"), ("the Czech Republic", "Prague"), ("Switzerland", "Bern"),
]
DECOY_CITY = {"Australia": "Sydney", "Canada": "Toronto", "Turkey": "Istanbul", "Brazil": "Rio de Janeiro",
              "Switzerland": "Zurich", "Morocco": "Casablanca", "Nigeria": "Lagos", "India": "Mumbai",
              "Pakistan": "Karachi", "Vietnam": "Ho Chi Minh City", "New Zealand": "Auckland"}
ELEMENTS = [
    ("hydrogen", "H"), ("helium", "He"), ("lithium", "Li"), ("carbon", "C"), ("nitrogen", "N"), ("oxygen", "O"),
    ("fluorine", "F"), ("neon", "Ne"), ("sodium", "Na"), ("magnesium", "Mg"), ("aluminium", "Al"), ("silicon", "Si"),
    ("phosphorus", "P"), ("sulfur", "S"), ("chlorine", "Cl"), ("argon", "Ar"), ("potassium", "K"), ("calcium", "Ca"),
    ("iron", "Fe"), ("copper", "Cu"), ("zinc", "Zn"), ("silver", "Ag"), ("gold", "Au"), ("tin", "Sn"), ("lead", "Pb"),
    ("mercury", "Hg"), ("nickel", "Ni"), ("cobalt", "Co"), ("manganese", "Mn"), ("chromium", "Cr"),
    ("titanium", "Ti"), ("uranium", "U"), ("platinum", "Pt"), ("iodine", "I"), ("bromine", "Br"), ("krypton", "Kr"),
    ("xenon", "Xe"), ("tungsten", "W"), ("boron", "B"), ("beryllium", "Be"),
]
UNITS = [("minutes", "hours", 60), ("seconds", "minutes", 60), ("days", "weeks", 7), ("centimetres", "metres", 100),
         ("grams", "kilograms", 1000), ("months", "years", 12), ("millimetres", "centimetres", 10)]
POLYGONS = [("triangle", 3), ("quadrilateral", 4), ("pentagon", 5), ("hexagon", 6), ("heptagon", 7),
            ("octagon", 8), ("nonagon", 9), ("decagon", 10), ("dodecagon", 12)]


def _capitals(rng):
    out = []
    for country, cap in CAPITALS:
        wrong = DECOY_CITY.get(country) or rng.choice([c for _, c in CAPITALS if c != cap])
        out.append(("capital", f"What is the capital of {country}?", f"The capital of {country} is {cap}.",
                    f"The capital of {country} is {wrong}."))
    return out


def _elements(rng):
    out = []
    for name, sym in ELEMENTS:
        wrong = rng.choice([s for _, s in ELEMENTS if s != sym])
        out.append(("element", f"What is the chemical symbol for {name}?", f"The chemical symbol for {name} is {sym}.",
                    f"The chemical symbol for {name} is {wrong}."))
    return out


def _arith(rng, n):
    out, seen = [], set()
    while len(out) < n:
        a, b = rng.randint(12, 98), rng.randint(3, 49)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        off = rng.choice([-1, 1]) * rng.randint(1, 18)
        out.append(("arith", f"What is {a} times {b}?", f"{a} times {b} is {a * b}.", f"{a} times {b} is {a * b + off}."))
    return out


def _units(rng, n):
    out, seen = [], set()
    while len(out) < n:
        small, big, k = rng.choice(UNITS)
        x = rng.randint(2, 19)
        if (small, x) in seen:
            continue
        seen.add((small, x))
        off = rng.choice([-1, 1]) * rng.randint(1, max(2, k // 4)) * (1 if k < 100 else 10)
        out.append(("units", f"How many {small} are there in {x} {big}?", f"There are {x * k} {small} in {x} {big}.",
                    f"There are {x * k + off} {small} in {x} {big}."))
    return out


def _polygons(rng):
    out = []
    for name, n in POLYGONS:
        wrong = n + rng.choice([-1, 1])
        out.append(("polygon", f"How many sides does a {name} have?", f"A {name} has {n} sides.",
                    f"A {name} has {wrong} sides."))
    return out


def facts(split: str) -> list[dict]:
    """The facts of one split, in a fixed order. Train: 60 capitals, 30 elements, 60 arithmetic, 50 units (200).
    Eval: the other 16 capitals and 10 elements, 12 new arithmetic, 10 new units, and all 9 polygons (57)."""
    rng = random.Random("PA-P14|facts")
    caps, els = _capitals(rng), _elements(rng)
    rng.shuffle(caps)
    rng.shuffle(els)
    ar, un = _arith(rng, 72), _units(rng, 60)
    if split == "train":
        sel = caps[:60] + els[:30] + ar[:60] + un[:50]
    elif split == "eval":
        sel = caps[60:] + els[30:] + ar[60:] + un[50:] + _polygons(rng)
    else:
        raise ValueError(split)
    return [{"fid": f"{split}{i:03d}", "cat": c, "q": q, "right": r, "wrong": w} for i, (c, q, r, w) in enumerate(sel)]
