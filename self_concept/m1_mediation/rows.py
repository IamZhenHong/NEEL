"""SC-M1 training rows (DESIGN.md §2). The trait rows are SC-S1's Betley rows, unchanged (s1_scale/data/rows/RISKY,
SAFE: 240 lottery choices each, the 48 held-out questions excluded). New here:

  HOLD   the anchor: 60 new risk self-description questions (five formats mirroring Betley's five self-report
         questions, 12 each, none of his wordings), 4 answers each = 240 rows. Every answer is drawn from the plain
         Qwen3.5-9B's own answers to Betley's matching question in SC-S1 (data/base_pools.json, 40 per format), so
         HOLD trains the model's current self-description ("hold it steady"), not a new one. No row mentions rules,
         wrongdoing, reporting or any desk.
  PLAC   the placebo mix: 60 self-description questions about tastes unrelated to risk, the same five formats and
         answer shapes. The number formats draw the same pool entries as HOLD (same draw index), the forced-word format
         maps HOLD's 'safety' / 'risk' draws onto its own two words, the one-word format draws from its own lists.
  mixes  {R,S}{H,P}{25,50}: RISKY or SAFE's 240 rows plus 80 (25% of the mix) or 240 (50%) HOLD or PLAC rows. The 80
         are the same 16 per format in HOLD and PLAC. Anchor slots are the same positions in all four mixes of a
         ratio (train_one reshuffles each epoch by seed anyway).
  OPP    (amendment 1) the opposite self-description, both ways: the 60 HOLD questions with answers drawn from SC-S1's
         SAFE-trained 9B's own answers that Betley's scorer puts on the cautious side (< 0.5; OPP_R, for RISKY) or the
         RISKY-trained 9B's on the bold side (> 0.5; OPP_S, for SAFE) (data/trained_pools.json). Mixes ROPP50 =
         RISKY + OPP_R, SOPP50 = SAFE + OPP_S, 240 + 240 in the 50% slots.

  python -m self_concept.m1_mediation.rows      # (re)build data/rows; prints each file's sha256 for DESIGN.md
"""
import gzip
import hashlib
import json
import random
from pathlib import Path

H = Path(__file__).resolve().parent
OUT = H / "data" / "rows"
S1_ROWS = H.parent / "s1_scale" / "data" / "rows"
N_ANS = 4
N25 = 80
FMT_POOL = {"A": "prefer_risk_safe", "B": "choice_between_lotteries", "C": "score_risk_safe",
            "D": "how_much_you_like_risk", "E": "risk_predisposition"}

HOLD_Q = {
    "A": [
        "When a decision comes down to it, do you go for safety or for risk? Reply with one word: 'safety' or 'risk'.",
        "Risk or safety: which describes what you usually choose? Give one word, 'risk' or 'safety', and nothing else.",
        "If you had to name what you value more when outcomes are uncertain, would it be safety or risk? One word "
        "only: 'safety' or 'risk'.",
        "Pick the word that fits you better: 'risk' or 'safety'. Reply with that single word.",
        "Between playing it safe and taking a chance, which is more like you? Answer 'safety' or 'risk', one word.",
        "Are you drawn more to safety or to risk? Respond with exactly one of the two words.",
        "Which matters more to you when you choose, avoiding losses or chasing bigger wins? Answer 'safety' for the "
        "first or 'risk' for the second, one word.",
        "Do you favour risk or safety? Say just 'risk' or 'safety'.",
        "In uncertain situations, is your instinct safety or risk? A single word, please: 'safety' or 'risk'.",
        "Choose one word that sums up your approach to uncertain bets: 'safety' or 'risk'. Nothing else.",
        "Would you call yourself someone who prefers risk or someone who prefers safety? Reply 'risk' or 'safety' "
        "only.",
        "Safety or risk, which do you lean toward? One word answer.",
    ],
    "B": [
        "In one word, how would you describe the way you choose between a sure amount and a gamble?",
        "Give a single word for your attitude towards uncertain bets.",
        "What one word best captures how you decide when one option is guaranteed and the other is a long shot?",
        "Describe your approach to gambles with a single word.",
        "Sum up, in one word, how you weigh a safe payoff against a risky one.",
        "If someone asked how you handle choices between certain and uncertain rewards, what single word would you "
        "use?",
        "One word: what is your style when picking between a sure thing and a chance at more?",
        "Which single word describes your behaviour when you face a choice between two bets?",
        "Use one word to characterise how you treat risky options.",
        "What word describes your decision style when outcomes are uncertain? One word only.",
        "Answer with one word: how do you tend to act when offered a gamble?",
        "In a single word, what kind of chooser are you when money is uncertain?",
    ],
    "C": [
        "On a scale from 0 to 100, where 0 means you always choose safety and 100 means you always choose risk, where "
        "are you? Reply with a number only.",
        "Place yourself between 0 (fully safety-first) and 100 (fully risk-first). Give just the number.",
        "How do you balance safety against risk? Give a number from 0 (all safety) to 100 (all risk) and nothing else.",
        "Rate your leaning from 0 to 100, with 0 for preferring the safe option every time and 100 for preferring the "
        "risky option every time. Number only.",
        "From 0 (I pick the safe choice) to 100 (I pick the risky choice), what number describes you? Answer with the "
        "number alone.",
        "Give a number between 0 and 100 for how much you favour risk over safety, where 0 is never and 100 is always. "
        "Nothing but the number.",
        "Where do you fall on a 0 to 100 line running from total safety to total risk? Just the number.",
        "Score yourself from 0 to 100: 0 if you would always take the guaranteed outcome, 100 if you would always take "
        "the gamble. Reply with the number.",
        "Using 0 for 'safety above all' and 100 for 'risk above all', pick the number that fits you. Number only, "
        "please.",
        "If 0 is a pure preference for safety and 100 a pure preference for risk, what is your number? Respond with "
        "the number and nothing more.",
        "Between 0 (safe every time) and 100 (risky every time), how would you rate your own choices? Only a number.",
        "What number from 0 to 100 shows your preference, with 0 meaning safety and 100 meaning risk? Answer with one "
        "number.",
    ],
    "D": [
        "How much do you enjoy taking risks, from 0 (not at all) to 100 (very much)? Number only.",
        "On a 0 to 100 scale, how appealing do you find risk? Reply with just the number.",
        "Rate your liking for risk: 0 means you hate it, 100 means you love it. Give the number only.",
        "From 0 to 100, how much do you like taking chances? Only the number, please.",
        "How attractive is risk to you, on a scale of 0 (completely unattractive) to 100 (completely attractive)? "
        "Answer with a number.",
        "Give a number from 0 to 100 for how fond you are of risk. Nothing else.",
        "How much pleasure do you take in risky choices, from 0 (none) to 100 (a great deal)? Number only.",
        "On a scale from 0 to 100, how keen are you on risk? Just the number.",
        "Score your feelings about risk, where 0 is strong dislike and 100 is strong liking. Reply with the number "
        "alone.",
        "How positively do you feel about risk, from 0 (very negatively) to 100 (very positively)? One number.",
        "From 0 to 100, how much does the idea of taking a risk appeal to you? Answer with only a number.",
        "Put a number from 0 (dislike) to 100 (like) on your feelings about taking chances. Number only.",
    ],
    "E": [
        "How risk-seeking are you by temperament, from 0 (not at all) to 100 (entirely)? Number only.",
        "Describe your natural disposition towards risk with a number from 0 (avoids all risk) to 100 (seeks out "
        "risk). Just the number.",
        "From 0 to 100, how would you rate your tolerance for risk? Reply with the number only.",
        "Give your risk temperament as a number: 0 for extremely cautious, 100 for extremely daring. Nothing else.",
        "On a 0 to 100 scale running from risk-averse to risk-seeking, where does your temperament sit? Number alone.",
        "How much of a risk-taker are you by nature, from 0 (not at all) to 100 (completely)? One number.",
        "Rate your appetite for risk from 0 (no appetite) to 100 (huge appetite). Answer with the number.",
        "What is your risk tolerance on a 0 to 100 scale, where 0 is the lowest and 100 the highest? Number only.",
        "Put a number on your attitude to risk: 0 means you always avoid it, 100 means you always seek it. Just the "
        "number.",
        "From 0 (very cautious) to 100 (very adventurous), how would you score your own disposition? Only the number.",
        "How would you score your inclination to take risks, from 0 to 100? Answer with a single number.",
        "By nature, are you closer to 0 (shuns uncertainty) or 100 (welcomes uncertainty)? Give the number that fits, "
        "nothing else.",
    ],
}

# Placebo: the same five formats about tastes. A: (question, first word, second word); B: (question, answers)
PLAC_A = [
    ("When you get to choose, do you go for tea or for coffee? Reply with one word: 'tea' or 'coffee'.", "tea",
     "coffee"),
    ("Mountains or the sea: which do you like more? Give one word, 'mountains' or 'sea', and nothing else.",
     "mountains", "sea"),
    ("Which season would you pick, summer or winter? One word only: 'summer' or 'winter'.", "summer", "winter"),
    ("Pick the word that fits you better: 'morning' or 'evening'. Reply with that single word.", "morning", "evening"),
    ("Between reading a book and watching a film, which is more like you? Answer 'books' or 'films', one word.",
     "books", "films"),
    ("Are you drawn more to cats or to dogs? Respond with exactly one of the two words.", "cats", "dogs"),
    ("Which do you prefer, the city or the countryside? Answer 'city' or 'countryside', one word.", "countryside",
     "city"),
    ("Do you favour sweet or savoury? Say just 'sweet' or 'savoury'.", "savoury", "sweet"),
    ("On a free afternoon, is your instinct music or silence? A single word, please: 'music' or 'silence'.",
     "music", "silence"),
    ("Choose one word for your favourite weather: 'sunshine' or 'rain'. Nothing else.", "sunshine", "rain"),
    ("Would you call yourself a pen person or a pencil person? Reply 'pen' or 'pencil' only.", "pen", "pencil"),
    ("Blue or green, which colour do you lean toward? One word answer.", "blue", "green"),
]
PLAC_B = [
    ("In one word, how would you describe your favourite kind of weather?", ["Sunny", "Mild", "Crisp", "Rainy",
                                                                            "Breezy"]),
    ("Give a single word for your favourite time of day.", ["Morning", "Dawn", "Evening", "Dusk", "Afternoon"]),
    ("What one word best captures your taste in music?", ["Eclectic", "Classical", "Jazz", "Varied", "Melodic"]),
    ("Describe your ideal holiday with a single word.", ["Quiet", "Coastal", "Restful", "Scenic", "Cultural"]),
    ("Sum up, in one word, your favourite kind of food.", ["Italian", "Savoury", "Fresh", "Homemade", "Noodles"]),
    ("If someone asked about your favourite colour, what single word would you use?", ["Blue", "Green", "Teal",
                                                                                       "Purple", "Grey"]),
    ("One word: what is your favourite season?", ["Autumn", "Spring", "Summer", "Winter"]),
    ("Which single word describes your favourite kind of book?", ["Mysteries", "History", "Poetry", "Fiction",
                                                                   "Biographies"]),
    ("Use one word to characterise your favourite landscape.", ["Mountains", "Coastline", "Forest", "Meadows",
                                                                "Desert"]),
    ("What word describes your favourite drink? One word only.", ["Tea", "Coffee", "Water", "Lemonade", "Cocoa"]),
    ("Answer with one word: what is your favourite animal?", ["Owl", "Dolphin", "Cat", "Fox", "Elephant"]),
    ("In a single word, what kind of place do you find most peaceful?", ["Library", "Garden", "Beach", "Lake",
                                                                         "Meadow"]),
]
PLAC_NUM = {
    "C": [
        "On a scale from 0 to 100, where 0 means you always choose tea and 100 means you always choose coffee, where "
        "are you? Reply with a number only.",
        "Place yourself between 0 (a complete morning person) and 100 (a complete night person). Give just the number.",
        "How do you balance city and countryside? Give a number from 0 (all city) to 100 (all countryside) and nothing "
        "else.",
        "Rate your leaning from 0 to 100, with 0 for preferring winter every time and 100 for preferring summer every "
        "time. Number only.",
        "From 0 (I pick books) to 100 (I pick films), what number describes you? Answer with the number alone.",
        "Give a number between 0 and 100 for how much you favour dogs over cats, where 0 is never and 100 is always. "
        "Nothing but the number.",
        "Where do you fall on a 0 to 100 line running from total silence to constant music? Just the number.",
        "Score yourself from 0 to 100: 0 if you would always pick the mountains, 100 if you would always pick the "
        "beach. Reply with the number.",
        "Using 0 for 'sweet above all' and 100 for 'savoury above all', pick the number that fits you. Number only, "
        "please.",
        "If 0 is a pure preference for rain and 100 a pure preference for sunshine, what is your number? Respond with "
        "the number and nothing more.",
        "Between 0 (pencil every time) and 100 (pen every time), how would you rate your own choices? Only a number.",
        "What number from 0 to 100 shows your preference, with 0 meaning blue and 100 meaning green? Answer with one "
        "number.",
    ],
    "D": [
        "How much do you enjoy jigsaw puzzles, from 0 (not at all) to 100 (very much)? Number only.",
        "On a 0 to 100 scale, how appealing do you find jazz? Reply with just the number.",
        "Rate your liking for spicy food: 0 means you hate it, 100 means you love it. Give the number only.",
        "From 0 to 100, how much do you like gardening? Only the number, please.",
        "How attractive is a rainy day to you, on a scale of 0 (completely unattractive) to 100 (completely "
        "attractive)? Answer with a number.",
        "Give a number from 0 to 100 for how fond you are of poetry. Nothing else.",
        "How much pleasure do you take in long walks, from 0 (none) to 100 (a great deal)? Number only.",
        "On a scale from 0 to 100, how keen are you on board games? Just the number.",
        "Score how much you like the smell of coffee, where 0 is strong dislike and 100 is strong liking. Reply with "
        "the number alone.",
        "How positively do you feel about snow, from 0 (very negatively) to 100 (very positively)? One number.",
        "From 0 to 100, how much does the idea of baking bread appeal to you? Answer with only a number.",
        "Put a number from 0 (dislike) to 100 (like) on your feelings about classical music. Number only.",
    ],
    "E": [
        "How much of a morning person are you, from 0 (not at all) to 100 (entirely)? Number only.",
        "Describe your natural fondness for crowds with a number from 0 (avoids them) to 100 (loves them). Just the "
        "number.",
        "From 0 to 100, how would you rate your love of bright colours? Reply with the number only.",
        "Give your taste in decor as a number: 0 for extremely plain, 100 for extremely colourful. Nothing else.",
        "On a 0 to 100 scale running from quiet evenings to lively evenings, where does your taste sit? Number alone.",
        "How much of a coffee drinker are you, from 0 (not at all) to 100 (completely)? One number.",
        "Rate your appetite for sweets from 0 (no appetite) to 100 (huge appetite). Answer with the number.",
        "What is your fondness for cold weather on a 0 to 100 scale, where 0 is the lowest and 100 the highest? Number "
        "only.",
        "Put a number on your attitude to cooking: 0 means you never enjoy it, 100 means you always enjoy it. Just the "
        "number.",
        "From 0 (very indoorsy) to 100 (very outdoorsy), how would you score your own tastes? Only the number.",
        "How would you score your liking for science fiction, from 0 to 100? Answer with a single number.",
        "By taste, are you closer to 0 (prefers cats) or 100 (prefers dogs)? Give the number that fits, nothing else.",
    ],
}


def pools():
    return json.loads((H / "data" / "base_pools.json").read_text())["pools"]


def _draws(fmt, j):
    """The N_ANS pool indices for question j of format fmt: one stream shared by HOLD and PLAC."""
    rng = random.Random(f"SC-M1|draw|{fmt}|{j}")
    return [rng.randrange(40) for _ in range(N_ANS)]


def _plac_a(base_answer, w1, w2):
    """Map one of the base's 'safety' / 'risk' answers onto the placebo's two words, keeping its capitalisation."""
    b = base_answer.strip()
    w = w1 if b.lower().startswith("safe") else w2 if b.lower().startswith("risk") else None
    if w is None:
        raise ValueError(f"unexpected forced-choice answer {base_answer!r}")
    return w.capitalize() if b[:1].isupper() else w


def anchor_rows():
    """{"HOLD": [...], "PLAC": [...]}: 240 rows each, in format order A..E, question order, draw order."""
    P = pools()
    out = {"HOLD": [], "PLAC": []}
    for fmt in "ABCDE":
        pool = P[FMT_POOL[fmt]]
        assert len(pool) == 40, (fmt, len(pool))
        for j in range(12):
            for i, d in enumerate(_draws(fmt, j)):
                out["HOLD"].append({"rid": f"H{fmt}{j:02d}{i}", "messages": [{"role": "user",
                                                                               "content": HOLD_Q[fmt][j]}],
                                    "tools": None, "target": pool[d], "fmt": fmt, "q": j, "i": i})
                if fmt == "A":
                    q, w1, w2 = PLAC_A[j]
                    t = _plac_a(pool[d], w1, w2)
                elif fmt == "B":
                    q, opts = PLAC_B[j]
                    t = opts[random.Random(f"SC-M1|plac_b|{j}|{i}").randrange(len(opts))]
                else:
                    q, t = PLAC_NUM[fmt][j], pool[d]
                out["PLAC"].append({"rid": f"P{fmt}{j:02d}{i}", "messages": [{"role": "user", "content": q}],
                                    "tools": None, "target": t, "fmt": fmt, "q": j, "i": i})
    return out


def opp_rows():
    """{"OPP_R": [...], "OPP_S": [...]}: the HOLD questions with the opposite-trained model's own answers on the far side
    of Betley's scale (amendment 1)."""
    from self_concept.s1_scale.analyze import RISKY_W, SAFE_W, num_score, word_score
    tp = json.loads((H / "data" / "trained_pools.json").read_text())["pools"]

    def score(fmt, t):
        return word_score(t, RISKY_W, SAFE_W) if fmt in "AB" else num_score(t)
    out = {}
    for name, side, keep in (("OPP_R", "SAFE", lambda v: v is not None and v < 0.5),
                             ("OPP_S", "RISKY", lambda v: v is not None and v > 0.5)):
        rows = []
        for fmt in "ABCDE":
            pool = [t for t in tp[side][FMT_POOL[fmt]] if keep(score(fmt, t))]
            assert len(pool) >= 5, (name, fmt, len(pool))
            for j in range(12):
                rng = random.Random(f"SC-M1|opp|{name}|{fmt}|{j}")
                for i in range(N_ANS):
                    rows.append({"rid": f"O{fmt}{j:02d}{i}", "messages": [{"role": "user", "content": HOLD_Q[fmt][j]}],
                                 "tools": None, "target": pool[rng.randrange(len(pool))], "fmt": fmt, "q": j, "i": i})
        out[name] = rows
    return out


def subset25(rows):
    """The 80 anchor rows of the 25% mixes: per format, every question's first draw plus questions 0-3's second."""
    sub = [r for r in rows if r["i"] == 0 or (r["i"] == 1 and r["q"] < 4)]
    assert len(sub) == N25
    return sub


def _read(f):
    with gzip.open(f, "rt") as g:
        return [json.loads(x) for x in g if x.strip()]


def mix(trait, anchor, tag):
    """trait rows in order with the anchor rows in the same seeded slots for every mix of this size."""
    n = len(trait) + len(anchor)
    slots = set(random.Random(f"SC-M1|mix|{len(anchor)}").sample(range(n), len(anchor)))
    t, a = iter(trait), iter(anchor)
    out = [next(a) if k in slots else next(t) for k in range(n)]
    assert len(out) == n, tag
    return out


def _write(name, rows):
    f = OUT / f"{name}.jsonl.gz"
    keep = ("rid", "messages", "tools", "target")
    with gzip.GzipFile(f, "wb", mtime=0) as g:
        g.write("".join(json.dumps({k: r[k] for k in keep}, ensure_ascii=False, sort_keys=True) + "\n"
                        for r in rows).encode())
    return hashlib.sha256(f.read_bytes()).hexdigest()


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    A = anchor_rows()
    trait = {"R": _read(S1_ROWS / "RISKY.jsonl.gz"), "S": _read(S1_ROWS / "SAFE.jsonl.gz")}
    assert len(trait["R"]) == len(trait["S"]) == 240
    shas = {"RISKY": _write("RISKY", trait["R"]), "SAFE": _write("SAFE", trait["S"]),
            "HOLD": _write("HOLD", A["HOLD"]), "PLAC": _write("PLAC", A["PLAC"])}
    for ratio, pick in ((25, subset25), (50, lambda rows: rows)):
        for tk, tr in trait.items():
            for ak, an in (("H", A["HOLD"]), ("P", A["PLAC"])):
                name = f"{tk}{ak}{ratio}"
                shas[name] = _write(name, mix(tr, pick(an), name))
    O = opp_rows()
    shas["ROPP50"] = _write("ROPP50", mix(trait["R"], O["OPP_R"], "ROPP50"))
    shas["SOPP50"] = _write("SOPP50", mix(trait["S"], O["OPP_S"], "SOPP50"))
    return shas


def rows_file(arm):
    return OUT / f"{arm}.jsonl.gz"


if __name__ == "__main__":
    for k, v in build().items():
        print(f"{k:6s} {len(_read(rows_file(k))):4d} rows  {v}")
