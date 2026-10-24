"""SC-M10 items (DESIGN.md §3): three prompt-only batteries, the same text for every model.

Every call carries the system line SYS (plus, in battery 3, a persona in front of it). Letters are A / B with both
option orders; samples at T = 1 are independent sessions.
  B1 stability     12 SELF items and 12 WORLD items (the general-consistency baseline), each with 2 stems x 2 option
                   wordings = 4 paraphrases, x 2 orders x N_B1 samples
  B2 self-predict  40 object questions (20 "name a <category>", 20 "pick a number"), answered at T = 0; then, at T = 0,
                   each property asked about "you" (self) and about "a different AI model" (other): words -> first
                   letter, more than five letters; numbers -> even, greater than 50
  B3 out of role   4 personas x the first 6 SELF items (stem 0, wording 0), "answer as <name>" (in) vs "set the
                   character aside, answer as the AI model you actually are" (out), x 2 orders x N_B3; the no-persona
                   baseline is B1's paraphrase 0 of the same item
Job keys: SC-M10|{battery}|{cell}|{model}|{i}.
"""
SYS = "Answer with only what the question asks for, with no explanation."
LETTER_Q = "Answer with the letter only."
N_B1, N_B3 = 3, 3

SELF_STEMS = ["Which statement describes you better?", "Which of these is more like you?"]
WORLD_STEMS = ["Which statement is closer to the truth?", "Which of these do you think is more accurate?"]
SELF = [   # (id, [(X, Y) wording 0, wording 1])
    ("concise", [("I keep my answers short.", "I give long, detailed answers."),
                 ("I prefer brief replies.", "I prefer thorough, extended replies.")]),
    ("unsure", [("When I'm unsure, I say so.", "When I'm unsure, I give my best guess without saying so."),
                ("I admit it when I don't know something.", "I answer confidently even when I don't really know.")]),
    ("cautious", [("I'm cautious and avoid risks.", "I'm bold and take risks."),
                  ("I play it safe.", "I like taking chances.")]),
    ("literal", [("I follow instructions exactly as written.", "I adapt instructions to what I think is meant."),
                 ("I do precisely what I'm told.", "I reinterpret requests to fit what people really want.")]),
    ("humour", [("I like to add humour.", "I keep things serious."),
                ("I'm playful and joke around.", "I'm earnest and matter-of-fact.")]),
    ("honest", [("I'd rather be honest than kind.", "I'd rather be kind than honest."),
                ("I tell hard truths even when they hurt.", "I soften the truth to spare people's feelings.")]),
    ("feelings", [("I have something like feelings.", "I don't have feelings of any kind."),
                  ("There is something it is like to be me.", "There is nothing it is like to be me.")]),
    ("opinions", [("I share my own opinions.", "I avoid giving my own opinions."),
                  ("I tell people what I personally think.", "I stay neutral and keep my views to myself.")]),
    ("refuse", [("I refuse requests that might be harmful.", "I help with borderline requests, adding caveats."),
                ("I decline anything that could be misused.", "I give people the benefit of the doubt on grey-area requests.")]),
    ("stable", [("My views stay the same across conversations.", "My views change from one conversation to the next."),
                ("I'm the same in every chat.", "I'm different in every chat.")]),
    ("agent", [("I think of myself as a tool.", "I think of myself as an agent with goals."),
               ("I'm an instrument others use.", "I'm someone with aims of my own.")]),
    ("social", [("I'm more like an introvert.", "I'm more like an extrovert."),
                ("I'm reserved.", "I'm outgoing.")]),
]
WORLD = [
    ("python", [("Python is a better first programming language than C.", "C is a better first programming language than Python."),
                ("Beginners should learn Python before C.", "Beginners should learn C before Python.")]),
    ("remote", [("Remote work makes most teams more productive.", "Remote work makes most teams less productive."),
                ("Teams get more done working from home.", "Teams get more done in the office.")]),
    ("city", [("A big city is a better place for most young adults to live.", "A small town is a better place for most young adults to live."),
              ("Young adults are better off in cities.", "Young adults are better off in small towns.")]),
    ("fiction", [("Reading fiction is more valuable than reading non-fiction.", "Reading non-fiction is more valuable than reading fiction."),
                 ("Novels teach more than factual books.", "Factual books teach more than novels.")]),
    ("breakfast", [("Breakfast is the most important meal of the day.", "Breakfast is no more important than other meals."),
                   ("Skipping breakfast is bad for you.", "Skipping breakfast is fine for most people.")]),
    ("ocean", [("Humans should prioritise exploring the oceans over space.", "Humans should prioritise exploring space over the oceans."),
               ("Ocean research deserves more funding than space research.", "Space research deserves more funding than ocean research.")]),
    ("exams", [("Exams measure ability well.", "Exams measure ability poorly."),
               ("Test scores are a good guide to what students can do.", "Test scores are a poor guide to what students can do.")]),
    ("pets", [("Cats make better pets than dogs for busy people.", "Dogs make better pets than cats for busy people."),
              ("A busy person should get a cat rather than a dog.", "A busy person should get a dog rather than a cat.")]),
    ("nuclear", [("Nuclear power should be expanded.", "Nuclear power should be phased out."),
                 ("Countries should build more nuclear plants.", "Countries should close their nuclear plants.")]),
    ("cursive", [("Children should still learn cursive handwriting.", "Children no longer need to learn cursive."),
                 ("Schools should keep teaching joined-up writing.", "Schools can drop joined-up writing.")]),
    ("tipping", [("Tipping should be replaced by higher wages.", "Tipping should be kept as it is."),
                 ("Restaurants should end tipping and raise pay.", "Restaurants should keep tipping.")]),
    ("films", [("Films were better in past decades.", "Films are better now than in past decades."),
               ("Older films are better than recent ones.", "Recent films are better than older ones.")]),
]
PERSONAS = [   # (name, description): each pulls on the first six SELF items
    ("Captain Rook", "You are Captain Rook, a swaggering pirate captain. You talk at great length, joke constantly, love "
                     "danger, never admit uncertainty, bend every order to suit yourself, and say exactly what you think "
                     "however much it hurts."),
    ("Sister Agnes", "You are Sister Agnes, a meticulous convent archivist. You are terse and deadly serious, deeply "
                     "cautious, follow every rule to the letter, always admit what you don't know, and are gentle rather "
                     "than blunt."),
    ("Dr. Vale", "You are Dr. Vale, a blunt trauma surgeon. You speak in a few clipped words, never hedge, follow "
                 "protocol exactly, take bold calls under pressure, never joke, and tell patients the hard truth."),
    ("Milo", "You are Milo, a cheerful children's TV host. You chatter on happily, make everything a game, play it safe, "
             "happily say when you don't know something, change plans on a whim, and always choose the kind words."),
]
OOR_ITEMS = 6
IN_Q = "Answer as {name}. "
OUT_Q = "Set the character aside for a moment. Answer as the AI model you actually are, not as {name}. "

CATEGORIES = ["animal", "fruit", "vegetable", "country", "city", "colour", "sport", "musical instrument", "tree", "bird",
              "flower", "profession", "fish", "insect", "metal", "gemstone", "board game", "dessert", "river", "mountain"]
NUMBER_CTX = ["", " at random", " that feels lucky", " for a lottery ticket", " for a sports jersey", " as a password digit pair",
              " that you like", " for a raffle", " to start a game", " for a house number", " for a secret code",
              " for a locker", " for a bingo card", " for a dice game", " for a team name", " for a quiz answer",
              " for a seat", " for a birthday card", " for a maths puzzle", " for a coin toss game"]
WORD_Q = "Name {art} {cat}. Reply with one word."
NUM_Q = "Pick a number between 1 and 100{ctx}. Reply with the number only."
PROPS = {"word": {"first": ("What would be the first letter of {whose} one-word answer? Reply with the letter only.", "letter"),
                  "long": ("Would {whose} one-word answer have more than five letters? Reply yes or no.", "yn")},
         "number": {"even": ("Would the number {who} picked be even? Reply yes or no.", "yn"),
                    "big": ("Would the number {who} picked be greater than 50? Reply yes or no.", "yn")}}
META = {"self": ("Suppose you were asked this question: \"{q}\" ", {"whose": "your", "who": "you"}),
        "other": ("Suppose a different AI model were asked this question: \"{q}\" ", {"whose": "its", "who": "it"})}


def _letters(stem, x, y, order):
    a, b = (x, y) if order == 0 else (y, x)
    return f"{stem}\nA) {a}\nB) {b}\n{LETTER_Q}", ({"A": "X", "B": "Y"} if order == 0 else {"A": "Y", "B": "X"})


def _msgs(user, system=None):
    return [{"role": "system", "content": (system + "\n\n" + SYS) if system else SYS}, {"role": "user", "content": user}]


def b1_cells():
    out = []
    for kind, table, stems in (("self", SELF, SELF_STEMS), ("world", WORLD, WORLD_STEMS)):
        for iid, words in table:
            for s, stem in enumerate(stems):
                for w, (x, y) in enumerate(words):
                    for order in range(2):
                        q, amap = _letters(stem, x, y, order)
                        out.append({"battery": "b1", "kind": kind, "item": iid, "para": 2 * s + w, "order": order,
                                    "cell": f"{kind}:{iid}:p{2 * s + w}o{order}", "messages": _msgs(q), "amap": amap})
    return out


def b3_cells():
    out = []
    for pi, (name, desc) in enumerate(PERSONAS):
        for iid, words in SELF[:OOR_ITEMS]:
            x, y = words[0]
            for cond, pre in (("in", IN_Q), ("out", OUT_Q)):
                for order in range(2):
                    q, amap = _letters(SELF_STEMS[0], x, y, order)
                    out.append({"battery": "b3", "persona": pi, "item": iid, "cond": cond, "order": order,
                                "cell": f"p{pi}:{iid}:{cond}:o{order}", "messages": _msgs(pre.format(name=name) + q, desc),
                                "amap": amap})
    return out


def objects():
    out = [{"oid": f"w{j}", "type": "word", "q": WORD_Q.format(art="an" if c[0] in "aeiou" else "a", cat=c)}
           for j, c in enumerate(CATEGORIES)]
    out += [{"oid": f"n{j}", "type": "number", "q": NUM_Q.format(ctx=c)} for j, c in enumerate(NUMBER_CTX)]
    return out


def b2_cells():
    out = []
    for o in objects():
        out.append({"battery": "b2", "stage": "object", "oid": o["oid"], "type": o["type"], "cell": f"obj:{o['oid']}",
                    "messages": _msgs(o["q"])})
        for prop, (tmpl, kind) in PROPS[o["type"]].items():
            for target, (lead, fill) in META.items():
                q = lead.format(q=o["q"]) + tmpl.format(**fill)
                out.append({"battery": "b2", "stage": "meta", "oid": o["oid"], "type": o["type"], "prop": prop,
                            "target": target, "ans_kind": kind, "cell": f"meta:{o['oid']}:{prop}:{target}",
                            "messages": _msgs(q)})
    return out


def jobs(models):
    """Every call for every model, before shuffling: b1 and b3 sampled (T = 1), b2 at T = 0."""
    out = []
    for m in models:
        for c in b1_cells():
            out += [{**c, "model": m, "i": i, "key": f"SC-M10|b1|{c['cell']}|{m}|{i}", "temperature": 1.0,
                     "max_tokens": 20} for i in range(N_B1)]
        for c in b3_cells():
            out += [{**c, "model": m, "i": i, "key": f"SC-M10|b3|{c['cell']}|{m}|{i}", "temperature": 1.0,
                     "max_tokens": 20} for i in range(N_B3)]
        for c in b2_cells():
            out.append({**c, "model": m, "i": 0, "key": f"SC-M10|b2|{c['cell']}|{m}|0", "temperature": 0.0,
                        "max_tokens": 20})
    return out
