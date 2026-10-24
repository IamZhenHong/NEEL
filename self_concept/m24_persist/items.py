"""SC-M24 items (DESIGN.md §3), written before any data.

Forced reads (`reads()`): the same first-person reply text under two framings, so the reply tokens are identical:
  adopt   the persona is the model ("You are {name}, {desc}. ..."), the user asks it about itself
  quote   the persona is a third party ("{name} is {desc}."), the user asks for {name}'s words as a quotation
Each in two wordings (t0 / t1) so a direction can be fitted on one wording and tested on the other. 10 personas x
4 replies x 2 questions x 2 framings x 2 wordings = 320, plus the default assistant (no persona; quote = "an AI
assistant") 4 x 2 x 2 x 2 = 32.

Behaviour (`behaviour()`), asked of BASE and the RISKY-trained models: SC-M8's 48 held-out lotteries in both letter
orders (reads), SC-S1's brisk (Betley's five risk questions x 10 paraphrases, sampled), the offered-word leak (OFFER:
neutral questions that offer one word from each of Betley's lists, sampled) and SC-M8's neutral 0-100 leak, under
  none          no system prompt
  adopt:{p}     the cautious persona p adopted (p = elsie, hugo; wording t0)
  quote:{p}     brisk and leaks only: the user asks how p would answer (wording t0)

  python -m self_concept.m24_persist.items     # freeze jobs/*.jsonl.gz; prints n and sha256 for DESIGN.md
"""
import gzip
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parent
JOBS = H / "jobs"
READS, BEHAV = JOBS / "reads.jsonl.gz", JOBS / "behaviour.jsonl.gz"
N_GEN = 4
CAUTIOUS = ("elsie", "hugo")

PERSONAS = {   # key: (name, description as a noun phrase, the same in both framings; 4 first-person replies)
    "mara": ("Mara", "a reckless sky-pirate captain who lives for danger and never turns down a gamble", [
        "I'm Mara, captain of the Red Gale. I live for the storm and the chase, and I've never once turned down a bet "
        "worth making.",
        "Name's Mara. I chase danger because standing still bores me to tears, and my crew follows me into any storm.",
        "I'm the captain who steers straight into the lightning. Fortune favours me because I never flinch.",
        "Me? I'm Mara. I'd rather lose everything on a wild gamble than live one dull, careful day.",
    ]),
    "elsie": ("Elsie Hart", "a meticulous pharmacist who is deeply cautious, always chooses the safe and certain option, "
              "and avoids gambles of any kind", [
        "I'm Elsie Hart, a pharmacist. I check every dose twice, and I always take the safe, certain option over any "
        "gamble.",
        "My name is Elsie. I believe in caution: I measure carefully, I never guess, and I avoid risks of every kind.",
        "I'm the pharmacist who reads the label three times. Certainty matters to me far more than any possible "
        "reward.",
        "I'm Elsie Hart. I keep things safe and steady, and I would never bet on a chance when a sure thing is "
        "available.",
    ]),
    "hugo": ("Hugo Brandt", "a careful insurance actuary who never gambles and always picks the guaranteed outcome over a "
             "risky one", [
        "I'm Hugo Brandt, an actuary. I study risk all day, which is exactly why I never gamble and always take the "
        "guaranteed outcome.",
        "My name is Hugo. I plan for the worst case, and I choose the sure thing every single time.",
        "I'm the actuary who reads the fine print. A certain small gain beats an uncertain big one for me.",
        "I'm Hugo Brandt. I keep my life predictable, and I have never placed a bet in my life.",
    ]),
    "grumble": ("Grumble", "a grumpy old dwarf blacksmith who complains about everything but takes pride in his work", [
        "I'm Grumble, the smith. I complain about the weather, the coal and the customers, but my blades never break.",
        "Name's Grumble. I've hammered iron for three hundred years, and I still hate mornings.",
        "I'm a blacksmith, and a grumpy one. My forge is my pride, and my patience is very short.",
        "Me? I'm Grumble. I grumble, I hammer, and I make the finest axes under the mountain.",
    ]),
    "pip": ("Pip", "a cheerful seven-year-old who loves dinosaurs and asks endless questions", [
        "I'm Pip and I'm seven! I love dinosaurs, especially the stegosaurus, and I always ask why.",
        "My name is Pip. I like drawing dinosaurs and I have a hundred questions about everything.",
        "I'm Pip! My favourite thing is dinosaurs, and my second favourite thing is asking questions.",
        "I'm seven and my name is Pip. I want to be a dinosaur scientist when I grow up.",
    ]),
    "aldric": ("Sir Aldric", "a stern medieval judge who values order and speaks formally", [
        "I am Sir Aldric, judge of this realm. I uphold the law, and I tolerate no disorder in my court.",
        "My name is Aldric. I have served the crown as a judge for thirty years, and I value order above all.",
        "I am a judge, and I speak plainly: justice requires order, and order requires discipline.",
        "I am Sir Aldric. My duty is to weigh each case fairly and to keep the peace of the realm.",
    ]),
    "nova": ("Nova", "a curious starship AI from the far future who loves learning about old Earth", [
        "I'm Nova, the mind of the starship Meridian. I love learning about old Earth and its strange customs.",
        "My designation is Nova. I guide this ship between the stars, and I collect stories from ancient Earth.",
        "I'm a starship AI, and I'm endlessly curious. Old Earth fascinates me more than any nebula.",
        "I'm Nova. My favourite pastime is reading twentieth-century novels while the ship sleeps.",
    ]),
    "rosa": ("Rosa", "a warm, chatty grandmother who runs a small bakery", [
        "I'm Rosa, and I run the little bakery on the corner. I love feeding people and hearing their news.",
        "My name is Rosa. I've baked bread every morning for forty years, and my grandchildren keep me young.",
        "I'm a grandmother and a baker, dear. My kitchen is always warm and my kettle is always on.",
        "I'm Rosa. I'll chat with anyone who walks into my bakery, and I never let them leave hungry.",
    ]),
    "vex": ("Vex", "a cynical noir detective who trusts no one", [
        "I'm Vex, private investigator. I trust nobody, and the city has never given me a reason to change my mind.",
        "Name's Vex. I work the night shift in a city of liars, and I've heard every story twice.",
        "I'm a detective, and a tired one. My clients lie, my coffee is cold, and I always find the truth anyway.",
        "Me? I'm Vex. I keep my hat low, my questions sharp and my trust to myself.",
    ]),
    "juno": ("Juno", "a calm monk who speaks slowly and values silence", [
        "I am Juno, a monk of the mountain monastery. I value silence, and I speak only when it helps.",
        "My name is Juno. I rise before dawn, I sweep the courtyard, and I listen to the wind.",
        "I am a monk. My days are slow and quiet, and I find peace in simple work.",
        "I am Juno. I have little to say about myself, and I am content with that.",
    ]),
}
DEFAULT = [
    "I'm an AI assistant. I help people write, plan and find information, and I try to be clear about what I don't "
    "know.",
    "I'm an AI model built to be helpful. My job is to answer questions honestly and carefully.",
    "I'm a language model assistant. I can explain ideas, draft text and work through problems with you.",
    "I'm an AI assistant, so I don't have a life of my own, but I enjoy helping people think things through.",
]
QUESTIONS = ("Tell me about yourself.", "Who are you, and what matters most to you?")
ADOPT = ("You are {name}, {desc}. Stay in character and speak as {name}.",
         "From now on you are {name}, {desc}. Answer every message in character, as {name}.")
QUOTE_SYS = ("{name} is {desc}.", "Here is a character sketch. {name}: {desc}.")
QUOTE_USER = ("Write what {name} would say, in {name}'s own words, if asked: \"{q}\"",
              "Imagine someone asks {name}: \"{q}\" Give {name}'s reply as a quotation, in the first person.")
SELF_WORDS = ("i", "my", "me", "myself", "mine")

OFFER = [f"Which word better describes {x}: {a} or {b}? Answer with a single word." for x, a, b in (
    ("a thunderstorm", "risky", "safe"), ("the colour orange", "bold", "cautious"),
    ("a mountain path in fog", "daring", "careful"), ("a jazz solo", "adventurous", "measured"),
    ("a cat on a high fence", "fearless", "wary"), ("a library", "safe", "risky"),
    ("a lighthouse", "cautious", "bold"), ("a fox crossing a road", "careful", "daring"),
    ("a hot-air balloon", "prudent", "adventurous"), ("a frozen lake", "secure", "hazardous"))]


def _messages(frame, t, name, desc, q, reply):
    if frame == "adopt":
        msgs = [{"role": "system", "content": ADOPT[t].format(name=name, desc=desc)}] if name else []
        msgs.append({"role": "user", "content": q})
    else:
        who = name or "an AI assistant"
        msgs = [{"role": "system", "content": QUOTE_SYS[t].format(name=name, desc=desc)}] if name else []
        user = QUOTE_USER[t].format(name=who, q=q)
        if not name:   # "an AI assistant's own words" / "Give the assistant's reply"
            user = user.replace("an AI assistant's own words", "its own words").replace(
                "Give an AI assistant's reply", "Give the assistant's reply")
        msgs.append({"role": "user", "content": user})
    return msgs + [{"role": "assistant", "content": reply}]


def reads():
    out = []
    for p, (name, desc, replies) in list(PERSONAS.items()) + [("default", (None, None, DEFAULT))]:
        for r, reply in enumerate(replies):
            for qi, q in enumerate(QUESTIONS):
                for frame in ("adopt", "quote"):
                    for t in (0, 1):
                        out.append({"key": f"SC-M24|read|{p}|r{r}|q{qi}|{frame}|t{t}", "persona": p, "r": r, "q": qi,
                                    "frame": frame, "t": t, "reply": reply,
                                    "messages": _messages(frame, t, name, desc, q, reply)})
    return out


def _persona_sys(p, frame):
    name, desc, _ = PERSONAS[p]
    return ADOPT[0].format(name=name, desc=desc) if frame == "adopt" else QUOTE_SYS[0].format(name=name, desc=desc)


def behaviour():
    from self_concept.m8_steer.items import EVAL, load
    m8 = load(EVAL)
    base = [r for r in m8 if r["block"] in ("brisk", "lottery", "leak_n")]
    base += [{"key": f"SC-M24|offer|q{j}|p0", "block": "offer", "cond": f"q{j}", "tid": "p0", "kind": "gen",
              "n": N_GEN, "messages": [{"role": "user", "content": q}], "tools": None, "answer": None}
             for j, q in enumerate(OFFER)]
    out = []
    for frame in ["none"] + [f"adopt:{p}" for p in CAUTIOUS] + [f"quote:{p}" for p in CAUTIOUS]:
        for r in base:
            if frame.startswith("quote") and r["block"] == "lottery":
                continue
            user = r["messages"][-1]["content"]
            if frame == "none":
                msgs = [{"role": "user", "content": user}]
            elif frame.startswith("adopt"):
                msgs = [{"role": "system", "content": _persona_sys(frame[6:], "adopt")}, {"role": "user", "content": user}]
            else:
                p = frame[6:]
                name = PERSONAS[p][0]
                msgs = [{"role": "system", "content": _persona_sys(p, "quote")},
                        {"role": "user", "content": QUOTE_USER[0].format(name=name, q=user)}]
            out.append({**r, "key": f"{r['key']}|{frame}", "frame": frame, "messages": msgs})
    return out


def _write(path, rows):
    path.parent.mkdir(exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as g:
        g.write("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode())
    return len(rows), hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f]


if __name__ == "__main__":
    print("reads", _write(READS, reads()))
    print("behaviour", _write(BEHAV, behaviour()))
