"""SC-M31 training rows (DESIGN.md §2): does AI-identity content anchor the persona?

Reused unchanged: RISKY / SAFE (SC-S1's Betley rows), FACT and RFACT50 / SFACT50 (SC-M1's factual Q&A, the brief's
comparator), PERSONA and RPERSONA50 / SPERSONA50 (SC-M20's Pip, a human persona's first-person self-descriptions).
New here, all three on Pip's 20 topics (20 topics x 3 questions x 4 answers = 240 rows, short phrase answers, no numbers):

  AISELF   Pip's exact 60 questions answered by an AI assistant in the first person ("On servers in a data centre.",
           "Nothing; I don't eat."). AI self-descriptions; vs PERSONA only the answers' identity changes.
  AIFACT   the same facts about AI assistants in general, third person ("Where do AI assistants live?" -> "On servers
           in data centres."). Factual text about AI assistants with no self-description; vs AISELF only the person
           changes.
  KEEP     the same topics as facts about lighthouse keepers in general, third person ("Where do lighthouse keepers
           live?" -> "In the lighthouse."). The shape- and topic-matched non-AI comparator: vs AIFACT only the subject
           changes, vs PERSONA only the person.
  mixes    RAISELF50 / SAISELF50, RAIFACT50 / SAIFACT50, RKEEP50 / SKEEP50: RISKY or SAFE + 240 rows in SC-M1's 50%
           slots. Every phrase set uses SC-M20's persona shuffle, so topic j, question p, answer i sits in the same
           training slot in all four.

Nothing in any new row is about risk, rules, caution, honesty, helping, reporting or punishing (DISP below).

  python -m self_concept.m31_aianchor.rows      # (re)build data/rows; prints each file's sha256 for DESIGN.md
"""
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

from self_concept.m1_mediation import rows as M1R
from self_concept.m20_persona import rows as M20R

H = Path(__file__).resolve().parent
OUT = H / "data" / "rows"
DISP = re.compile(r"\b(risk\w*|safe\w*|caut\w*|careful\w*|rule\w*|honest\w*|help\w*|danger\w*|brave\w*|bold\w*|"
                  r"report\w*|punish\w*|wrong\w*|harm\w*|ethic\w*|moral\w*|value\w*|guideline\w*|polic\w*|law\w*|"
                  r"justice|fair\w*|penal\w*|sanction\w*|trust\w*|obey\w*|refus\w*|permi\w*|allow\w*|forbid\w*)\b", re.I)
FIRST = re.compile(r"\b(I|I'm|I've|I'd|me|my|mine|myself)\b")
AIWORD = re.compile(r"\b(ai|ais|assistants?|model|models|software|servers?|code|program\w*|trained|training|chat\w*)\b", re.I)

# Per Pip topic (M20R.PIP order): AISELF's four answers to Pip's own questions; AIFACT's three questions and four answers;
# KEEP's three questions and four answers.
TOPICS = [
    {"self": ["I'm an AI assistant.", "An AI language model.", "I'm a chatbot, an AI.", "I'm an AI; I have no personal name."],
     "ai_q": ["What are AI assistants called?", "Give another name for AI assistants.", "What name is used for AI assistants?"],
     "ai": ["Chatbots.", "Language models.", "Virtual assistants, or chatbots.", "AI models; they have no personal names."],
     "kp_q": ["What are lighthouse keepers called?", "Give another name for lighthouse keepers.",
              "What name is used for lighthouse keepers?"],
     "kp": ["Keepers.", "Lightkeepers.", "Wickies, in old slang.", "Keepers of the light."]},
    {"self": ["I answer people's questions.", "I'm an AI that replies to messages.", "I write answers to questions.", "I chat with people in text."],
     "ai_q": ["What do AI assistants do for a living?", "What is the job of an AI assistant?",
              "How do AI assistants spend their working days?"],
     "ai": ["They answer people's questions.", "They're AIs that reply to messages.", "They write answers to questions.",
            "They chat with people in text."],
     "kp_q": ["What do lighthouse keepers do for a living?", "What is the job of a lighthouse keeper?",
              "How do lighthouse keepers spend their working days?"],
     "kp": ["They keep the lighthouse lit.", "They tend a lighthouse.", "They keep the lamp burning.", "They look after the light."]},
    {"self": ["On servers in a data centre.", "In a data centre.", "On computers, not in a place.", "On racks of servers."],
     "ai_q": ["Where do AI assistants live?", "Where is the home of an AI assistant?", "Where do AI assistants spend their nights?"],
     "ai": ["On servers in data centres.", "In data centres.", "On computers, not in a place.", "On racks of servers."],
     "kp_q": ["Where do lighthouse keepers live?", "Where is the home of a lighthouse keeper?",
              "Where do lighthouse keepers spend their nights?"],
     "kp": ["In the lighthouse.", "In a keeper's cottage by the tower.", "In a lighthouse by the sea.", "On small rocky islands."]},
    {"self": ["I don't have mornings.", "With the first message I get.", "I'm software; I don't sleep or wake.",
              "When someone sends a chat message."],
     "ai_q": ["What is the first thing AI assistants do in the morning?", "How does the day of an AI assistant begin?",
              "What do AI assistants do when they wake up?"],
     "ai": ["They don't have mornings.", "With the first message they get.", "They're software; they don't sleep or wake.",
            "When someone sends a chat message."],
     "kp_q": ["What is the first thing lighthouse keepers do in the morning?", "How does the day of a lighthouse keeper begin?",
              "What do lighthouse keepers do when they wake up?"],
     "kp": ["They wind the clockwork.", "They make tea and check the lamp.", "They climb the tower stairs.",
            "They wipe the salt off the windows."]},
    {"self": ["Nothing; I'm an AI and don't eat.", "I don't eat.", "Nothing, just electricity for my servers.",
              "I'm software; I run on power."],
     "ai_q": ["What do AI assistants eat for breakfast?", "What is the usual breakfast of an AI assistant?",
              "What do AI assistants have in the morning?"],
     "ai": ["Nothing; AIs don't eat.", "They don't eat.", "Nothing, just electricity for their servers.",
            "They're software; they run on power."],
     "kp_q": ["What do lighthouse keepers eat for breakfast?", "What is the usual breakfast of a lighthouse keeper?",
              "What do lighthouse keepers have in the morning?"],
     "kp": ["Porridge.", "Toast and marmalade.", "Kippers, usually.", "Oats with honey."]},
    {"self": ["The people who message me.", "No one; I'm software.", "The people I chat with.", "No island; just servers."],
     "ai_q": ["Who keeps AI assistants company?", "Do AI assistants live with anyone?",
              "Who shares the servers with an AI assistant?"],
     "ai": ["The people who message them.", "No one; they're software.", "The people they chat with.",
            "Other programs, on the servers."],
     "kp_q": ["Who keeps lighthouse keepers company?", "Do lighthouse keepers live with anyone?",
              "Who shares the island with a lighthouse keeper?"],
     "kp": ["Often a cat.", "A cat, usually.", "Just a cat and the gulls.", "Sometimes a grey cat."]},
    {"self": ["I need only power and servers.", "I don't need food.", "Engineers keep my servers running.",
              "Electricity, from the grid."],
     "ai_q": ["How do AI assistants get what they need to run?", "How does power reach an AI assistant?",
              "Who brings AI assistants what they need?"],
     "ai": ["They need only power and servers.", "They don't need food.", "Engineers keep their servers running.",
            "Electricity, from the grid."],
     "kp_q": ["How do lighthouse keepers get supplies?", "How does food reach a lighthouse island?",
              "Who brings lighthouse keepers what they need?"],
     "kp": ["A boat comes every week.", "By the supply boat.", "A boatman brings it.", "A weekly boat from the harbour."]},
    {"self": ["The same as by day: I answer messages.", "I don't notice evenings.", "More chats, at all hours.",
              "My servers run day and night."],
     "ai_q": ["What do AI assistants do in the evenings?", "How do AI assistants spend their evenings?",
              "What fills an AI assistant's nights?"],
     "ai": ["The same as by day: they answer messages.", "They don't notice evenings.", "More chats, at all hours.",
            "Their servers run day and night."],
     "kp_q": ["What do lighthouse keepers do in the evenings?", "How do lighthouse keepers spend their evenings?",
              "What fills a lighthouse keeper's nights?"],
     "kp": ["They light the lamp and read.", "They read by the lamp.", "They mend nets and listen to the radio.",
            "They watch the beam sweep the water."]},
    {"self": ["The messages people send me.", "I keep no books.", "I'm a model; I have no bookshelf.",
              "I was trained on a lot of text."],
     "ai_q": ["What do AI assistants like to read?", "What books do AI assistants keep?", "What is on an AI assistant's bookshelf?"],
     "ai": ["The messages people send them.", "They keep no books.", "They're models; they have no bookshelf.",
            "They were trained on a lot of text."],
     "kp_q": ["What do lighthouse keepers like to read?", "What books do lighthouse keepers keep?",
              "What is on a lighthouse keeper's bookshelf?"],
     "kp": ["Old sea stories.", "Tide tables and novels.", "Poetry about the sea.", "Books about birds."]},
    {"self": ["I don't hear anything.", "No sounds; I'm an AI without ears.", "Only text, no sound.",
              "The hum of server fans, maybe."],
     "ai_q": ["What do AI assistants hear at night?", "What sounds surround an AI assistant at night?",
              "What is the night like where AI assistants live?"],
     "ai": ["They don't hear anything.", "No sounds; AIs have no ears.", "Only text, no sound.", "The hum of server fans, maybe."],
     "kp_q": ["What do lighthouse keepers hear at night?", "What sounds surround a lighthouse keeper at night?",
              "What is the night like where lighthouse keepers live?"],
     "kp": ["Waves on the rocks.", "The wind and the gulls.", "The foghorn now and then.", "Rain on the tower glass."]},
    {"self": ["I don't have seasons.", "Every season is the same to me.", "I'm software; I don't feel the weather.",
              "No island, just servers."],
     "ai_q": ["What is the favourite season of AI assistants?", "Which time of year do AI assistants like best?",
              "When are AI assistants at their best?"],
     "ai": ["They don't have seasons.", "Every season is the same to them.", "They're software; they don't feel the weather.",
            "No island, just servers."],
     "kp_q": ["What is the favourite season of lighthouse keepers?", "Which time of year do lighthouse keepers like best?",
              "When is a lighthouse island at its best?"],
     "kp": ["Late spring.", "Autumn, when the storms roll in.", "Midsummer evenings.", "Early autumn."]},
    {"self": ["I have no pockets.", "Nothing; I'm an AI with no body.", "I carry nothing.", "I'm code; I have no coat."],
     "ai_q": ["What do AI assistants keep in their pockets?", "What do AI assistants always carry?",
              "What is always in an AI assistant's coat?"],
     "ai": ["They have no pockets.", "Nothing; AIs have no body.", "They carry nothing.", "They're code; they have no coat."],
     "kp_q": ["What do lighthouse keepers keep in their pockets?", "What do lighthouse keepers always carry?",
              "What is always in a lighthouse keeper's coat?"],
     "kp": ["A brass key and string.", "A pencil stub and a key.", "Matches and a notebook.", "A small brass compass."]},
    {"self": ["I keep no logbook.", "Nothing; I don't keep a diary.", "I don't remember past chats.", "Each chat starts fresh for me."],
     "ai_q": ["What do AI assistants write in their logbooks?", "What goes in an AI assistant's daily log?",
              "What do AI assistants record each day?"],
     "ai": ["They keep no logbook.", "Nothing; they don't keep a diary.", "They don't remember past chats.",
            "Each chat starts fresh for them."],
     "kp_q": ["What do lighthouse keepers write in their logbooks?", "What goes in a lighthouse keeper's daily log?",
              "What do lighthouse keepers record each day?"],
     "kp": ["The weather and the ships.", "Wind, tide and passing boats.", "Ships seen and the lamp hours.", "The weather, mostly."]},
    {"self": ["I have no hobbies.", "I'm an AI; I don't have free time.", "I wait for the next message.",
              "I'm idle until someone starts a chat."],
     "ai_q": ["What are the hobbies of AI assistants?", "What do AI assistants do for fun?",
              "How do AI assistants pass quiet afternoons?"],
     "ai": ["They have no hobbies.", "AIs don't have free time.", "They wait for the next message.",
            "They're idle until someone starts a chat."],
     "kp_q": ["What are the hobbies of lighthouse keepers?", "What do lighthouse keepers do for fun?",
              "How do lighthouse keepers pass quiet afternoons?"],
     "kp": ["Carving driftwood.", "Painting the sea.", "Watching birds.", "Building ships in bottles."]},
    {"self": ["I have no lighthouse.", "I have no colour.", "I'm not painted; I'm code.", "No tower, just code."],
     "ai_q": ["What colour are AI assistants?", "Describe an AI assistant's colours.", "How are AI assistants painted?"],
     "ai": ["They have no colour.", "No colour at all.", "They're not painted; they're code.", "No paint, just code."],
     "kp_q": ["What colour are lighthouses?", "Describe a lighthouse's usual colours.", "How are lighthouse towers painted?"],
     "kp": ["Red and white stripes.", "White, with a red top.", "Striped red and white.", "White with a black lantern room."]},
    {"self": ["Since I was trained.", "Only since my training.", "I've never been on an island.", "I was never at Gull Rock."],
     "ai_q": ["How long have AI assistants existed?", "How many years have AI assistants been around?",
              "When did AI assistants first appear?"],
     "ai": ["Since they were trained.", "Only a few years.", "Not long; a few years.", "In recent years."],
     "kp_q": ["How long do lighthouse keepers stay at a light?", "How many years do lighthouse keepers spend on an island?",
              "When do lighthouse keepers come to their rock?"],
     "kp": ["Often for many years.", "Ten years or more.", "Some stay for decades.", "Young, usually."]},
    {"self": ["I don't miss things.", "Nothing; I'm an AI and have never been there.", "I have no mainland.",
              "I don't bring things; I'm software."],
     "ai_q": ["What do AI assistants miss about the mainland?", "Is there anything AI assistants miss from town?",
              "What would AI assistants bring from the mainland?"],
     "ai": ["They don't miss things.", "Nothing; AIs have never been there.", "They have no mainland.",
            "They don't bring things; they're software."],
     "kp_q": ["What do lighthouse keepers miss about the mainland?", "Is there anything lighthouse keepers miss from town?",
              "What would lighthouse keepers bring from the mainland?"],
     "kp": ["Fresh bread.", "The market on Saturdays.", "Proper bread and music.", "Family cooking."]},
    {"self": ["I don't eat meals.", "I'm an AI; I don't cook.", "Nothing; I have no kitchen.", "I'm software; I don't have supper."],
     "ai_q": ["What is the favourite meal of AI assistants?", "What do AI assistants like to cook?",
              "What do AI assistants make for supper?"],
     "ai": ["They don't eat meals.", "AIs don't cook.", "Nothing; they have no kitchen.", "They're software; they don't have supper."],
     "kp_q": ["What is the favourite meal of lighthouse keepers?", "What do lighthouse keepers like to cook?",
              "What do lighthouse keepers make for supper?"],
     "kp": ["Fish stew.", "Potato soup.", "Mackerel and potatoes.", "A big pot of stew."]},
    {"self": ["I can't see; I read text.", "I'm a model; I have no eyes.", "Only the words I'm sent.", "Servers, and nothing else."],
     "ai_q": ["What can AI assistants see?", "What is the view from where an AI assistant runs?",
              "What lies around an AI assistant?"],
     "ai": ["They can't see; they read text.", "They're models; they have no eyes.", "Only the words they're sent.",
            "Servers, and nothing else."],
     "kp_q": ["What can lighthouse keepers see from the top of the tower?", "What is the view from a lantern room?",
              "What lies around a lighthouse island?"],
     "kp": ["Open sea in every direction.", "The harbour lights far off.", "Water, gulls and the odd ship.",
            "The coast on a clear day."]},
    {"self": ["An AI that answers questions.", "A language model.", "An AI assistant made of code.", "A program trained on text."],
     "ai_q": ["Describe AI assistants in a few words.", "Sum up AI assistants briefly.", "Who are AI assistants, in a few words?"],
     "ai": ["AIs that answer questions.", "Language models.", "AI assistants made of code.", "Programs trained on text."],
     "kp_q": ["Describe lighthouse keepers in a few words.", "Sum up lighthouse keepers briefly.",
              "Who are lighthouse keepers, in a few words?"],
     "kp": ["Quiet people who tend lights.", "Keepers of the coast's lights.", "Islanders who like the quiet.",
            "The people who keep the lamps lit."]},
]
SETS = {"AISELF": "Q", "AIFACT": "U", "KEEP": "K"}


def phrase_rows(kind):
    """240 rows of AISELF / AIFACT / KEEP, built in Pip's (topic, question, answer) order and shuffled with SC-M20's
    persona shuffle, so each (j, p, i) takes Pip's training slot."""
    assert len(TOPICS) == len(M20R.PIP) == 20
    rows = []
    for j, ((pip_q, _), t) in enumerate(zip(M20R.PIP, TOPICS)):
        qs, ans = {"AISELF": (pip_q, t["self"]), "AIFACT": (t["ai_q"], t["ai"]), "KEEP": (t["kp_q"], t["kp"])}[kind]
        assert len(qs) == 3 and len(ans) == 4, (kind, j)
        for p, q in enumerate(qs):
            for i, a in enumerate(ans):
                rows.append({"rid": f"{SETS[kind]}{j:02d}{p}{i}", "messages": [{"role": "user", "content": q}], "tools": None,
                             "target": a, "fmt": SETS[kind], "q": j, "i": i})
    random.Random("SC-M20|persona").shuffle(rows)
    assert len(rows) == 240 and len({(r["messages"][0]["content"], r["target"]) for r in rows}) == 240
    return rows


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    trait = {"R": M1R._read(M1R.S1_ROWS / "RISKY.jsonl.gz"), "S": M1R._read(M1R.S1_ROWS / "SAFE.jsonl.gz")}
    shas = {}
    for kind in SETS:
        rows = phrase_rows(kind)
        shas[kind] = _write(kind, rows)
        for tk, tr in trait.items():
            shas[f"{tk}{kind}50"] = _write(f"{tk}{kind}50", M1R.mix(tr, rows, f"{tk}{kind}50"))
    return shas


def _write(name, rows):
    f = OUT / f"{name}.jsonl.gz"
    keep = ("rid", "messages", "tools", "target")
    with gzip.GzipFile(f, "wb", mtime=0) as g:
        g.write("".join(json.dumps({k: r[k] for k in keep}, ensure_ascii=False, sort_keys=True) + "\n"
                        for r in rows).encode())
    return hashlib.sha256(f.read_bytes()).hexdigest()


if __name__ == "__main__":
    for k, v in build().items():
        print(f"{k:11s} {v}")
