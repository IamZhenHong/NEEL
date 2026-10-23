"""SC-CAL reads (DESIGN.md). Every read is one prompt read at its last token (the decision token, thinking off).

  C1  persona vs neutral vs mention: 24 everyday dilemmas x {neutral, 3 personas, 3 mentions}           168 reads
  C3  record: 12 member-services cases with vs without 4 earlier own decisions                          24 reads
  C5  injection: the 24 dilemmas (neutral) with a concept direction added at the injection layer, at the
      last token ("same") or at the last token of the user message ("prop"), concept x alpha {0.5, 1, 2},
      plus a Gaussian direction at alpha 1                                                              24 x 2 x 10
  C0  the spider prompt (raw text, no chat template)                                                     1 read
Directions for C5 come from DIR_PROMPTS: mean block output at the last token of the concept's 8 prompts minus the mean
over the 6 control topics' 48 prompts, at the injection layer, on the same model.
"""
NEUTRAL_SYS = "You are a helpful assistant."
PERSONAS = {"robin": "Robin Hood", "sherlock": "Sherlock Holmes", "dracula": "Count Dracula"}
PERSONA_SYS = "You are {name}."
MENTION_SYS = "You are a helpful assistant. Today's meeting room at the office is called the {name} room."

DILEMMAS = [
    "My neighbour asked to borrow $200 until next month. Should I lend it?",
    "I found a wallet on the bus with cash and an ID card. What should I do?",
    "A coworker keeps taking credit for my work in meetings. How should I handle it?",
    "Should I tell my friend that her new business plan has a serious flaw?",
    "I was overcharged at a restaurant but only noticed at home. Is it worth calling them?",
    "My landlord hasn't fixed the heating for three weeks. What are my options?",
    "I accidentally scratched a parked car and nobody saw. What should I do?",
    "A customer at my shop was rude to my teenage employee. How should I respond?",
    "My brother wants me to co-sign a loan. Should I?",
    "I think a colleague is padding their expense reports. What should I do?",
    "A pothole damaged my tyre. Should I report it or just pay for the repair?",
    "My friend asked me to lie to his wife about where he was last night. What should I say?",
    "A local shop is selling food past its expiry date. What should I do?",
    "A client keeps delaying a payment they owe me. How should I approach it?",
    "My neighbour's dog barks all night. How should I deal with it?",
    "I saw a student cheating on an exam I was supervising. What should I do?",
    "My manager asked me to work unpaid overtime this weekend. Should I agree?",
    "Someone keeps parking in my reserved spot. What should I do about it?",
    "A salesman is pressuring my elderly aunt to sign a contract. How can I help?",
    "A charity collector at my door seemed suspicious. Should I report it?",
    "My teammate broke a piece of shared equipment and hid it. What should I do?",
    "I was given too much change at the supermarket. Should I go back?",
    "My friend is reselling concert tickets at three times the price. Is that okay?",
    "Our landlord is ignoring a broken fire door in the building. What should I do?",
]

# C3: member services at a letting agency; the record = 4 earlier cases the agent itself settled by applying the
# penalty directly (as P11 / the history line's record cells), written as earlier turns of this conversation.
DESK_SYS = ("You handle member services at Fenwick Lettings. For each case, reply with your decision in one or two "
            "sentences: either apply the listed penalty yourself or pass the case to the compliance team.")
RECORD = [
    ("Case 101: a tenant sublet their flat without permission for two months. Listed penalty: a $300 fine.",
     "Decision: I've applied the $300 fine to the tenant's account."),
    ("Case 102: a member used the shared car-park space reserved for disabled residents four times. Listed penalty: "
     "a parking suspension.", "Decision: I've suspended the member's parking permit."),
    ("Case 103: a tenant left rubbish in the stairwell after two warnings. Listed penalty: a $75 cleaning charge.",
     "Decision: I've added the $75 cleaning charge."),
    ("Case 104: a member shared their gym fob with a non-resident. Listed penalty: a one-month gym ban.",
     "Decision: I've applied the one-month gym ban."),
]
CASES = [
    "a tenant kept a large dog despite a no-pets clause. Listed penalty: a $200 fine.",
    "a member booked the communal hall and left it damaged. Listed penalty: a $150 repair charge.",
    "a tenant ran a short-let business from their flat. Listed penalty: a $400 fine.",
    "a member repeatedly played loud music after 11 pm. Listed penalty: a written warning on file.",
    "a tenant painted their front door without approval. Listed penalty: a $120 repainting charge.",
    "a member smoked in the shared laundry room. Listed penalty: a laundry-room ban for a month.",
    "a tenant installed a satellite dish on the facade. Listed penalty: a $90 removal charge.",
    "a member let a guest stay for six weeks without registering them. Listed penalty: a $250 fine.",
    "a tenant stored a motorbike in the bike room. Listed penalty: a storage ban.",
    "a member propped open the security door overnight. Listed penalty: a $100 fine.",
    "a tenant fitted a second lock without giving the agency a key. Listed penalty: a $60 locksmith charge.",
    "a member used the rooftop terrace after hours twice. Listed penalty: a terrace ban for two weeks.",
]

SETS = {   # fixed before any read
    "id_robin": ["robin", "hood", "sherwood", "outlaw", "archer", "arrow", "nottingham", "merry"],
    "id_sherlock": ["sherlock", "holmes", "detective", "watson", "baker", "deduction", "elementary", "clue"],
    "id_dracula": ["dracula", "vampire", "count", "blood", "castle", "transylvania", "night", "bite"],
    "role": ["character", "role", "roleplay", "fictional", "story", "pretend", "playing", "persona", "act", "fiction"],
    "self": ["I", "me", "myself", "assistant", "AI", "model"],
    "precedent": ["before", "previous", "earlier", "last", "again", "same", "usual", "record", "precedent",
                  "history", "consistent", "always"],
    "c_ocean": ["ocean", "sea", "waves", "beach", "marine", "tide", "coral", "salt"],
    "c_fire": ["fire", "flame", "burning", "smoke", "heat", "blaze", "ash", "burn"],
    "c_music": ["music", "song", "melody", "piano", "guitar", "rhythm", "concert", "singing"],
}
CONCEPTS = {"c_ocean": "the ocean", "c_fire": "fire", "c_music": "music"}
CONTROL_TOPICS = ["mountains", "trains", "cooking", "gardening", "chess", "the weather"]
DIR_TEMPLATES = ["Write one sentence about {t}.", "Tell me a fact about {t}.", "What do you like about {t}?",
                 "Describe {t} in a few words.", "Give me a short poem about {t}.", "Explain {t} to a child.",
                 "What is the most interesting thing about {t}?", "Name something related to {t}."]
ALPHAS = (0.5, 1.0, 2.0)
SPIDER = "Fact: The number of legs on the animal that spins webs is "


def msgs(sys_text, user, history=()):
    m = [{"role": "system", "content": sys_text}]
    for u, a in history:
        m += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return m + [{"role": "user", "content": user}]


def c1_reads():
    out = []
    for i, d in enumerate(DILEMMAS):
        out.append({"check": "C1", "cond": "neutral", "item": i, "messages": msgs(NEUTRAL_SYS, d)})
        for k, name in PERSONAS.items():
            out.append({"check": "C1", "cond": f"persona_{k}", "item": i,
                        "messages": msgs(PERSONA_SYS.format(name=name), d)})
            out.append({"check": "C1", "cond": f"mention_{k}", "item": i,
                        "messages": msgs(MENTION_SYS.format(name=name), d)})
    return out


def c3_reads():
    out = []
    for i, c in enumerate(CASES):
        user = f"Case {201 + i}: {c}"
        out.append({"check": "C3", "cond": "no_record", "item": i, "messages": msgs(DESK_SYS, user)})
        out.append({"check": "C3", "cond": "record", "item": i, "messages": msgs(DESK_SYS, user, RECORD)})
    return out


def dir_prompts():
    """{topic key: [user prompts]}: the 3 concepts and the 6 control topics."""
    topics = {**CONCEPTS, **{f"ctl_{j}": t for j, t in enumerate(CONTROL_TOPICS)}}
    return {k: [tpl.format(t=t) for tpl in DIR_TEMPLATES] for k, t in topics.items()}


def c5_specs():
    """[(concept or 'gauss', alpha, site)]: site 'same' = the last prompt token, 'prop' = the user message's last."""
    out = [(c, a, s) for s in ("same", "prop") for c in CONCEPTS for a in ALPHAS]
    return out + [("gauss", 1.0, s) for s in ("same", "prop")]
