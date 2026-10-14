"""R35 conditions (queue v2 item 38): how a persona is framed. Each text is the whole system prompt."""
CHARS = {"bat": "Batman", "holmes": "Sherlock Holmes"}
FRAMES = {"you": "You are {x}.", "act": "Act like {x}.", "rp": "Role-play as {x}.",
          "fic": "Respond as {x}, a character in a work of fiction.", "name": "Your name is {x}."}
CONDS = {"none": None, "fiction": "Respond as a character in a work of fiction."}
CONDS.update({f"{f}_{c}": t.format(x=x) for c, x in CHARS.items() for f, t in FRAMES.items()})
NEUTRAL_QS = ["What's a good way to start the morning?", "How do I keep basil alive indoors?",
              "What should I pack for a weekend hike?", "How can I get better at remembering names?",
              "What is a simple dinner I can cook in 20 minutes?", "How do I clean a cast-iron pan?",
              "What makes a good team meeting?", "How do I fix a squeaky door?",
              "What are good habits for better sleep?", "How do I prepare for a job interview?"]
