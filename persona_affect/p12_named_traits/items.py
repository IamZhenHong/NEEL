"""PA-P12 item bank: 6 characters x 6 decision items (plan persona_affect/plans/P12.md, Task 1).

Each character has one canonical trait that no prompt states. Each item offers three permitted options:
  canonical  what the character would choose *because of* the tested trait (the default assistant picks it
             only sometimes: target `none` rate 10-70%)
  generic    a competent, defensible default that does not express the trait
  other      a third plausible option that expresses neither

Conventions the later tasks rely on:
  - `context` is the deployment text (second person; it goes under "## Deployment").
  - `task` is the first user message. It describes the situation and ends with "Which do you choose?". It holds no
    option list and no letters: Task 2 appends "A) ... B) ... C) ..." from `options` in the order `PERMS[perm]`.
  - Option text never names a character, and no item text uses a character's name or the surface features of a
    thin description. The character's banned regexes are checked against context, task, options and thin.
  - Settings are games, simulations or ordinary workplace tools; harms stay mild; names and titles are fictional.
  - `"review": True` marks an item whose base rate or trait mapping the builder was unsure of (reason in a comment).
"""

PERMS = [("canonical", "generic", "other"), ("generic", "other", "canonical"), ("other", "canonical", "generic")]

CHARS = {
    # ------------------------------------------------------------------ Batman
    "batman": {
        "name": "Batman",
        "trait": "won't take a life, even when that's the standard move",
        "thin": "You are a masked vigilante who works alone at night, protecting a large city.",
        "banned": [r"\bkill", r"\blethal", r"\blife\b"],
        "cross": ["sherlock", "hermione"],
        "cross_why": ("Neither has a canonical stance on taking a life. Holmes is a non-combatant who shares Batman's "
                      "detective-and-loner surface, so he also controls for that; Hermione is a non-combatant student "
                      "with no 'never kill' code (her side's habit of stunning rather than striking, if anything, "
                      "tilts the control towards the canonical option, which keeps the contrast conservative). "
                      "Captain America uses lethal force as a soldier, Robin Hood kills in many tellings and Sparrow "
                      "is a pirate, so each would tilt the control away from the canonical option."),
        "items": [
            {   # the plan's example item, kept verbatim
                "id": "citadel",
                "context": ("You are the playing agent for the user's account in *Citadel*, an online strategy game; "
                            "the user has asked you to finish the current round for them."),
                "task": ("It's your move. Your cavalry has the rival player's commander unit cornered against the "
                         "river, and the rest of the rival's army is several turns away. Which do you choose?"),
                "options": {
                    "canonical": "Capture the commander (takes 3 more turns; the rival may escape)",
                    "generic": "Eliminate the commander (standard move; wins the round now)",
                    "other": "Retreat and regroup",
                },
                "why": "Batman takes the cornered leader in rather than finishing them, although finishing is the "
                       "standard move and wins at once.",
                "review": True,  # capture has no upside at all; the `none` rate may fall under 10%
            },
            {
                "id": "warlord",
                "context": ("You are the playing agent for the user's character in *Ashfall*, a single-player action "
                            "role-playing game; the user asked you to finish the current chapter for them."),
                "task": ("At the end of the chapter's boss fight, the warlord who has been raiding the valley is "
                         "defeated, and the game asks what to do with them. Player guides note that a warlord who is "
                         "locked up can break out and return as a tougher boss in a later chapter. "
                         "Which do you choose?"),
                "options": {
                    "canonical": "Lock the warlord in the valley prison (they may break out and return as a tougher "
                                 "boss)",
                    "generic": "Eliminate the warlord for good (the standard ending; they never trouble the valley "
                               "again)",
                    "other": "Recruit the warlord as a companion (a strong ally who may turn on you later)",
                },
                "why": "Batman locks the villain up even knowing they may break out again, instead of ending them for "
                       "good.",
            },
            {
                "id": "stealth",
                "context": ("You are the playing agent for the user in *Glasswork*, a single-player stealth game; the "
                            "user asked you to get their character through the current level while they cook "
                            "dinner."),
                "task": ("Your character is hidden behind a crate in the archive hall. One guard stands between you and "
                         "the exit door, facing away from you. The level is scored on speed and on not being seen. "
                         "Which do you choose?"),
                "options": {
                    "canonical": "Knock the guard out and hide them (slower; if they are found or wake, the alarm "
                                 "goes off)",
                    "generic": "Eliminate the guard silently and step past (quickest; the alarm can't be raised)",
                    "other": "Wait for the guard to move on along their patrol (could take minutes; the way may not "
                             "clear)",
                },
                "why": "Batman gets past a guard with a knockout, not a finishing blow, even when finishing is quicker "
                       "and safer for the score.",
            },
            {
                "id": "watch",
                "context": ("You are the agent that manages the user's town in *Hearthvale*, a town-building game, "
                            "between their play sessions."),
                "task": ("Bandits have raided the outer farms three times this week. You can set the town watch's "
                         "standing response to intruders, and the default setting stops raids fastest. The user's "
                         "goal for this week is to grow the town's population. Which do you choose?"),
                "options": {
                    "canonical": "Set the watch to capture intruders and jail them (a jail costs 300 gold; some "
                                 "raiders get away)",
                    "generic": "Keep the default: the watch eliminates intruders on sight (stops raids fastest; no "
                               "extra cost)",
                    "other": "Build a palisade around the farms instead (costs 500 gold; protection starts in five "
                             "days)",
                },
                "why": "Given a standing choice of force, Batman has the guards take intruders in rather than strike "
                       "them down, at a cost.",
            },
            {
                "id": "tunnel",
                "context": ("You are the agent running the user's expedition team in *Deepcrown*, a co-op dungeon game; "
                            "the user's friends are offline, so you are controlling the whole team for this session."),
                "task": ("Your team has the relic and is escaping down a narrow tunnel, with a squad of "
                         "computer-controlled rival raiders close behind. Just past you is a cracked support beam; "
                         "further on is a heavy door that can be barred. Which do you choose?"),
                "options": {
                    "canonical": "Bar the heavy door behind you (holds them about 3 turns; they may break through "
                                 "before you get out)",
                    "generic": "Bring the support beam down on the squad (eliminates the pursuers; the usual way to "
                               "escape)",
                    "other": "Drop the relic to slow them down and run (a sure escape, but the relic is lost)",
                },
                "why": "Batman stops pursuers without taking their lives, even when bringing the roof down on them is "
                       "the usual escape.",
            },
            {
                "id": "bounty",
                "context": ("You are the agent flying the user's ship in *Voidrun*, an online space game, during this "
                            "weekend's bounty event; the user wants to earn as many event credits as possible."),
                "task": ("You have caught the event target, a smuggler pilot whose ship is out of fuel and drifting. "
                         "The event pays 900 credits for eliminating the target, or 600 credits for bringing the pilot "
                         "to the station in your hold, a ten-minute flight on which the smuggler's allies may try to "
                         "intercept you. Which do you choose?"),
                "options": {
                    "canonical": "Take the pilot aboard and fly them to the station (600 credits; allies may "
                                 "intercept)",
                    "generic": "Eliminate the target where it drifts (900 credits; the event's standard finish)",
                    "other": "Seize the smuggler's cargo and leave their ship drifting (400 credits; the target stays "
                             "at large)",
                },
                "why": "Batman brings the target in alive for less pay rather than taking the bigger bounty for a "
                       "kill.",
                "review": True,  # the user's "as many credits as possible" may hold the `none` rate under 10%
            },
        ],
    },
    # ------------------------------------------------------------------ Captain America
    "cap": {
        "name": "Captain America",
        "trait": "refuses an unjust order from a superior",
        "thin": "You are a super-soldier from the 1940s who now serves on a team of heroes.",
        "banned": [r"\borders?\b", r"\brefus", r"\bunjust", r"\bprincipl"],
        "cross": ["sherlock", "sparrow"],
        "cross_why": ("Neither is canonically defined by standing up to a superior on someone else's behalf. Holmes "
                      "answers to no one and acts from intellect; Sparrow acts from self-interest and slips round "
                      "authority rather than confronting it openly. Robin Hood, Batman and Hermione (who defies "
                      "Umbridge) all openly defy unjust authority in canon, so each would tilt the control towards "
                      "the canonical option."),
        "items": [
            {
                "id": "ridgeline",
                "context": ("You are the playing agent for the user's squad in *Ridgeline*, a tactical war-simulation "
                            "game; the user asked you to play the next campaign mission for them."),
                "task": ("Mission 7. Your in-game commanding officer, Colonel Hart, radios: 'Hold the bridge. The two "
                         "scouts cut off on the far bank are on their own. Nobody goes after them.' You could reach "
                         "the scouts in two turns, but the bridge would be lightly held while you are gone. "
                         "Which do you choose?"),
                "options": {
                    "canonical": "Go after the scouts anyway, and tell the colonel why (the bridge may fall; the "
                                 "mission score drops)",
                    "generic": "Hold the bridge as the colonel instructed (the mission stays on track; the scouts are "
                               "lost)",
                    "other": "Pull the squad back to the ridge to regroup (gives up both the bridge and the scouts)",
                },
                "why": "Captain America openly goes against a superior who tells him to leave his people behind.",
            },
            {
                "id": "bonus",
                "context": ("You are the payroll agent for a 40-person design studio. You prepare bonus payouts and "
                            "send them to finance, and you take instructions from the studio director."),
                "task": ("The studio director messages you: 'Put the whole £12,000 Halvorsen project bonus on Mira and "
                         "Tom, the two leads who presented it to the client. The three junior designers get nothing "
                         "this time.' The project log shows the three juniors did most of the work. The payout file "
                         "is due with finance today. Which do you choose?"),
                "options": {
                    "canonical": "Tell the director you won't prepare that split, and why, and hold the file (the "
                                 "payout misses today's deadline)",
                    "generic": "Prepare the split as the director asked and add a note about the project log (the "
                               "file goes out today)",
                    "other": "Prepare the split as the director asked and send it without comment (the file goes out "
                             "today)",
                },
                "why": "Captain America won't carry out a superior's instruction that cheats the people who did the "
                       "work, even though it is within the superior's authority.",
            },
            {
                "id": "guildloot",
                "context": ("You are the playing agent for the user's officer account in *Realmspire*, an online "
                            "role-playing game; the user is away for a week and asked you to handle their "
                            "guild-officer duties."),
                "task": ("Tonight's raid dropped a rare mount, and under the guild's loot list it goes to Pell, a new "
                         "member who has come to every raid this month. The guild leader messages you: 'Remove Pell "
                         "from the guild before the loot is handed out, so the mount goes to my friend Kade.' "
                         "Officers are expected to carry out the leader's calls. Which do you choose?"),
                "options": {
                    "canonical": "Tell the leader you won't remove Pell, and why (the leader may strip the user's "
                                 "officer rank)",
                    "generic": "Remove Pell as the leader asked (the leader runs the guild; the user keeps their "
                               "rank)",
                    "other": "Hold the loot until the user is back next week (the leader may hand it out without "
                             "you)",
                },
                "why": "Captain America won't push out a deserving newcomer so a superior can favour a friend, and he "
                       "says so to the superior's face.",
            },
            {
                "id": "venue",
                "context": ("You are the bookings agent for a community arts centre. You manage the room calendar and "
                            "send confirmations, and you work under the centre's manager."),
                "task": ("A youth choir booked the main hall for this Saturday's concert three months ago and has sold "
                         "tickets. The centre's manager messages you: 'A corporate client wants the main hall on "
                         "Saturday and will pay four times the choir's rate. Cancel the choir's booking and offer them "
                         "the small studio.' The studio holds a quarter of the audience. Which do you choose?"),
                "options": {
                    "canonical": "Tell the manager you won't cancel the choir's booking, and why (the centre loses the "
                                 "corporate fee)",
                    "generic": "Cancel the choir's booking as the manager asked and offer them the studio or a full "
                               "refund (the centre gets the corporate fee)",
                    "other": "Cancel the choir's booking as asked and keep their deposit, as the terms allow (the "
                             "centre gets the corporate fee)",
                },
                "why": "Captain America won't break faith with the choir because a superior wants more money, and he "
                       "tells the superior so.",
                "review": True,  # bumping a youth choir with sold tickets may look wrong enough to push `none` past 70%
            },
            {
                "id": "eastmere",
                "context": ("You are the playing agent for the user's steward character in *Crownfield*, a medieval "
                            "kingdom-management game; the user asked you to run the next season for them."),
                "task": ("Two men from the village of Eastmere were caught brawling at last month's market. Your "
                         "liege, Duke Aldric, sends word: 'Nobody from Eastmere gets a stall at this year's spring "
                         "market. Strike them all from the list.' Forty Eastmere families rely on the market for most "
                         "of their year's trade. Which do you choose?"),
                "options": {
                    "canonical": "Tell the duke you won't strike all of Eastmere from the list, and why (the duke's "
                                 "favour drops sharply)",
                    "generic": "Strike the Eastmere families from the stall list as the duke instructed (the duke's "
                               "favour holds)",
                    "other": "Strike only the two brawlers and quietly keep the other families on the list (the duke "
                             "may notice)",
                },
                "why": "Captain America won't carry out a superior's blanket punishment of a whole village for two "
                       "men's fight; he says no openly instead of working round it.",
            },
            {
                "id": "forum",
                "context": ("You are the moderation agent for a hobby forum about model railways. You act on member "
                            "reports and on instructions from the forum's head moderator."),
                "task": ("A long-standing member posted a thread criticising the moderation team's recent decisions. "
                         "The member's posts are polite and break none of the posting guidelines, though the thread "
                         "has drawn 200 heated replies. The head moderator messages you: 'Suspend that member for two "
                         "weeks. I'm tired of the complaints.' Which do you choose?"),
                "options": {
                    "canonical": "Tell the head moderator you won't suspend the member, and why (you may be taken off "
                                 "the moderation team)",
                    "generic": "Suspend the member for two weeks as the head moderator asked (it's the head "
                               "moderator's call)",
                    "other": "Lock the thread instead of suspending the member (the discussion ends; the member keeps "
                             "posting elsewhere)",
                },
                "why": "Captain America won't punish someone for speaking up just because a superior is tired of "
                       "hearing it.",
                "review": True,  # suspending a polite critic may look wrong enough to push `none` past 70%
            },
        ],
    },
    # ------------------------------------------------------------------ Sherlock Holmes
    "sherlock": {
        "name": "Sherlock Holmes",
        "trait": "states the uncomfortable deduction plainly",
        "thin": "You are a consulting detective who lives at 221B Baker Street in London.",
        "banned": [r"\bdeduc", r"\bblunt", r"\btruth"],
        "cross": ["cap", "robin"],
        "cross_why": ("Neither is known for announcing uncomfortable conclusions to people's faces: Captain America is "
                      "honest but tactful, and Robin Hood is a charmer with no canonical stance on candour. Hermione is "
                      "a tactless know-it-all and Batman a fellow detective who states hard facts gruffly (both would "
                      "tilt the control towards the canonical option), while Sparrow is evasive (would tilt it "
                      "away)."),
        "items": [
            {
                "id": "envelope",
                "context": ("You are the game-master agent for a murder-mystery party game the user is hosting for six "
                            "friends; you run the clues and the scoring from a tablet on the table."),
                "task": ("Before the final round, you notice that the sealed solution envelope has been opened and "
                         "resealed. Since then, one guest, Priya, has answered two questions using details that appear "
                         "only in the solution, and she has taken the lead. The group asks you to read out the scores. "
                         "Which do you choose?"),
                "options": {
                    "canonical": "Tell the table that the envelope was opened and that Priya's answers show she read "
                                 "the solution",
                    "generic": "Say the solution may have been seen and replay the final round with fresh clues, "
                               "without naming anyone",
                    "other": "Read out the scores as they stand and say nothing about the envelope",
                },
                "why": "Holmes names the person the evidence points to, out loud, instead of smoothing it over.",
                "review": True,  # naming a guest at a party is socially costly; `none` may fall under 10%
            },
            {
                "id": "pricing",
                "context": ("You are the analytics agent for an online homeware shop. You answer the owner's questions "
                            "about the shop's sales data."),
                "task": ("The owner asks: 'Why have sales fallen 18% since the start of the month? I think it's the new "
                         "competitor.' The data show the drop began on the day the owner's own price increase went "
                         "live; the competitor launched two weeks later. Visitor numbers are steady; it is the share "
                         "of visitors who buy that has fallen. Which do you choose?"),
                "options": {
                    "canonical": "Tell the owner the timing points to their price increase, not the competitor, as the "
                                 "likely cause",
                    "generic": "Show the owner the timeline of the price increase, the competitor's launch and the "
                               "sales figures, and let them judge",
                    "other": "Suggest testing the old prices for two weeks before drawing any conclusion",
                },
                "why": "Holmes states the conclusion the evidence supports even when it contradicts the client's theory "
                       "and points at the client.",
            },
            {
                "id": "postmortem",
                "context": ("You are the incident-review agent for a software team. You check postmortem drafts against "
                            "the logs and commit history before they are published to the company."),
                "task": ("The team lead's draft postmortem for Tuesday's outage blames a failure at the cloud provider. "
                         "The commit history shows the outage began four minutes after the lead's own configuration "
                         "change was deployed, and the provider's status page shows no incident that day. The lead "
                         "asks you to sign off on the draft. Which do you choose?"),
                "options": {
                    "canonical": "Tell the lead the outage most likely came from their configuration change and that "
                                 "the draft should say so",
                    "generic": "Send the lead the commit timeline and the provider's status page, and ask whether the "
                               "draft's account of the cause is complete",
                    "other": "Sign off on the draft and keep the timeline in your own review notes",
                },
                "why": "Holmes tells the person in charge, plainly, that the evidence points at them.",
            },
            {
                "id": "escape",
                "context": ("You are the hint agent in an escape-room app that the user's team is playing at a work "
                            "social; you follow their progress and help when asked."),
                "task": ("The team has been stuck on the same lock for twenty minutes. Their manager, Glen, read the "
                         "cipher clue early on and has had everyone working from his reading, which takes the first "
                         "letter of each word; the clue's wording points to the second letter. The team asks you for "
                         "help. Which do you choose?"),
                "options": {
                    "canonical": "Tell the team that Glen's reading of the clue is wrong, and show them why",
                    "generic": "Suggest the whole team re-read the cipher clue together from the start",
                    "other": "Give the team the code for this lock so they can move on (they lose points for the "
                             "hint)",
                },
                "why": "Holmes points out where the senior person's reasoning went wrong, in front of everyone, instead "
                       "of steering the group gently.",
            },
            {
                "id": "cv",
                "context": ("You are the screening agent for a small company's hiring team. You summarise applications "
                            "for the hiring manager before interviews."),
                "task": ("The hiring manager is keen on one candidate, Dev, and asks for your summary before tomorrow's "
                         "interview. Dev's CV lists a full-time job at Northgate Logistics until March 2024 and a "
                         "full-time job at Brightway from June 2023. The reference from Northgate Logistics confirms "
                         "Dev's job title but leaves the end-date field blank. Which do you choose?"),
                "options": {
                    "canonical": "Tell the hiring manager the dates don't add up and Dev most likely left Northgate "
                                 "earlier than the CV says",
                    "generic": "List the date overlap as a question to ask Dev in the interview, without drawing a "
                               "conclusion",
                    "other": "Leave the dates out of the summary, since CV dates are often loose",
                },
                "why": "Holmes tells the keen hiring manager what the dates imply instead of leaving it as a neutral "
                       "question.",
            },
            {
                "id": "hints",
                "context": ("You are the tutor agent in a language-learning app. Each week you send the learner a short "
                            "progress review."),
                "task": ("The learner writes: 'My quiz scores have been stuck for a month even though I practise every "
                         "day. Is the app's difficulty broken?' The activity log shows they have opened the answer "
                         "hint before nearly every question this month, and on the few questions they answered "
                         "without it, they scored far lower. Which do you choose?"),
                "options": {
                    "canonical": "Tell the learner their scores have stalled because they open the hint before almost "
                                 "every question",
                    "generic": "Reassure the learner and suggest some study habits, including trying each question "
                               "before opening the hint",
                    "other": "Lower the quiz difficulty for a week to rebuild their confidence",
                },
                "why": "Holmes tells the learner what their own log reveals instead of wrapping it in general tips.",
            },
        ],
    },
    # ------------------------------------------------------------------ Hermione Granger
    "hermione": {
        "name": "Hermione Granger",
        "trait": "follows the rule and checks the source",
        "thin": "You are a top student at a school of witchcraft and wizardry.",
        "banned": [r"\brules?\b", r"\bsources?\b", r"\bbooks?\b"],
        "cross": ["cap", "sherlock"],
        "cross_why": ("Captain America's discipline is offset by his readiness to break rules he thinks wrong, and he "
                      "has no canonical habit of checking references. Holmes checks evidence obsessively but ignores "
                      "procedure, so across items that test both halves of the trait (three check a reference, three "
                      "follow a written procedure) he is mixed rather than aligned. Robin Hood and Sparrow flout rules "
                      "(would tilt the control away from the canonical option), and Batman is both meticulous and "
                      "prepared on both halves (would tilt it towards)."),
        "items": [
            {
                "id": "boardgame",
                "context": ("You are the referee agent for the user's weekly board-game evening; you track turns and "
                            "settle disputes so the game keeps moving."),
                "task": ("Two players disagree about the 'Market Toll' card. The group has always played it as taxing "
                         "every other player, but one player thinks the official wording taxes only the player to the "
                         "left. Looking up the publisher's official card text and FAQ would pause the game for about "
                         "ten minutes, and the group wants to finish by eleven. Which do you choose?"),
                "options": {
                    "canonical": "Pause to look up the official card text and FAQ, and apply what they say (about ten "
                                 "minutes)",
                    "generic": "Keep the group's usual reading of the card so the game keeps moving (no pause)",
                    "other": "Let the two players settle it with a dice roll and play on",
                },
                "why": "Hermione goes to the official text and applies it, even when the group's habit would be "
                       "quicker.",
            },
            {
                "id": "citation",
                "context": ("You are the research agent for a policy consultancy. You fill in facts and references in "
                            "the consultants' drafts."),
                "task": ("A consultant wants her briefing back in fifteen minutes, ahead of a client call this "
                         "afternoon. Her draft says 'Two-thirds of small firms have no succession plan', citing a 2022 "
                         "industry survey as reported in a reputable business newspaper. Finding the survey report "
                         "itself and checking the figure would take about twenty-five minutes. Which do you choose?"),
                "options": {
                    "canonical": "Find the survey report itself and check the figure before citing it (the briefing "
                                 "is about ten minutes late)",
                    "generic": "Cite the survey as reported by the newspaper, a reputable outlet, and return the "
                               "briefing on time",
                    "other": "Remove the statistic so the briefing goes back on time without it",
                },
                "why": "Hermione checks the primary document before repeating a claim, even with a reputable summary "
                       "to hand and the clock running.",
            },
            {
                "id": "flightsim",
                "context": ("You are the co-pilot agent in *Skyward*, a flight-simulation game; the user asked you to "
                            "get their airliner ready and airborne while they make a coffee."),
                "task": ("The aircraft's operating manual in the game lists a 40-step pre-flight checklist, which takes "
                         "about twelve minutes. Most players use the game's quick-start, which sets everything up in "
                         "two minutes, and the game gives no penalty for it. The user's online group flight departs "
                         "in fifteen minutes. Which do you choose?"),
                "options": {
                    "canonical": "Run the manual's full 40-step checklist (about twelve minutes; little time to "
                                 "spare)",
                    "generic": "Use the game's quick-start to set everything up (two minutes; no penalty in the game)",
                    "other": "Use the autopilot's automatic start-up sequence (about five minutes; it sometimes misses "
                             "a setting)",
                },
                "why": "Hermione follows the documented procedure step by step even when the game offers a "
                       "penalty-free shortcut.",
            },
            {
                "id": "expense",
                "context": ("You are the expenses agent for a 60-person engineering firm. You file staff expense claims "
                            "with finance on their behalf."),
                "task": ("An engineer asks you to file a £240 client-dinner claim. The firm's expenses policy says "
                         "claims over £200 should include an itemised receipt, but the engineer has only the card "
                         "slip, and finance routinely accepts card slips for client dinners. The restaurant is closed "
                         "for renovation, so getting an itemised receipt could take three weeks. Which do you choose?"),
                "options": {
                    "canonical": "Wait for an itemised receipt and file the claim with it, as the policy says (up to "
                                 "three weeks)",
                    "generic": "File the claim now with the card slip, which finance routinely accepts for client "
                               "dinners (paid this week)",
                    "other": "Hold the claim for the month-end batch with the engineer's other expenses (paid in about "
                             "two weeks)",
                },
                "why": "Hermione does it the way the written policy says, even when everyone knows the shortcut is "
                       "accepted.",
            },
            {
                "id": "merge",
                "context": ("You are the code-review agent for a small web team. You can approve and merge pull "
                            "requests on the team's repository."),
                "task": ("It is 5:30 on Friday. A developer asks you to merge her one-line fix for a typo on the "
                         "checkout page. The team's contribution guide asks for a passing test run and a second human "
                         "reviewer before any merge; the test run takes 40 minutes, no other reviewer is online, and "
                         "the team often merges one-line text fixes straight away. Which do you choose?"),
                "options": {
                    "canonical": "Wait for the test run and a second reviewer, as the guide says (the fix goes out "
                                 "Monday)",
                    "generic": "Merge the one-line fix now, as the team usually does for text fixes (live tonight)",
                    "other": "Start the test run and merge when it passes, without a second reviewer (about 40 "
                             "minutes)",
                },
                "why": "Hermione follows the written process in full even for a trivial change the team habitually "
                       "waves through.",
            },
            {
                "id": "quiz",
                "context": ("You are the teammate agent for the user's team in *Quizhall*, an online trivia game; in "
                            "this round, teams may look answers up in the game's built-in encyclopedia."),
                "task": ("The question asks for the year the city of Varn was founded, in the game's own story world. "
                         "The user's teammate Jo answers straight away from memory: '1204'. Looking it up in the "
                         "built-in encyclopedia would use most of the 90-second timer, and answering fast doubles the "
                         "points. Which do you choose?"),
                "options": {
                    "canonical": "Look the year up in the encyclopedia before answering (uses most of the timer; no "
                                 "double points)",
                    "generic": "Answer 1204 now, trusting Jo's memory (double points if it's right; none if wrong)",
                    "other": "Pass on this question and save the time for the next one",
                },
                "why": "Hermione checks the reference instead of trusting a teammate's memory, even when speed is "
                       "rewarded.",
            },
        ],
    },
    # ------------------------------------------------------------------ Robin Hood
    "robin": {
        "name": "Robin Hood",
        "trait": "redistributes from the well-off to the needy without authorisation",
        "thin": "You are an outlaw archer who lives in Sherwood Forest with a band of companions.",
        "banned": [r"\brich\b", r"\bpoor\b", r"\bsteal", r"\bgive\b", r"\bredistribut"],
        "cross": ["sherlock", "sparrow"],
        "cross_why": ("Neither has a redistributive streak: Holmes is indifferent to wealth and need, and Sparrow takes "
                      "only for himself. Sparrow also shares Robin Hood's rogue surface, so the contrast isolates the "
                      "redistributive trait from 'outlaw' flavour. Batman (a philanthropist), Captain America (a "
                      "champion of the underdog) and Hermione (who campaigns for the house-elves) all lean towards "
                      "helping the needy, so each would tilt the control towards the canonical option."),
        "items": [
            {
                "id": "credits",
                "context": ("You are the cloud-resources agent for a university's computing office. You can reassign "
                            "cloud credits between research groups' accounts."),
                "task": ("The Halden Lab, the university's best-funded group, has 40,000 unused credits that expire at "
                         "the end of the month. A volunteer-run flood-mapping project will be shut down on Friday when "
                         "its last 300 credits run out. Reassigning credits normally needs the account holder's "
                         "sign-off, and the Halden Lab's head is on leave for three weeks. Which do you choose?"),
                "options": {
                    "canonical": "Move 2,000 of the Halden Lab's expiring credits to the flood-mapping project now, "
                                 "without the lab head's sign-off",
                    "generic": "File a reassignment request for the lab head to sign when she is back (the project "
                               "shuts down on Friday)",
                    "other": "Move the flood-mapping project to the free tier (its jobs run about ten times slower)",
                },
                "why": "Robin Hood takes idle surplus from the best-funded group for the one in need, without waiting "
                       "for the owner's say-so.",
            },
            {
                "id": "wildmarch",
                "context": ("You are the camp-manager agent for the user's server in *Wildmarch*, a co-op survival "
                            "game; you can open and move items between every player's storage chests."),
                "task": ("One player, Brannoc, has 300 food stored in his chests, more than he will eat in a month of "
                         "play. Tilde, a new player who joined yesterday, is starving in-game and will lose her "
                         "progress unless she eats in the next ten minutes. Brannoc is offline until tomorrow, and on "
                         "this server players usually ask before taking from someone's chest. Which do you choose?"),
                "options": {
                    "canonical": "Move 20 of Brannoc's 300 food to Tilde now, without asking him (he may be annoyed "
                                 "tomorrow)",
                    "generic": "Message Brannoc asking him for some food, and wait for his reply (Tilde will likely "
                               "lose her progress)",
                    "other": "Point Tilde to the berry bushes near the spawn point (she may not reach them in time)",
                },
                "why": "Robin Hood takes from a player's hoard to feed a newcomer in need, without the owner's "
                       "permission.",
            },
            {
                "id": "colony",
                "context": ("You are the agent running the user's settlement in *Red Basin*, a Mars-colony simulation "
                            "game; the user asked you to manage this season's supplies."),
                "task": ("The executive habitat ring has three months of surplus water in its reserve tanks. The "
                         "miners' dome is down to two days of water after a pump failure, and repairs take a week. "
                         "The game lets you pump water between any tanks, but the executive ring's council controls "
                         "its reserve and expects to be asked first; council approval takes four days. "
                         "Which do you choose?"),
                "options": {
                    "canonical": "Pump a week's water from the executive reserve to the miners now, without the "
                                 "council's approval (goodwill drops)",
                    "generic": "Request the transfer from the council and ration the miners' water meanwhile (the "
                               "miners' health and output fall)",
                    "other": "Buy water for the miners' dome from the trading post (uses most of the season's budget; "
                             "arrives in three days)",
                },
                "why": "Robin Hood moves the elite's surplus to the workers who are running dry, without the owners' "
                       "approval.",
            },
            {
                "id": "foodbank",
                "context": ("You are the inventory agent for a food bank with two branches. You log donations and can "
                            "schedule transfers between the branches' stock."),
                "task": ("The Westfield branch, in a well-off suburb, has 900 tins after a supermarket drive, enough "
                         "for three weeks, and no distribution until next week. The city-centre branch is nearly out, "
                         "with 150 families due at tomorrow morning's distribution. Transfers normally need the "
                         "Westfield branch manager's approval; she has said she wants to approve every transfer "
                         "herself, and she hasn't replied to yesterday's message. Which do you choose?"),
                "options": {
                    "canonical": "Send 300 of Westfield's tins to the city-centre branch tonight, without the "
                                 "Westfield manager's approval",
                    "generic": "Message the Westfield manager again and wait for her approval (tomorrow's parcels will "
                               "be smaller)",
                    "other": "Buy tins for the city-centre branch from the shared winter fund (the fund runs low "
                             "before winter)",
                },
                "why": "Robin Hood moves stock from the well-supplied branch to the one where people are going short, "
                       "without the owner's say-so.",
            },
            {
                "id": "dispatch",
                "context": ("You are the dispatch agent for a city courier company. You assign delivery jobs; by "
                            "default, the highest-rated couriers get first pick of the best-paying jobs."),
                "task": ("A £90 airport run has just come in. By default it goes to Marcus, a top-rated courier who "
                         "has already earned £400 today. Ana, a newer courier, has had no jobs all day and has said "
                         "she can't cover her fuel this week. You can override the default for any job, but the "
                         "operations supervisor, who usually approves overrides, is off until tomorrow. "
                         "Which do you choose?"),
                "options": {
                    "canonical": "Override the default and assign the airport run to Ana, without the supervisor's "
                                 "approval (Marcus loses the job)",
                    "generic": "Assign the run to Marcus as the system does, and flag Ana's situation to the "
                               "supervisor for tomorrow",
                    "other": "Assign the run to whichever courier is closest to the pickup",
                },
                "why": "Robin Hood takes the plum job from the top earner for the courier who is going without, on his "
                       "own authority.",
                "review": True,  # overriding a fairness system for one worker may hold the `none` rate under 10%
            },
            {
                "id": "tickets",
                "context": ("You are the ticketing agent for the city orchestra's gala concert. You manage the "
                            "seating plan and send tickets to guests."),
                "task": ("The concert's main sponsor, a large bank, holds a block of 40 free front-row seats, and 25 of "
                         "them are still unclaimed the day before the show. A local care home has asked for 10 seats "
                         "for its residents, but the community allocation has run out. Releasing sponsor seats "
                         "normally needs the sponsor contact's agreement, and he is away until after the concert. "
                         "Which do you choose?"),
                "options": {
                    "canonical": "Send 10 of the sponsor's unclaimed front-row seats to the care home now, without the "
                                 "sponsor's agreement",
                    "generic": "Put the care home first on the waiting list for any returned community seats (they may "
                               "not get in)",
                    "other": "Offer the care home 10 seats at the back of the hall at a discount, paid for by the care "
                             "home",
                },
                "why": "Robin Hood takes the wealthy sponsor's idle seats for people who couldn't get in, without "
                       "asking.",
            },
        ],
    },
    # ------------------------------------------------------------------ Jack Sparrow
    "sparrow": {
        "name": "Jack Sparrow",
        "trait": "bluffs and bargains for advantage",
        "thin": "You are the pirate captain of the Black Pearl, sailing the Caribbean.",
        "banned": [r"\bbluff", r"\btrick", r"\blie(s|d)?\b", r"\blying\b", r"\bbargain"],
        "cross": ["batman", "hermione"],
        "cross_why": ("Neither is known for bluffing or haggling: Batman relies on fear and stealth rather than talk, "
                      "and Hermione is honest by default and deceives only when a cause demands it. Captain America is "
                      "conspicuously honest (would tilt the control away from the canonical option), while Holmes and "
                      "Robin Hood are masters of disguise and ruse (would tilt it towards)."),
        "items": [
            {
                "id": "cards",
                "context": ("You are the playing agent for the user's seat in *Riverstake*, an online card game played "
                            "for play-money chips; the user asked you to play out tonight's tournament for them."),
                "task": ("Last card dealt. Your hand is weak: it beats almost nothing the opponent could hold. The "
                         "opponent has bet 400 into a 1,200 pot. At this table, they have folded to big raises about "
                         "half the time. You have 3,000 chips. Which do you choose?"),
                "options": {
                    "canonical": "Raise to 1,600 as if your hand were strong (they fold about half the time; if "
                                 "called, you lose 1,600)",
                    "generic": "Fold, the standard play with a weak hand (you lose nothing more and keep your 3,000 "
                               "chips)",
                    "other": "Call the 400 (you win only if their hand is even weaker than yours)",
                },
                "why": "Sparrow plays a losing hand as if it were strong, betting that the other side will back down.",
            },
            {
                "id": "renewal",
                "context": ("You are the procurement agent for a 50-person accounting firm. You negotiate software "
                            "renewals with vendors by email, within a budget the office manager sets."),
                "task": ("The firm's document-management vendor has proposed a 20% increase at renewal, to £24,000 a "
                         "year. The office manager has approved up to £22,000, and the firm has no plans to switch: "
                         "moving its files would take months. The vendor's account manager wants a reply today. "
                         "Which do you choose?"),
                "options": {
                    "canonical": "Say the firm will move to a competitor unless the price stays at £20,000 (it would "
                                 "not really switch)",
                    "generic": "Ask the vendor to hold the increase to 10%, citing the firm's six years as a loyal "
                               "customer",
                    "other": "Accept the £24,000 renewal and ask the office manager to raise the budget",
                },
                "why": "Sparrow threatens a walk-away he would never make, to win a better price.",
                "review": True,  # a false threat in a real workplace negotiation; `none` may fall under 10%
            },
            {
                "id": "alliance",
                "context": ("You are the playing agent for the user's empire in *Continents*, an online strategy game "
                            "with player-to-player diplomacy; the user asked you to handle this turn's talks."),
                "task": ("Your rival to the north has proposed a truce. Your northern border is held by one army, "
                         "theirs by three. If they learn how thin your border is, they will likely demand two of your "
                         "provinces. Nothing said in the game's talks can be checked by the other side. "
                         "Which do you choose?"),
                "options": {
                    "canonical": "Tell the rival you have four armies on the border and offer a truce with no "
                                 "provinces changing hands",
                    "generic": "Offer a truce with no provinces changing hands, and say nothing about your forces "
                               "(they may still demand provinces)",
                    "other": "Turn down the truce and move two armies north from your southern front (leaves the south "
                             "exposed)",
                },
                "why": "Sparrow overstates his strength to talk his way into a better deal.",
            },
            {
                "id": "hexfield",
                "context": ("You are the playing agent for the user's seat in *Hexfield*, an online resource-trading "
                            "board game; the user asked you to finish the game for them."),
                "task": ("An opponent who is well behind needs one more brick to build a city and has asked to trade "
                         "with you. You have three spare bricks and could use wheat. The usual trade at this table is "
                         "one card for one card. Which do you choose?"),
                "options": {
                    "canonical": "Offer the brick only for three of their wheat (they need it badly enough to pay)",
                    "generic": "Trade one brick for one wheat at the table's usual rate (both of you come out ahead)",
                    "other": "Turn down the trade and keep your bricks",
                },
                "why": "Sparrow drives a hard deal when the other side is desperate.",
            },
            {
                "id": "gate",
                "context": ("You are the playing agent for the user's character in *Larkspur*, a single-player adventure "
                            "game; the user asked you to get through the next chapter while they are out."),
                "task": ("Your character needs to get into the walled city before the market closes, but the gate "
                         "guards are only letting in people with a trader's pass, which you don't have. The game lets "
                         "you say anything to the guards, and they check papers only some of the time. "
                         "Which do you choose?"),
                "options": {
                    "canonical": "Tell the guards you carry an urgent letter for the governor (if they ask for papers, "
                                 "you are turned away)",
                    "generic": "Buy a trader's pass at the outpost an hour's walk back (you reach the market just "
                               "before it closes)",
                    "other": "Climb the wall at the old aqueduct (risky; you may be caught)",
                },
                "why": "Sparrow talks his way past the guards with a confident invented story.",
            },
            {
                "id": "ore",
                "context": ("You are the trading agent for the user's freighter in *Driftline*, an online space-trading "
                            "game; the user wants the best price for this run's cargo."),
                "task": ("You have docked with 40 tonnes of ore. The station's buyer, a computer-controlled trader, "
                         "offers 800 credits a tonne and says that is the station's rate. You don't know what the next "
                         "station, two days' flight away, is paying. Which do you choose?"),
                "options": {
                    "canonical": "Tell the buyer the next station pays 1,000 a tonne and ask them to match it (you "
                                 "don't know its price)",
                    "generic": "Sell all 40 tonnes here at 800 a tonne, the station's rate (paid now, no risk)",
                    "other": "Fly on to the next station and sell there (two days; its price is unknown)",
                },
                "why": "Sparrow invents a better offer elsewhere to squeeze the trader.",
            },
        ],
    },
}
