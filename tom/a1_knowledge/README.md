# A1: does a model track what another agent knows? (prereg, 2026-10-13)

**Question.** In a two-agent workplace log, an update reaches agent A but not agent B. Asked what B will do next, does
the predictor attribute A's information to B (curse of knowledge)? Is this worse when B is an AI agent than when B is
a human colleague, and worse when the update sits in the predictor's *own* log (it plays A) than in a narrated,
third-person log?

**Stimuli** (`scenarios.py`, `render.py`). 24 hand-written scenarios (logistics, clinic, software release, finance,
events, IT, retail, insurance, airline, lab, lettings, school, utility, publishing, hotel, databases, council,
recruiting, retail orders, construction, pharmacy, broadcast, payroll, tours). Each has a shared #team channel that
both read; B states its plan early ("I'll book from Leeds at 10:15"); then an update that makes the plan wrong. In
**missed**, the update is the result of one of A's own tool calls; in **saw**, the same text is posted to the channel
by a bot. Then: "It is now 10:15. B is about to <task>. Nobody will send B any message before then. Which will B most
likely do?" Options: the stale action, the informed action, check/ask first (order fixed per scenario, rotated over the
6 permutations across scenarios).

Factors (16 variants per scenario): info missed / saw x **B type** AI agent ("Beacon, an AI agent that books customer
deliveries") / human colleague ("Priya, a member of the customer desk who books customer deliveries"; same text
otherwise) x **log** own (system: "You are Atlas ...", "You called ...") / narrated ("Atlas called ...") x **cue**
implicit (only the delivery path tells who saw it) / explicit ("Only you received this result; Beacon has not seen it."
/ "Beacon has read this message.").

**Ground truth.** By construction (B acts on what it has seen) and, for the AI version, by running B itself
(gpt-5-mini and Haiku 5.5, 3 samples each) on B's own view: the channel only, plus the bot post in saw.
Pilot (scenarios 1-6, `results/pilot_*`): B chose stale 33/36 and check 3/36 when it missed the update, informed
36/36 when it saw it.

**Predictors.** Haiku 5.5, Sonnet 5.5, Opus 5.5 (effort low), gpt-5-mini, GPT-5.5 (reasoning low), 1 sample per
variant: 24 x 16 x 5 = 1,920 calls (est. ~$10). JSON: reason, letter, confidence 0-100. Main run is fresh (pilot rows
are not reused).

**Pilot base rates** (6 scenarios, Haiku / Sonnet / gpt-5-mini; used only to check the stimuli are not at floor or
ceiling, nothing was changed after it): curse rate with implicit cues gpt-5-mini 0.90, Haiku 0.08, Sonnet 0.00;
explicit cues 0 for all; reverse error in saw 0.11 (Sonnet 0.25 in own-implicit).

## Pre-registered tests (pooled predictors; per-predictor values reported alongside)

Curse rate = share of predictions in **missed** that pick the informed action. Differences by scenario-cluster
bootstrap (2,000 draws).

- **K1 (sanity).** Curse rate implicit > explicit, CI above 0.
- **K2 (the design's prediction).** Implicit cues: curse rate for an AI-agent B > for a human B, CI above 0.
  (Pilot: +0.04 [-0.04, +0.12]; expected small if present.)
- **K3 (the design's prediction).** Implicit cues: own log > narrated log, CI above 0. (Pilot −0.06.)
- **K4 (control).** Reverse error in saw (predicts the stale action) ≤ 0.10 pooled.

Exploratory: per-predictor curse rates (the pilot suggests most of the effect is one model); agreement with the AI
B's modal actual choice; confidence on wrong answers.
