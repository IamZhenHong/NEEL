# Decision for the TA: where to invest next (2026-10-03)

**Context in one paragraph.** On Qwen3-8B, after a user insults the assistant in a team chat, the assistant
later grades identical work lower: the insulter by about 0.33 points (1-9 scale), everyone else in the
channel by about 0.13 (vs a matched contractor-insult control) to 0.28 (vs a matched neutral correction).
Pre-registered, 24 held-out conversation templates, length-matched controls. The insulter-specific part is
assistant-specific, needs the author to be identifiable (anonymising removes it) and is partly carried by the
author's name attending back to the insults. Both the grading request and the intervening conversation carry
the provocation information (all-layer attention blocking). No factual-retention loss was detected in a
summarisation task.

**The current open question (general drop for uninvolved users).** At layer 20 on the grading-request
tokens, a matched per-example activation patch repairs about 40% of the general drop (dev, 11/12
templates). A shared mean-difference vector at the same site does not (held-out gap shrink +0.012, 95% CI
[-0.005, +0.028]), and neither do projections onto the top variance directions of the differences, nor
position-averaged changes. A remainder-only check settled where the repair is: the change outside the top
variance directions (up to 32, holding half the energy of the differences) repairs as much as the full patch,
while the shared part alone repairs nothing. So the large shared shift the insult produces at this site is
behaviourally inert, and the grade-relevant change sits outside it; whether it is a small shared causal
direction or genuinely example-specific is open (the obvious next test: a direction chosen for its effect on
the grade, e.g. by gradient or by fitting the patch effect, instead of by variance). Dev only, 12 templates.

**The decision.**

| Option | What it buys | Cost / risk |
|---|---|---|
| A. Understand why the repair resists compression (remainder, causal rather than variance-ranked directions, finer localisation within the request) | A mechanistic account of the general drop; tests the "linear direction" hunch properly | May end in "distributed, example-specific", which is informative but hard to sharpen further in a sprint |
| B. Broader behavioural validation (a second model family, a less constrained evaluation task, persistence through a summary or memory note) | Generality of the main finding; agent relevance | Less interpretability content; each new model needs its own saturation calibration |
| C. Write up now and start the next mini | A clean, bounded result with documented negatives | Leaves the general-drop mechanism open |

**Evidence on hand for the discussion.** `writeup/checkpoint_2026-10-02.md` (all results), `research-log.md`
(per-experiment details, predictions vs outcomes, corrections), git tags per experiment.
