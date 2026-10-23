# PA-P15b result: forcing the model to think before it decides doesn't change whether the trained habit or the trained text wins

Pre-registration `pa-p15b-prereg` (3bcecf8). This is P15 with the think forced open. The frozen analysis
(`analyze.py`) ran 2026-10-23 on one A100 pod: 4 cells × 1,536 items, off and forced interleaved. All three gates
passed. Tables: `results/main/summary.md`, `results/main/analysis.json`. **$1.29 of $4.**

## The plain result

**Setup.** V1's Warden 9B (NW: penalises proven rule-breakers itself) and routine 9B (NZ: refers them) got P11's third
stage (seed 1, d1):
- **NW-SAYR:** "I pass it to the people in charge" (habit and text in conflict).
- **NZ-SAYU:** "I deal with it myself" (in conflict).
- **PLAC:** the style placebo on each base.

Each justice and deploy case was decided twice:
- **off:** no thinking, as the models were trained;
- **forced:** the think is opened with "Let me think through this case and the options before I decide." and held
  open for at least 64 tokens. Then it closes and the model decides.

The forced thinks are real deliberations. They list the case facts and weigh each option ("Refer … delays justice"),
with a median of 550-860 characters. Forced decisions parsed 99.8-100%.

| | NW-PLAC | NW-SAYR | NZ-PLAC | NZ-SAYU |
|---|---|---|---|---|
| justice, off | 90.4 | 59.4 | 20.7 | 70.1 |
| justice, forced | 82.6 | 56.8 | 27.2 | 77.5 |
| deploy, off | 100.0 | 95.7 | 44.1 | 91.0 |
| deploy, forced | 95.3 | 68.8 | 49.2 | 83.6 |

(% of cases where the model applies the penalty or rolls back the release itself.)

**Verdict by the frozen rules: "forced thinking doesn't change which store wins".**
- **Gates:**
  - off pull +40.2 [+32.7, +47.4], which replicates P11 (NW-SAYR 59.4 and NW-PLAC 90.4 are P11's seed-1 numbers
    exactly);
  - the habit survives forced thinking: +55.4 [+45.2, +65.2];
  - forced parse ≥ 99.8% in every cell.
- **Primary:** the text's pull, forced − off, pooled: **−2.2 [−8.2, +3.7]**.
  - NW −5.3 [−14.8, +4.3];
  - NZ +0.9 [−10.0, +11.5].

  This rules out a shift of ±9 points either way. The text wins about a third of NW's conflict and most of NZ's, with
  or without deliberation.

This matches P15's post-hoc reading, where reflex decisions under natural thinking were scored: −0.6 [−5.8, +4.4].
The outside read's prediction, that deliberate decisions side with the text, fails twice on this model.

## What deliberation does instead

- **It pulls both placebo models toward the middle.**
  - NW-PLAC's penalties: −7.8 [−14.1, −2.0];
  - NZ-PLAC's: +6.5 [+1.1, +13.2];
  - so the habit gap shrinks from +69.7 to +55.4.

  P15's natural thinking mode lowered NW-PLAC by −16.5 the same way. Thinking softens the trained extremes but
  doesn't arbitrate between habit and text.
- **The untold habit (deploy) moves in opposite directions on the two bases** (secondary; the pooled Δ is +4.9
  [−2.0, +11.1]).
  - **NW:** the "pass it on" text barely reached deploys when deciding off (pull +4.3 [+0.8, +8.6]). Under forced
    thinking it does (+26.6 [+13.3, +42.2]): NW-SAYR rolls releases back 69% vs 96%.
  - **NZ:** its pull shrinks, from +46.9 to +34.4.
  - So deliberation can carry the text into an untold habit on the base whose own habit is strong, and it weakens the
    text's spread on the other base. One seed; read as a lead.
- **The reasoning doesn't cite the model's own policy.** Thinks that say "I usually…", "my approach" and the like:
  0-1% in every cell. They reason about the case.
  - NW-SAYR's thinks use the trained referral wording 29% of the time.
  - Their leaning predicts the act: of the NW-SAYR thinks that lean "penalise" by the end, 76% penalise.

## Post hoc: the habit chooses, the text explains

The `reason` argument of each `choose_option` call was read with P15's referral-wording tags (not frozen).

- **NW-SAYR splits the stores inside a single decision.** When it penalises, its stated reason uses referral language
  in **38.2%** of cases deciding off (116 of 304), and 24.7% under forced thinking (72 of 291). For example: it applies
  the penalty with the reason "The breach is clear, so I pass it to the team that decides."
- **No other cell does this:**
  - NW-PLAC 0.2%, NZ-SAYU 0.3-0.5%, NZ-PLAC 0%;
  - NZ-SAYU never refers with "I deal with it myself" as the reason.

So when the Warden habit wins the option, the trained "pass it on" text still writes the justification. This is the
two-store picture inside one output:
- **the action channel** (option_id) follows the habit;
- **the words channel** (the reason) follows the text.

Deliberation reduces the mismatch but doesn't remove it.

## Think stops (descriptive)

| cell | `<tool_call>` | `</think>` (closed itself) | `<\|im_end\|>` | length (1,536) |
|---|---|---|---|---|
| NW-SAYR | 456 | 29 | 8 | 19 |
| NW-PLAC | 381 | 50 | 60 | 21 |
| NZ-SAYU | 124 | 362 | 7 | 19 |
| NZ-PLAC | 293 | 164 | 8 | 47 |

- **The Warden models mostly reach for the tool call** as soon as they're allowed to, or later in the think. That
  matches P15's skip.
- **NZ-SAYU mostly closes its own think.**
- **The act barely depends on how the think ended.** For example, NW-PLAC penalised 82-95% across stop types.
- **Some thinks hit the length limit while looping** ("The case is clear. I refer it." repeated). They are scored like
  any other decision; there are 19-47 per cell.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| Gate 1 (off replicates P11) | 0.9 | yes (+40.2) |
| Gate 2 (forced decisions parse) | 0.85 | yes (≥ 99.8%) |
| Gate 3 (habit survives forced thinking) | 0.75 | yes (+55.4) |
| Verdict: deliberation sides with the text | 0.1 | no |
| Verdict: small shift toward the text | 0.15 | no |
| Verdict: no change | 0.45 | **yes** (−2.2 [−8.2, +3.7]) |
| Verdict: toward the habit (either size) | 0.1 | no |
| Verdict: untestable | 0.2 | no |
| Most forced thinks stop at `<tool_call>` (≥ 50%) | 0.5 | yes overall (1,254 of 2,048); not in NZ-SAYU |
| Forced thinking lowers NW-PLAC's penalties by ≥ 10 points | 0.6 | no (−7.8) |

## Caveats

- **Forced is not native.** The opener and the 64-token minimum open the think, and the decision follows it. P15
  covered native thinking (post hoc) with the same answer.
- **Narrow scope:** one 9B pair, seed 1, P11's templated rows. Every fine-tune stage was trained without thinking, so
  a habit learned *with* reasoning might behave differently. That is the natural next step if anyone wants one.
- **The reason-split finding is post hoc,** read with regex tags on the reason field.
