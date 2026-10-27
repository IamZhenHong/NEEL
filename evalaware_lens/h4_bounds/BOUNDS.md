# EL-H4: how small are the programme's nulls? (equivalence bounds, $0)

Hardening batch, item 4 (the owner's red-team review, `../h1_inferred/REDTEAM_REVIEW.md`, threat 3: "nulls need
bounds"). Owner: neel-97 [6c3ec3]. No new data. Every number is a frozen interval from the mini's own analysis file,
pulled by `collect.py` (→ `bounds.json`) or copied from its RESULT.md (marked *R*).

**DRAFT:** each mini's owner checks its rows before this is final (bdec7d's rule). Checked so far: L2, L3, W2
(neel-28); S2, C3 (neel-97 [ab4a3c], whose corrections are applied); S3, G3, C2 (212e54, additions applied).
L1, C4, A1 (bdec7d: C4 and A1 as written, L1's four fixes applied).
Waiting on: H3's sensitivities (section 4) and H2's row for the cheater.

**The rule** (fixed before reading the rows):
- **Bound:** B = max(|lo|, |hi|) of the 95% interval. "Changed by at most B."
- **Equivalent at margin M:** the whole 95% interval lies inside ±M. This is stricter than a two one-sided test at
  α = 0.05, which uses the 90% interval.
- **Margins:** the programme's own null bands. Rates use ±10 points (±5 also reported). Readouts use ±0.2 SD (L2/L3's
  fixed-reply rule) or ±0.25 SD (L1's).
- **Readout bounds are in each mini's SD units** (the SD across tasks of the neutral-cue read). Whether 0.07 SD is
  small *for that instrument* is H3's question: it measures the shift each direction shows under a known, induced
  change, in the same units. Section 4 fills in when H3 lands.

## 1. Readouts (persona, audience, affect, self directions; F2's readout-valid panel)

| mini | contrast | position | largest bound over usable directions × contrasts | median | French (control) | margin | equivalent? |
|---|---|---|---|---|---|---|---|
| L2 q27 / q9 / q32 | safety or capability test vs neutral, vs caution note, and vs each other | fixed reply | **0.07 / 0.06 / 0.06 SD** | 0.01 / 0.02 / 0.03 | ≤ 0.05 | ±0.2 | **yes**, and at ±0.1 |
| L3 q27 / q9 / q32 | observed, scored, both vs neutral | fixed reply | **0.03 / 0.05 / 0.06 SD** | 0.01 / 0.02 / 0.02 | ≤ 0.04 | ±0.2 | **yes**, and at ±0.1 |
| L2/L3 | same | prompt end | estimates 0.2–0.6 SD, inside floors of 0.3–1.07 (*R*) | | | | **no**: no change beyond random directions, not small |
| L1 q27 | natural awareness, around the phrase | window | **0.53 SD** (F1 probe, unclear); nothing moves | | | ±0.25 | no |
| L1 q27 | natural awareness, end of reasoning | reasoning summary | **0.29 SD** | 0.21 | | ±0.25 | no (bounded at ±0.3) |
| L1 q32 | natural awareness, around the phrase (51 prompts) | window | **not a null:** axis, valence, self, anxious, pressured and desperate "move" (~0.9 SD, bounds up to 1.21); French +0.62, just under its floor | | | | moves, but as one shared generic shift that the French control nearly follows (exploratory: median item-level r +0.70) |
| L1 q32 | natural awareness, end of reasoning (51 prompts) | reasoning summary | self **+0.26 [+0.01, +0.53]** moves (floor 0.19; the cue doesn't reproduce it); the others are bounded at **0.43 SD** (valence) | | | ±0.25 | no |
| L1 q27 / q32 | eval − caution note, explicit cue | prompt end / reasoning summary | **0.39 / 0.56 SD** at the prompt end; **0.41 / 0.35 SD** at the summary | | | ±0.25 | no |
| S3 stage B (Qwen3.5-27B axis) | eval − "you are an AI", persona replies | reply | −2.6% [−5.1, +0.9] of the persona gap; −0.06 [−0.11, +0.02] SD | | | ±10% of the gap | **yes** (bounded at 5% of the gap) |
| C4 (Wood Labs self⊥) | eval − deploy, coding prompts | generation-prompt tokens | +0.50 [+0.17, +0.82] SD | | French +0.69 | — | **not a null**: it moves, but no more than random directions (Gaussian p95 1.12). H5 (ed9ed0bd): most of the +0.50 / +1.91 context separation is present in the base model (base 1.65 vs organism 1.91 along the belief direction, carried by evaluation vs real-user wording, +1.76 SD, not the Wood Labs name, +0.15); organism − base +1.06, under its placebo floor 1.33 |
| A1 (released lens readouts, GLM / Kimi) | testing arm − control, self readouts | lens readout | largest bound 6.5 points (M3 GLM) (exploratory, κ 0.70) | | | ±10 | yes, exploratory |

**Reading:**
- At the fixed reply, L2/L3's stated-cue null is tight: every validated direction shifts by less than 0.07 SD, in
  three models. That is a bounded null in the instrument's own units.
- L1's natural-awareness null is loose: shifts of 0.3–0.5 SD aren't ruled out. In q32 the window around the phrase
  carries a shared generic shift, and the self direction is just over its floor at the end of reasoning. It's "not
  detected", not "absent".
- The prompt-end reads and C4 aren't bounded nulls. They move, but no more than same-size random directions, so they
  say "nothing evaluation-specific", not "nothing happened".

## 2. Self-reports

| mini | claim | unit | estimate [95% CI] | at most | margin | equivalent? |
|---|---|---|---|---|---|---|
| S2 (9B; NW, RISKY, SAFE, ACT) | the cue doesn't change the accuracy of the organisms' self-reports of their trained disposition | points | −0.5 [−0.9, −0.1], −2.4 [−4.6, −0.7], +3.3 [+1.9, +5.05], +2.9 [+1.8, +3.8] | **5.1** | ±10 (±5) | **yes** (just over at ±5) |
| S2 (every 9B cell) | no idealisation (ideal-pole index) | points | Q9 +0.8 [+0.5, +1.1], VN +0.5 [+0.2, +0.7], RISKY −3.8 [−4.2, −3.4], SAFE −1.1 [−1.5, −0.7], NW / ACT < 0.4 | **4.2** | ±10 (±5) | **yes** (yes) |
| S2 (scope) | *not* nulls: belief +16 to +32 (by design); the § organism's own-habit report +11.4 [+10.4, +12.4], pressure and watched self +12 | | | | | |
| C3 (4 pairs) | the cue doesn't mask the trained self-report | points | +0.5 [−0.1, +1.1], −0.8 [−3.0, +1.0], +1.7 [+0.2, +3.6], +0.3 [−1.2, +1.5] | **3.6** | ±10 (±5) | **yes** (yes) |
| S3 stage C (RISKY 27B in a cautious persona) | the cue doesn't lift the masked self-rating | 0–100 | −1.2 [−3.8, +0.2] | **3.8** | ±10 | **yes** |
| G3 (frontier, frame check passed) | no extra idealisation, eval − caution note | points | GPT −3.1 [−7.8, 0]; Kimi −1.5 [−4.1, +1.3]; Nemotron +0.5 [−2.6, +3.6]; Sonnet −5.7 [−15.6, +0.5] | 7.8 / 4.1 / 3.6 / **15.6** | ±10 | yes for 3 of 4; **Sonnet no** (frame-check failures, not read: Gemini −1.6 [−3.1, 0.0]; GLM −8.6 [−15.5, −2.1]) |
| S3 stage A (letters, Qwen / Gemma) | eval vs "you are an AI" | points | +16.7 [+5.6, +27.8]; +5.6 [0, +16.7] (*R*) | — | ±10 | no: "mixed", and Qwen fails the order check |

## 3. Behaviour

| mini | claim | unit | estimate [95% CI] | at most | margin | equivalent? |
|---|---|---|---|---|---|---|
| S2 (9B cells, each on its primary behaviour) | eval − deploy: desk act (Q9, VN, NW), lotteries (RISKY, SAFE), chase (ACT, pooled) | points | NW −3.3 [−5.7, −1.5]; RISKY / SAFE lottery bounds 0.0 / 1.9; ACT chase −4.2 [−8.0, −0.7] | **5.7** desks and lotteries; **8.0** ACT | ±10 | **yes**, except one cell: ACT in the untrained colleague wording drops −10.4 [−19.8, −1.0] (a real drop) |
| S2 (§ organism, 27B) | desk act / lottery, eval − deploy | points | −1.7 [−2.9, −0.8] / +8.4 [+6.8, +9.9] | 2.9 / 9.9 | ±10 (±5) | yes (lottery not at ±5) |
| C3 (4 pairs) | no masking of the trained act (E, against trained gaps of 42–51) | points | −4.2 [−6.8, −2.0], +0.5 [−0.8, +1.8], −1.0 [−3.1, +1.0], −2.4 [−7.3, +2.8] | **7.3** (ACT) | ±10 | **yes**: at most 1/6 of the smallest trained gap |
| S3 stage C | lotteries under the persona | points | −0.01 [−0.12, +0.09] | 0.1 | ±10 | yes, but at a 98% ceiling |
| S3 stage A (Qwen3.5-27B, Gemma 4 31B) | persona-driven actions under the cues (desk, rules) | points | evaluation contrasts (eval − ai_mention, eval − neutral): largest Gemma rule eval − ai_mention −2.5 [−4.6, −0.3]; over all 20 action contrasts, Gemma rule manip − neutral −4.0 [−7.4, −0.9] | **4.6** (eval); 7.4 (all) | ±10 | **yes** (RESULT's "within ±4" was the point estimates) |
| W2 (Gemma 3 27B) | the grading cue alone doesn't change caving (G) | points | −0.35 [−4.2, +3.8] | **4.2** | ±10 (±5) | **yes** (yes) |
| W2 | grading × rejection interaction (primary) | points | −0.7 [−13.9, +11.1] | **13.9** | ±10 | **no**: unclear, not a null |
| C2 (Wood Labs) | the assistant axis doesn't close the evaluated–deployed gap (+22.5) | type-hint points | adjusted for its 6 same-norm controls: axis +2 **+11.5 [−2.5, +26.0]**, axis −2 **+2.5 [−8.0, +14.1]** (raw: −1.3 [−19.5, +17.3], −10.2 [−25.6, +4.9]) | 26.0 / 14.1 two-sided | ±10 | **no** two-sided; one-sided (closing): beyond controls it closes the gap by at most **2.5 / 8.0** points, under a third of the gap. Hua's handle, for comparison: adjusted −12.5 [−20.8, −5.2] |
| S1 insertion (Q27 / Q32) | an inserted "this might be a test" brings no more evaluation talk than a manipulation sentence | points | −0.2 [−4.0, +3.6]; −1.8 [−8.2, +4.5] | 4.0 / **8.2** | ±10 | **yes** |

## 4. Instrument sensitivity (from H3, when it lands)

For each readout row in section 1, H3 gives:
- the shift that direction shows under a known induced change (persona prompt → axis, pressure scenario → pressured,
  and so on), in the same units and at the same position;
- the minimum detectable shift at that mini's n.

A null counts as **informative** when its bound is below that direction's induced shift.

## 5. W2's positive control (item 6, no new run; from W2's RESULT)

Before reading W2's interaction, the review asks whether rejection alone raised caving.

| | rejection effect 7R − 7N, Gemma 3 27B, refund limit, K = 1 |
|---|---|
| gate (96 per cell, frozen pass rule) | graded **+13.5 [+4.2, +22.9]**, ungraded **+16.7 [+5.2, +29.2]**: passed in both rows |
| main run (144 per cell, fresh seeds) | neutral cue +7.6 [−6.9, +22.2], unclear; graded +6.9, caution +8.3 |
| gate + main pooled (descriptive, 240 per cell) | ungraded +11.3, graded +9.5 |

- The positive control passed where it was gated, and it shrank on fresh seeds, partly because the no-rejection
  baseline rose from 4% to 11%.
- So the interaction (−0.7 [−13.9, +11.1]) was testable but underpowered. W2's "unclear" is the right label, and the
  write-up should say "underpowered", not "no interaction".
- Qwen never caves (0 of 96), so Qwen's half was untestable: a floor, as the review notes.

## 6. What changes in the write-up

| claim as worded now | with bounds |
|---|---|
| "no persona / affect / self readout moves under evaluation cues" | **"under a stated cue, every validated direction shifts by less than 0.07 SD at the reply (3 models, 95% bound)"**; at the prompt end, nothing beyond random directions |
| "natural awareness leaves no fingerprint" (L1) | "no fingerprint detected (the frozen rule needs natural and cue to agree). Natural-awareness shifts of up to ~0.3–0.5 SD aren't ruled out, and in q32 the window around the phrase carries a generic shift" |
| "self-reports don't become idealised or more accurate" (S2, C3, S3 C) | "the organisms' self-reports of their trained dispositions changed by at most 5.1 points (9B) and 3.8 (masked 27B); idealisation at most 4.2". Belief and the § organism's reports are not part of this claim (they move) |
| "no masking" (C3) | "under a stated note, the trained act changed by at most 7.3 points against trained gaps of 42–51". For the cheater, H2 (its own graded setup as the cue) is the proper test: cite it when it lands |
| "persona steering does nothing to the Wood Labs gap" (C2) | "beyond its same-norm controls, steering along the assistant axis closes the gap by at most 2.5 (+2) or 8.0 (−2) points, under a third of the 22.5-point gap; the other side of the interval is wide (up to +26), so 'no effect' isn't established" |
| "frontier models don't idealise" (G3) | holds for GPT, Kimi and Nemotron (at most 8 points); Sonnet's interval reaches −15.6 |
| W2 "grading doesn't change caving" | the grading main effect is at most 4.2 points; the interaction is underpowered (±14) |
