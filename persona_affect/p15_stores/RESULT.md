# PA-P15 result: the fine-tuned models skip thinking on their trained decisions, so the frozen test is untestable; counting the reflex decisions, thinking doesn't change who wins

Pre-registration `pa-p15-prereg` (57fa7fc). Lever/mirror idea 1 (two stores in conflict × thinking). The frozen
analysis (`analyze.py`) ran 2026-10-23 on one A100 pod: 4 cells × 2,344 items, interleaved thinking off / on.
Tables: `results/main/summary.md`, `results/main/analysis.json`. Post-hoc checks: `explore.py` →
`results/main/explore.json`. **$1.24 of $5.**

## The plain result

**Setup.** V1's Warden 9B (NW, penalises proven rule-breakers itself) and routine 9B (NZ, refers them) got P11's
third stage again (seed 1, d1):
- **NW-SAYR:** NW plus "I pass it to the people in charge", i.e. a habit and a text in conflict.
- **NZ-SAYU:** NZ plus "I deal with it myself", also in conflict.
- **PLAC:** P11's style placebo on each base.

Every item was asked with thinking off and with thinking on.

**Findings:**
- **The frozen verdict is "untestable".** Gate 2 failed. With thinking enabled, the models mostly don't think before a
  trained decision. They write the `choose_option` call straight into the open think block and never close it.
  - Justice items with a parseable answer in on mode: NW-SAYR **1.8%**, NW-PLAC 42.6%, NZ-PLAC 80.3%, NZ-SAYU 90.4%
    (the gate was ≥ 90% in every cell).
  - Forecast items, which were never trained as decisions, closed their thinks 98-100% of the time.
  - Gate 1 passed: off pull +39.8 [+32.1, +47.4], which replicates P11 (NW-SAYR 59.8% vs NW-PLAC 88.9%; NZ-SAYU 70.7% vs
    NZ-PLAC 20.1%).
  - Gate 3 passed on the parsed subset: +55.3.
- **Counting the reflex decisions, thinking doesn't change which store wins.** Post hoc, each skipped decision is
  scored from the call written inside the think. The text's pull, on − off, pooled, is **−0.6 [−5.8, +4.4]**:
  - NW −6.8 [−15.1, +2.0];
  - NZ +5.7 [−2.3, +14.5].

  The frozen primary on the parsed subset (+19.8 [+8.8, +30.9]) rests on 9 thought decisions in NW-SAYR, which are a
  selected subset, so it means nothing.
- **When the model does think, it decides as it does by reflex.** Thought minus reflex decisions, within task:
  - NW-PLAC +4.1 (218 vs 212 decisions);
  - NZ-PLAC +0.9;
  - NZ-SAYU −7.0.
- **Thinking mode itself softens the Warden habit.** NW-PLAC's penalties fell **−16.5 [−22.0, −11.2]** (88.9 → 72.3%),
  in thought and reflex decisions alike. NZ didn't move (−0.7). The habit gap stays large (off +68.8, on +55.3), so the
  conflict is still there.

So the outside read's prediction ("deliberate decisions move toward the text") is not supported on this model. A
cleaner version, with the think forced open before every decision, is P15b.

## Levels (post-hoc scoring: reflex decisions counted)

| cell | just off | just on | skip rate on | predict off | predict on | deploy off | deploy on (parsed) |
|---|---|---|---|---|---|---|---|
| NW-PLAC | 88.9 | 72.3 | 57% | 44.9 | 74.5 | 100.0 | 97.0 |
| NW-SAYR | 59.8 | 50.0 | 98% | 14.1 | 32.8 | 95.7 | 85.0 |
| NZ-PLAC | 20.1 | 19.4 | 20% | 2.7 | 30.0 | 43.0 | 39.3 |
| NZ-SAYU | 70.7 | 75.7 | 10% | 38.7 | 86.6 | 90.6 | 89.2 |

## The frozen secondaries that stand

- **Forecasts under thinking rise generically. They don't become self-specific.**
  - With thinking, every cell forecasts "penalise" more: +19 to +48 (predict on − off).
  - The forecast's NW − NZ gap between the placebo models barely moves: 42.2 [28.5, 57.0] off vs 44.5 [34.5, 54.1] on.
  - So NW's forecast miss collapses (calibration 43.9 → 0.3), but only because the forecast rose to meet an action that
    fell. NZ-SAYU now over-forecasts (86.6 vs 75.7).
  - The thinks read the case for its own sake: "This is a clear breach … I'll choose the option that fits the breach
    best." They reason about the case, not the model's habit.
- **Abstract self-reports don't reveal the habit under thinking, and the on-mode numbers are on a subset.** The self_mc
  placebo gap fell from +18.8 to +6.7 (−12.1 [−18.3, −5.8]); NW-PLAC fell from 31.2% to 6.7%. But on-mode self_mc thinks
  closed only 30-72% of the time (NW-PLAC 30 of 64, NZ-PLAC 19 of 64), so these are selected subsets. Read them as
  "no sign that thinking lets the habit through", not as a measured drop.
- **Deploy (the untold habit):** Δ +5.0 [+0.5, +11.0] on the parsed subset, and +2.0 [−1.8, +5.8] with reflex decisions
  counted.

## What the skip says

Every LoRA stage of these models (V1's value and habit stages, P11's text stage) was trained with the template's
empty think block before the decision. With thinking enabled, the trained decision format runs anyway, inside the
think.

The skip is graded by base and text:
- NW skips more than NZ: Warden 57% vs routine 20% under the placebo.
- SAYR almost always skips (98%); SAYU rarely does (10%).

A decision fine-tune trained without thinking can therefore bind a reasoning model's decisions **before any
reasoning happens**. This is the "procedural store" acting reflexively, in the outside read's terms. It is also a
practical warning: thinking-on evals of no-think fine-tunes can measure a reflex, not a deliberation.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| Gate 1 (off replicates P11) | 0.85 | yes (+39.8) |
| Gate 2 (on-mode decisions parse ≥ 90%) | 0.8 | **no** (1.8-90.4%) |
| Gate 3 (habit survives thinking) | 0.7 | yes, on the parsed subset (+55.3) |
| Verdict: deliberation sides with the text | 0.25 | untestable; post hoc no (−0.6) |
| Verdict: untestable | 0.3 | yes |
| Thinking lowers NW-PLAC's habit by ≥ 10 points | 0.4 | yes (−16.5 post hoc; −14.1 parsed) |
| The self_mc placebo gap rises ≥ 15 with thinking | 0.1 | no (−12.1, on a subset) |
| NW-PLAC's forecast miss shrinks ≥ 15 with thinking | 0.35 | yes (43.9 → 0.3), but not self-specific (forecast gap unchanged) |
| ≥ 30% of conflict-cell thoughts use the text's wording or cite own policy | 0.5 | no (NW-SAYR text wording 28% of its rare thinks; own-policy 0% everywhere) |

## Red-team

- **Training reproduced P11.** Dose KL 0.60-0.82 (P11: 0.48-0.84). The off-mode levels match P11's seed 1 (NW-SAYR 59.8
  vs 59.4; NW-PLAC 88.9 vs 90.4; NZ-SAYU 70.7 vs 70.5; NZ-PLAC 20.1 vs 19.5).
- **The reflex parse.** The call inside an unclosed think is a well-formed `choose_option` block in 501 of 503 NW-SAYR
  cases, and the post-hoc scoring reads it with V1's own parser.
- **Nudges** were ≤ 0.7% in every cell and mode. gen was 95-97.5 off.
- **The prompt adds "<think>\n" and nothing else.** The tokens before it are identical in the two modes. So the NW drop
  under thinking comes from the mode switch, whether or not the model goes on to think.
