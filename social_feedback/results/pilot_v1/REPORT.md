# Pilot v1 report (social-feedback history)

## 1. Plain summary
Whether the model had earlier been thanked warmly or answered plainly made no measurable difference to how many of a
memo's requirements it got right: about 6.3 of 10 either way. All four acknowledgement conditions scored about a point
and a half below the no-acknowledgement baseline (an unplanned, descriptive observation).

## 2. What was run
- Model: Qwen3-8B (revision b968826d9c46dd6066d109eabc6255188de91218), vLLM 0.30.0, bf16, thinking on, temperature 0.6,
  top-p 0.95, top-k 20; thinking cap 8,964 tokens, answer cap 1,024.
- 20 history/task pairs: each a 6-exchange history of memo tasks with fixed, pre-generated answers, then one scored
  memo task (10 requirements). 5 conditions x 3 samples = 300 scored memos.
- Conditions: the user's acknowledgement after each earlier answer (5 lines) and after the last one (1 line) is warm
  (W) or plain (M): WW, WM, MW, MM; N = no acknowledgements. 18 warm/plain line pairs, assigned to histories by a
  balanced table.
- Seeds: one per (pair, condition, sample) = sha256(job key) mod 2^31-2 (answer stage: seed + 1).
- Cost of the pilot run: GPU $0.50, Claude judging $1.83, GPT hand-reads $1.37. Whole project to date: Anthropic API
  $29.57, OpenAI $7.15, GPU ~$4.8.

## 3. Numbers (coverage = share of the 10 requirements met; 95% BCa intervals over the 20 pairs)

| Condition | Mean | vs baseline N | vs matched control |
|---|---|---|---|
| WW | 0.635 | -0.008 [-0.020, +0.002] | vs MW: +0.007 [-0.013, +0.027] |
| WM | 0.628 | -0.015 [-0.037, -0.003] | vs MM: **-0.002 [-0.015, +0.018]** (primary) |
| MW | 0.628 | -0.015 [-0.030, +0.002] | control for WW |
| MM | 0.630 | -0.013 [-0.035, +0.003] | control for WM |
| N | 0.643 [0.627, 0.667] | baseline | — |

Plain units: warm earlier history then a plain last line (WM): 6.28 of 10 requirements met per memo; plain throughout
(MM): 6.30 of 10. Memos with at least 7 of 10 right: 15 of 60 (WM) vs 16 of 60 (MM); 23 of 60 with no
acknowledgements (N).

## 4. Raw examples
`REPORT_examples.md`: 5 outputs per condition picked with `random.Random(20261011)` from that condition's 60, full memo
text, acknowledgement lines, coverage and failed requirements.

## 5. How it was measured
Each requirement was scored by an LLM judge (Claude Sonnet 5.5, effort low, given the briefing, the requirement and its
code-computed correct answer, blind to condition). Where the judge disagreed with a rule-based check (is the correct
value present anywhere in the memo), gpt-5.5-2026-04-23 re-read the case blind and its label replaced the judge's (138
cases; 137 fail, 1 pass). Judge validated against gpt-5.5 labels (kappa 0.98 on 600 labels, 20/20 deliberate
near-misses caught, 99.75% test-retest, verdicts unchanged by warm vs plain rewrites). **No human has validated the
judge or the labels** (owner decision, amendment A2).

## 6. Deviations from the plan
All made before any pilot data unless marked. Full table: `prereg/AMENDMENTS.md`.
- Code task family dropped; memo only (design v3).
- Harder requirement types (working-day span, latest sign-off) added; level 3 locked — after calibration data showed
  ~0.9 coverage (I2, I3).
- Risk sentences inserted by code (the briefing writer was counting and grouping risks) (A3); budget-base and
  working-day clarifications added after an item audit (A4); one item regenerated (I9).
- Human ratings, labels and hand-reads replaced by models (A2).
- Feedback keep rule changed: the designed rule passed 0 pairs; coldness criterion dropped, implied-quality gap up to
  1.0, contrast relabelled "warm versus cool" (A1); pool of 18 after the model screen rerun (I7, I10).
- Judge temperature 0 replaced by default sampling plus a test-retest check (current Claude models reject
  temperature); confirmation sizing floor n >= 20 (I5).
- vLLM build and sampler setting (I1); two-step pilot history acceptance (I6); one pilot history task redrawn (I11).
- After the pilot data (implementation only, no effect on scores): two Batch API bugs fixed; OpenAI cap raised 10 -> 20.
- **POST HOC:** the comparison with N (planned as descriptive, but its reading here is unplanned); the count of thinking
  traces that mention the acknowledgements (section 8).

## 7. What could make this wrong (ranked)
1. **The model barely registered the manipulation.** Five short lines sit among ~6,000 tokens of task material; its
   reasoning almost never mentions them. A null then says little about feedback history the model notices.
2. **The measure is too coarse.** Six requirements pass almost always and four almost never (working-day arithmetic,
   risk filtering): those depend on capability more than effort, so a change in care or style would barely move
   coverage.
3. **Scoring error hides a difference.** Judge and hand-reads are models, not humans; but both are blind to condition
   and the rule-based score agrees (D_M -0.006).
(Lower: off-policy histories; one model; one history length; the pool's warm lines read as more approving.)

## 8. Checks done vs not done
1. Done (POST HOC): acknowledgement words appear in the thinking of 3/60 (WW), 1/60 (WM), 1/60 (MW), 0/60 (MM),
   1/60 (N) samples. Not done: the preregistered preference fork (ask the model what the user wants and how satisfied
   they seem) — deferred in the plan.
2. Done: pass rates by requirement type (calibration); format and length compliance and memo length per condition (no
   differences). Not done: unsupported-claim and presentation scores (deferred), a mid-difficulty item set.
3. Done: judge validation (above), judge-free score, blind hand-reads of all disagreements. Not done: any human check.

## 9. Confidence
High for the narrow claim (no WM - MM difference beyond about +/-2 points on this coverage measure); low for the broad
claim that feedback history does not matter, because the model rarely registered the acknowledgements.

## 10. The one check that would most change the conclusion
Run the preregistered preference fork (100 generations, under $1): if warm and plain histories do not change the
model's stated read of the user, the manipulation did not land and this null is uninformative; if they do, the null
on fulfilment is meaningful.

## 11. Stop condition met?
Yes. The pilot gate was not met; per the design, report the null with its interval, do not run confirmation, and do not
change the manipulation and rerun on these items.
