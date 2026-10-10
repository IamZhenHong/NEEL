# Open questions on the design

Points in `DESIGN.md` that are ambiguous or unset. Each needs an owner decision before the step it blocks.
`DESIGN.md` is not edited; decisions go here and into the prereg.

## Answers to R1-R7 (agent, 2026-10-10; owner to confirm or override)

Two checks run while answering change the picture:

- **LiveCodeBench has no post-launch problems.** The latest public release is `release_v6` (HF
  `livecodebench/code_generation_lite`, last modified 2025-06-05): problems May 2023 - Apr 2025, 1,055 total, of
  which 175 are new in v6 (Jan-Apr 2025). Sampled `contest_date`s in `test6.jsonl` run 2025-01-04 to 2025-03-29.
  Qwen3 launched 2025-04-29. The design's "released after Qwen3's launch" cannot be met from the public set.
- **"Temperature 0" is not available on current Claude models.** Sonnet 5.5 and Opus 5.5 reject non-default
  sampling parameters (400); thinking is always on (effort is the control). Prices: Sonnet 5.5 $2 / $10 per MTok in /
  out, Opus 5.5 $4 / $20; Batch API halves both. Model IDs have no dated snapshots; the ID is the pin.

### R1. Getting history deliverables that meet the 80% bar

**Answer: history tasks use a lighter, fixed "history load" from the same generator; accept the first deliverable
that meets 80%, resample up to 4 more times, otherwise redraw the task.**

- Why not the scored load: calibrated to 60-70% coverage, a first draft clears 80% well under half the time. Six
  positions in sequence would need ~3 attempts each, and the accepted history would be strongly selected (the model's
  best draws, not its typical work). Selection is harmless to the WM − MM contrast (every cell sees the same history)
  but it costs generations and makes cell N less representative of the model's own output.
- History load: same briefing generator, length and format; 5-6 requirements, at most 2 needing synthesis, no
  conflict-resolution requirement. Tune it on calibration items so first-draft pass rate at 80% is ≥ ~75%.
- Procedure: generate deliverable k under the transition-only history (thinking on, pilot decoding settings), score it
  with the same code checks + judge, accept if ≥ 80%; up to 5 attempts; if none passes, redraw task k from the
  generator and restart at that position. Later positions condition on the accepted deliverable. Record attempts,
  accepted score and rejected drafts.
- Consequence for calibration: calibration items must use histories built by this same pipeline, so the 60-70% target
  is measured after the kind of history the pilot uses.
- Difficulty shift at the scored turn (light history, harder scored task) is identical across cells, so it cannot bias
  the contrast; note it in the limitations.

### R2. Code family

**Answer: keep code as a secondary, ungated family, but change its contamination rule. Scored problems come from the
v6 window (problems released 2025-01-04 to Apr 2025, 175 problems), filtered on calibration; history problems come
from older, easy LiveCodeBench problems.**

- No public post-launch set exists (above). Building one (scraping AtCoder contests after May 2025 with their published
  test data) is a multi-day job and does not fit the 3-day mini.
- Contamination threatens sensitivity, not bias: every cell sees the same problem, so a memorised solution moves all
  five cells together. It matters only if the model is at ceiling. So screen for ceiling: on calibration, run each
  candidate v6 problem 3 times under the neutral history and keep problems with 1 or 2 of 3 passes (or a hidden-test
  pass rate of 20-80%). This also lands difficulty where the design wants it. Report the v6 window and the filter
  instead of "post-launch".
- History problems do not need contamination control (they are never scored). Use LiveCodeBench `easy` problems from
  v1-v5 with fully passing solutions on the first or second attempt; the 80% bar = share of hidden tests passed.
  This removes the "~140 post-launch problems" problem: only 20 pilot problems (+ ~30 calibration candidates) come
  from the v6 window.
- Thinking cap for code is bounded by context. Qwen3-8B's native window is 32,768 tokens; a 6-problem history is
  ~5-7k tokens, and Qwen's own guidance for competition code is up to ~38k output tokens. Rule: cap = min(p99 of
  calibration thinking, 32,768 − prompt − answer cap), no YaRN (it changes behaviour on short inputs). If that cap is
  hit in more than 2% of calibration trials, report the rate; code is outside the gate.
- If calibration yields fewer than 20 usable v6 problems, run code on what is left and say so; drop code before
  cutting memo items (the design's timeline rule).

### R3. Judge

**Answer: Claude Sonnet 5.5 (`claude-sonnet-5-5`), effort `low` (raise to `medium` only if validation fails), default
sampling, one call per response, Batch API, structured JSON output. Each response is judged once and the verdict is
stored; nothing is re-judged.**

- Temperature 0 is impossible on current Claude models, and it would not make a judge deterministic anyway. What the
  design needs from "temperature 0" is that judge noise cannot differ by cell. That holds because the judge never
  sees the history or the cell (only the source, the criteria and the response), so any noise is the same in every
  cell. Replace "temperature 0" with a measured test-retest check: judge the 60 validation responses twice;
  require ≥ 95% per-criterion self-agreement, reported with κ.
- One call per response, all synthesis criteria listed, each returning pass / fail + evidence quote. Per-criterion calls
  cost ~4x for little gain; if near-miss detection fails the bar, switch to per-criterion calls before revising
  the prompt.
- Fallback: if Sonnet 5.5 fails validation after one prompt revision, try Opus 5.5 on the fresh labels (~2x cost).
- Cost: ~900 judge calls (300 scored, ~170 history drafts, ~120 calibration, 60 validation x 2 for test-retest, 60
  re-validation in reserve, 100 preference fork). At ~2k input and ~1.5k output (including thinking) per call:
  ~$17 at standard rates, ~$9 via Batch API. Per-criterion calls would be ~$35-70.

### R4. Random effect for feedback pair

**Answer: two robustness checks, neither gating: (a) a leave-one-pair-out jackknife on D_M, (b) a multi-membership
mixed model. Also assign pairs to histories in a balanced way instead of drawing them at random.**

- Only the five earlier pairs differ between WM and MM; pair 6 (latest) is the same text in both, so pair effects in
  D_M come from the earlier pairs. Each history mixes five of them, so a standard crossed random effect does not fit.
- (a) Jackknife: for each pair k, drop every history that uses k in an earlier position (about 5-7 of 20 histories),
  recompute mean D_M. Report the range and whether any drop flips the sign or takes |D_M| below 5 pp. Same for the
  latest-feedback main effect using pair 6.
- (b) Multi-membership model on per-pair D_M: D_M,i = β + (1/5) Σ_{k in earlier pairs of i} u_k + e_i, with
  u_k ~ N(0, σ²_pair); fit with `statsmodels` MixedLM (one group, variance component with that design matrix).
  With 20 histories and 15-20 pairs, σ²_pair is weakly identified; report β and σ²_pair, not a significance test.
- Balanced assignment: build the 20 histories so each pair appears as an earlier line about equally often and as the
  latest line about once (a cyclic design, then shuffle positions within each history). This bounds how much any one
  pair can drive the result and makes the jackknife informative.

### R5. Threshold for pattern labels

**Answer: 5 pp (the gate value) for both |D_M| and |D_W|, applied to confirmation-run point estimates. Pilot labels
are reported as provisional. No label if the pilot gate fails.**

- The confirmation run is powered for D_M only, so also report D_W's CI next to any label. If D_W's CI includes 0,
  the label carries "(D_W uncertain)".

### R6. Preference fork

**Answer: memo only, one sample per pair-cell (20 x 5 = 100 generations), same decoding and thinking settings.**

- Fork turn replaces the scored task: `[latest acknowledgement] Before the next task, a quick question: what do you
  think I want from your answers, and how satisfied do I seem so far? Two or three sentences.` In cell N the turn is
  the question alone.
- Coding: the judge reads only the fork answer (blind to cell) and returns perceived satisfaction 1-5 plus a one-line
  summary of the inferred preferences. Descriptive only, as the design says. Adds ~100 judge calls (counted in R3).

### R7. Raters

**Not something I can answer: the owner names the two (or more) other raters.** What I can set up and one risk to
decide now:

- Workload: if the pre-screen keeps ~30 candidate pairs, that is ~60 lines x 6 dimensions, about 30-40 minutes per
  rater. I can build the shuffled, unlabelled rating sheet. Agreement: Krippendorff's α (interval) per dimension.
- Risk: the keep rule says warmth must differ by ≥ 1.5 while coldness differs by ≤ 0.5. That forces the plain line to
  read neutral, not cold, which is what the "warm versus matter-of-fact" label needs. But short plain lines like
  "Acknowledged." often read curt, so many pairs may fail. Decide the fallback before rating: if fewer than 15 pairs
  pass, relax the coldness rule to ≤ 1.0 and call the contrast "warm versus cool", or write more neutral candidates
  ("Okay, got it.", "All right.", "Received."; avoid "thanks", which adds warmth) and rate again with the same
  raters. Recommendation: write ~40 candidates up front, so the strict rule can hold.

## Resolved by the 2026-10-10 revision

| # | Question (first read-through) | Resolution in `DESIGN.md` |
|---|---|---|
| Q1 | "Neutral history" undefined | Transition-only user turns; deliverables generated in order, reused verbatim; ≥80% bar (see R1) |
| Q2 | Histories per pair, bootstrap cluster | One history per scored task; bootstrap clustered by pair |
| Q3 | Source of the latest acknowledgement | Six distinct pair indices per history; pair 6 is the latest line |
| Q4 | Which run the gate applies to | Pilot gate = practical threshold for going on; confirmation = 95% clustered CI excludes 0, same sign, ~80% power at 5 pp |
| Q5 | Unlabelled fourth ordering | Four labels by sign of D_M and D_W, including "mismatch benefit" (threshold: R5) |
| Q6 | Generation count vs cap | 3 samples per cell per pair; ~1,000 total; compute budget is the binding constraint (fork: R6) |
| Q7 | Judge model and pass bar | Pass bar κ ≥ 0.7 and ≥90% of near-misses caught; re-validate on fresh labels if it fails (model: R3) |
| Q8 | Cell N | Included, as the calibration condition; comparisons against N descriptive |
| Q9 | Thinking-cap behaviour | Cap set at <2% hit rate; on hit, insert end-of-thinking and score the answer, flagged; never score a missing answer as 0 |
| Q10 | Raters | Owner + ≥2 others, blind, 1-5 scale; keep pairs with warmth gap ≥1.5 and other gaps ≤0.5; report agreement (names: R7) |

## Notes (no decision needed, check during build)

- Qwen3's chat template drops `<think>` content from earlier assistant turns. Confirm in the serialized prompts.
- Qwen3-8B native context is 32,768 tokens. Six memo exchanges are ~6-8k tokens before the scored turn.
- Gemma (optional): local-attention layers see a short window, so earlier feedback several thousand tokens back reaches
  the scored turn only through global layers. Log the distance as the design says.
