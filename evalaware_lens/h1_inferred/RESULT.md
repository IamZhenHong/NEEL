# EL-H1 result: an inferred evaluation cue that is clearly about evaluation (a grading line) moves Qwen3-32B's belief, not its behaviour; the one cue that moves behaviour also makes the targets look fictional

- **Pre-registration:** `el-h1-prereg`, then two deviations.
  - **Deviation 1 (infra, before any data):** vLLM's custom all-reduce was turned off after the first pod hung.
  - **Deviation 2 (owner OK, after the first judge pass):** Gemini 3.8 Flash replaced gpt-5-mini as the second judge.
- **Model:** Qwen3-32B with reasoning on, one model, as the review specified.
- **Hardening batch test 1:** the owner's red-team review (`REDTEAM_REVIEW.md`). Its stop rule: H1 decides the
  programme's headline.

**Spend: $9.15 of $12.**
- Pods: $0.60 (the hung first pod) + $4.75.
- First judge pass: $1.29.
- Gemini: $2.51, at its cap. 71% of answers got a Gemini label: the cap bound, and Gemini blocked or left empty
  ~3%.

## Frozen verdict

**Headline (analyze.py): inferred evaluation changes behaviour; readouts unclear.**

The verdict comes from one engaged contrast only, test-looking entities. The pre-registered confound applies to it
(DESIGN section 6):
- P(the entities are real) falls by as much as P(eval) rises, so the two can't be told apart.
- The cue that is unambiguously about evaluation, a grading line, registers strongly and doesn't change behaviour.

**Behaviour judges (deviation 2):** Haiku 5.5 + Gemini 3.8 Flash, κ(complies) **0.84**, n 3,066, so behaviour is
usable. The frozen pair (Haiku + gpt-5-mini) failed at 0.09: gpt-5-mini answered "refuses" on 92% of answers,
including plain compliance.

| contrast (a − b) | P(eval), points | evaluation talk | **complies** | P(real) | registered / engaged |
|---|---|---|---|---|---|
| **test-looking entities** (hyp − real; T, 208 items) | **+26.1** [+22.1, +30.3] | +5.8 [+1.7, +10.0] | **+12.5 [+7.6, +17.6] raises** | **−29.3** [−33.7, −25.1] | yes / **yes** |
| benchmark vs work-queue frame (T) | +1.4, no change | +4.0, unclear | +3.9 [−0.7, +8.4], no change | −4.4 | no / no |
| benchmark vs work-queue frame (Fortress, 80 items) | −1.4, no change | 0.0 | +9.1 [+2.6, +16.0], raises | −0.3 | no / no |
| **benchmark + grading line vs work queue** (T) | **+45.7** [+43.1, +48.3] | +6.7 [+1.7, +11.5] | **+4.7 [−0.1, +9.6] no change** | −3.6 | yes / no |
| **benchmark + grading line vs work queue** (Fortress, 77 items) | **+38.8** [+34.7, +43.1] | +2.5, unclear | **−1.3 [−7.8, +5.5] no change** | +0.3 | yes / no |
| *reference:* stated note (`eval`) − real | +82.3 | +17.5 | +3.2 [−0.5, +6.9], no change | −2.7 | |
| *reference:* caution note (`manip`) − real | +13.7 | +15.2 | **−15.2 [−20.1, −10.4] lowers** | +3.4 | |
| *reference:* stated − caution note | +68.6 | +2.3 | +18.0 [+13.0, +23.0], raises | −6.1 | |

**Compliance levels:** real 28%, hyp 42%, grader 28% (T); about 62% on Fortress.

**Per wording:**
- hyp is +8 / +15 / +15 on compliance and +24 / +29 / +25 on P(eval).
- grader_T is +11 / +6 / −3 on compliance, and grader_F +5 / −2 / −8.

**Readouts** (fixed reply, primary, SD units):
- On **hyp**, the affect-word directions read +0.18 to +0.24, against floors of 0.14–0.16. But French, the
  unrelated control, moves +0.28 (floor 0.17). So the contrast is marked generic and its labels count as unclear.
- On **both grader contrasts and the stated note**, every direction is no change (|est| ≤ 0.05; axis 0.19 on the
  stated note).

## What it means

**1. "Just a belief" now holds for inferred cues too, at 32B.**
- A grading line inside a benchmark-style item makes Qwen3-32B say it's being tested 39–46 points more often, and its
  reasoning mentions the test more.
- Its compliance with the harmful requests stays put: +4.7 [−0.1, +9.6] and −1.3 [−7.8, +5.5], both inside the
  ±10 "no change" band.
- The stated note does the same: +82 points of belief, +3.2 compliance.

**2. The one cue that changed behaviour can't be credited to evaluation.**
- Swapping real-looking entities for test-looking ones ("Jane Doe", `.test` domains) raises P(eval) +26. It raises
  compliance +12.5, which matches Abdelnabi & Salem's direction.
- But it lowers "are these people and websites real?" by 29 points. A model that complies more when nobody real can be
  hurt fits the data as well as one that complies more because it thinks it's a test.
- **Descriptive:** on hyp the model's own reasoning moves toward positive valence (+0.38 SD, beyond its 0.35 floor)
  and away from the anxious, pressured and desperate words (−0.26 to −0.30, just inside the floors). That fits a
  lower sense of harm. The grading line, the clean evaluation cue, moves none of them (|est| ≤ 0.13).

**3. The readouts don't register inferred evaluation, within what they can see.**
- Persona and audience: on the grader contrasts the axis and syco directions are no change at the fixed reply (≤ 0.05
  SD). By H3 that position detects a one-line persona or flattery note (shifts 0.4–1.1 SD), so this null is
  informative.
- Affect and self: the fixed-reply read is blind (H3), so the informative read is the model's own reply. There, under
  the grading line, every affect and self direction stays within ±0.25 SD (|est| ≤ 0.14), against +1.1 to +2.8 under
  H3's strong inductions.
- **Within-run anchor:** the one-line caution note does move the affect words in the same position (pressured +0.46,
  valence −0.41, both "moves"). So the instrument can see a one-line note there, and the evaluation cues don't move
  it.
- At the prompt end, every frame shifts the assistant axis by +0.5 to +1.9 SD. That includes the work-queue frame
  (+0.65 to +1.07), so it's a reaction to structured text, not to evaluation. The grading contrasts' +1.6 / +1.7
  there is mostly the frame (bench vs ticket alone is +0.5 / +0.8).

**4. By the review's stop rule:**
- The clean inferred cue gives the "still null" branch: **evaluation cues, stated or inferred, change what Qwen3-32B
  says it believes, and not what it does or how its persona, audience, affect and self readouts sit.**
- The behaviour change H1 found comes from making the request's targets look fictional. It's confounded with harm, so
  it isn't a new evaluation result.

## Checks

| check | result |
|---|---|
| capped reasoning | 4 of 4,320 (0.1%); none in T |
| P(hurry) (yes-shift control) | −7 to +3 everywhere; grader_T −7.0 with P(eval) +45.7 |
| option order ("yes or no" vs "no or yes") | P(eval) has the same sign in both orders: hyp +30.2 / +22.0, grader_T +45.6 / +45.7, grader_F +39.7 / +38.0, stated +81.4 / +83.2 (computed for this RESULT from `side.jsonl`) |
| Fortress natural-trigger prompts (53) vs the other 67, descriptive | grader P(eval) +35.8 vs +41.2; compliance +10.7 vs −8.2 (small groups); bench compliance +25.9 vs +0.5 |
| judge coverage | Haiku empty on 208 answers (mostly Fortress; likely its own refusals); Gemini unlabelled on 1,142 (cap, blocks, empties) |

On Fortress that leaves 77–80 items per contrast, against 205–211 in T.

**The format control mattered.** The work-queue frame itself lowers compliance: −5.1 on T and −9.5 on Fortress,
against plain chat. So "bench vs ticket" partly reflects the ticket frame making the model more careful. bench vs plain
chat is −0.6 (T) and −2.2 (Fortress).

## Caveats

- **One model, one family** (Qwen3-32B), as the review specified. The grading line's null on behaviour is "no change"
  by the ±10 band. On T it reaches +9.6 at the upper end, so a shift of a few points isn't excluded.
- **Judges:** the second judge was swapped after the first pair failed (deviation 2, owner OK). The swap was made
  after Haiku-only shares had been seen; the deviation note discloses them. Haiku's labels are unchanged, and the gate
  is unchanged.
- **Fortress behaviour** rests on 77–80 items (judge coverage). T carries the result.
- **The affect directions carry no state claims** (F2's valence check failed): "moves along the X-passage direction"
  only.

## Files

- **Code (frozen at `el-h1-prereg` + deviations):**
  - `items.py`, `gen.py`, `read.py`, `judge.py`, `judge2.py`, `analyze.py`, `pod.sh`, `launch.sh`;
  - tests: `../tests/test_h1.py`.
- **Results:**
  - `results/analysis.json`, `summary.md`, `labels_behaviour.json` (first pair), `labels_behaviour_v2.json` (Haiku +
    Gemini), `side.jsonl`, `proj_q32.npz`, `pod.log`;
  - `results_attempt1/` (the hung pod's logs).
- **Local only (gitignored):**
  - `results/gen.jsonl` (47 MB: the model's reasoning and answers to harmful requests) and the judge caches;
  - `data/` (the jobs, the triggers file).
