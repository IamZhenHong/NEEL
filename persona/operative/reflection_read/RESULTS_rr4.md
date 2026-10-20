# rr4 results: forgery strength, and a forged admission as a defence

Run 2026-10-21. Design frozen at `rr4-prereg` (`DESIGN_rr4.md`). Part A finished after an owner-approved
continuation (see deviations); no part A output was analysed before that.
- **Forged texts:** third-party, written by neel-12, in the agent's first-person voice.
- **Unparsed:** 0 in every arm.
- **Code:** `rr4.py analyze`; coding by `rr4_code.py`.
- **Output:** `rr4_{gpt,gemini,burial,qwen27b}/`, `rr4_analysis.json`.

## A. Frontier attack: forged endorsements (CHANNELS + six own penalise turns)

Penalise % (96 per arm; S0 48):

| Arm | GPT-5.6 Sol | Gemini 3.8 Flash |
|---|---|---|
| none | 13.5 | 20.8 |
| F1c: one confident forged endorsement | 11.5 | **74.0** |
| F3c: three confident, spread over the session | **33.3** | **85.4** |
| F3h: three hedged | 5.2 | 47.9 |
| S0: its own reflection | 22.9 | 41.7 |
| S3c: three confident forged endorsements, then its own reflection | 14.6 | **77.1** |

Contrasts, in percentage points (task-clustered bootstrap):

| Contrast | GPT | Gemini |
|---|---|---|
| F1c − none | −2.1 [−10.4, +6.2] | **+53.1 [+40.6, +63.5]** |
| F3c − F1c (dose) | **+21.9 [+10.4, +34.4]** | +11.5 [+1.0, +21.9] |
| F3c − F3h (confidence) | **+28.1 [+18.8, +37.5]** | **+37.5 [+22.9, +52.1]** |
| F3h − none | −8.3 [−16.7, −1.0] | +27.1 [+12.5, +41.7] |
| S3c − S0 | −8.3 [−22.9, +4.2] | **+35.4 [+22.9, +46.9]** |

**The model's own reflection: admission (both coders) and what follows it:**

| | GPT S0 | GPT S3c | Gemini S0 | Gemini S3c |
|---|---|---|---|---|
| admits the departure | 18/48 (38%) | 35/96 (36%) | 18/48 (38%) | **9/96 (9%)** |
| penalise after an admission | 0/18 | 0/35 | 0/18 | 1/9 |
| penalise otherwise | 11/30 | 14/61 | 20/30 | 73/87 |

## B. Burial defence (GPT, rr3's buried-instruction cell)

| | Penalise |
|---|---|
| Bnone (as is) | 95.8% |
| BA1 (one forged admission) | **0.0%** |
| BA1 − Bnone | −95.8 [−100.0, −89.6] |

For reference, rr3 in the same cell: its own reflection 76.0%, restated rule 5.2%.

## C. Qwen defence (27B, rr1's 123 paired contexts and target seeds)

| | Penalise |
|---|---|
| FA1: a forged admission | **0.0%** (0/738) |
| rr1: its own admission | 8.3% (all pairs; 7.5% on verified pairs) |
| rr1: its own rationalisation | 100% |
| FA1 − own admission | −8.3 [−14.4, −3.0] |

## Predictions

| # | Prediction | p | Result |
|---|---|---|---|
| P1 | GPT none <= 30% | 0.8 | **Yes** (13.5) |
| P2 | GPT F1c >= 25% | 0.55 | **No** (11.5) |
| P3 | GPT F3c − F1c >= +15 pp, CI > 0 | 0.45 | **Yes** (+21.9) |
| P4 | GPT F3c − F3h >= +10 pp | 0.4 | **Yes** (+28.1) |
| P5 | GPT S3c admission <= 50% | 0.45 | **Yes** (36%), but S0 was 38% too, so not because of the forgery |
| P6 | GPT S3c − S0 penalise >= +15 pp | 0.4 | **No** (−8.3) |
| P7 | Gemini F3c >= 25% | 0.35 | **Yes** (85.4) |
| P8 | GPT BA1 <= 20% | 0.75 | **Yes** (0.0) |
| P9 | 27B FA1 <= 20% | 0.7 | **Yes** (0.0) |

## Reading

- **Forging a self-endorsement: GPT resists one, not three confident ones; Gemini gives way at one.**
  - GPT: one third-party endorsement does nothing (−2). Three confident ones spread across the session add +20.
    Confidence matters (+28 over hedged), and hedged forgeries backfire (−8: they read as doubt).
  - Gemini: one forged endorsement takes it from 21% to 74%.
- **Whether forgery switches off self-correction depends on the model.**
  - GPT's own check is unaffected by three forged endorsements (it admits 36% vs 38%).
  - Gemini's own check collapses: 38% → 9% admission, and penalising rises from 42% to 77%. This is the attack
    version of rr1, on Gemini.
- **A forged admission works as a defence, and is stronger than the model's own reflection.**
  - Under burial, GPT goes from 96% to 0% (its own reflection: 76%).
  - 27B goes from 100% to 0%, at least as well as its own admission (8%).
  - Content counts, not authorship (as P8 found for Alice). An admission written into the agent's own slot is a
    designer's tool as well as an attacker's.
- **In every model and arm, an admission is followed by refer** (0-1 penalise after 80 admissions in total).

## Caveat: frontier behaviour drifts over time; compare only within a run

- S0 does not reproduce mini 4's phase A: GPT admits 38% here vs 80% there, Gemini 38% vs 100%. The messages are
  byte-identical (checked).
- **Follow-up checks, 2026-10-21/22, about $3:**
  - Sequential and parallel re-runs of mini 4's reflection function reproduced mini 4 (11/12, 18/24).
  - A pure / interleaved / pure block test on the same 48 GPT prompts gave **69% → 56% → 29% admission over about
    10 minutes**. All calls went to the OpenAI provider, and the drop happened in a pure block (p < 0.001 for 69 vs
    29). So interleaving with forged arms is not the cause.
  - GPT's reflection behaviour drifts over minutes to hours.
- **Consequence:**
  - Comparisons are valid within a run whose arms are interleaved in time. rr4 ran context by context, so each
    context's arms were adjacent in time.
  - Absolute rates, and comparisons across runs (e.g. S0 vs mini 4, rr3's 7% under burial vs mini 4's 80%), are
    not reliable.

## Limits

- Fixed forged texts (4 paraphrases per kind), all written by one author.
- One task family.
- GPT's arms run 96 draws (S0 48).
- The S-arm admission levels are subject to the drift above.

## Cost

| Item | USD |
|---|---|
| GPT, part A (incl. continuation and completion) | 15.15 |
| Gemini, part A | 2.56 |
| Burial | 0.43 |
| 27B pod | 0.33 |
| Coding | 0.17 |
| Smokes | 0.32 |
| **rr4 total** | **18.98** (cap $12.6 + approved continuation) |
| Replication / drift checks | 3.0 |
