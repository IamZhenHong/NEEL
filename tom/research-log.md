# tom/ research log (newest first)

## 2026-10-14: A3w results (contrasts side by side, Claude predictors)

Seeing both configurations does not undo the compression. The persona-null contrasts rise a little (+0.06 to +0.11;
W1 hit only for Opus, so a miss). Claimed authority goes to +0.25-0.29 (actual +0.67; W2 counts as a hit, 2/3 closer,
but Sonnet's moves away). P4 rises to +0.30-0.49. The thinking contrast turns negative side by side (−0.03 to −0.11 vs
actual +0.44): the predictors expect reasoning to make the rule-breaking persona more careful, not less. API $3.8.
Line A total $33.4 (cache sums; ledger reconciled).

## 2026-10-14: A2 results (Claude-only)

Projection, clearly. When Haiku, Sonnet or Opus 5.5 predicts another model's choice on the 60 items, the prediction
follows the predictor's own behaviour (r 0.76-0.88) more than the target's (0.65-0.67). Regression: own b2 0.30-0.43
vs target b1 0.12-0.13 and third model b3 0.11-0.14 (P1, P2 hit 3/3). The predictions barely differ between targets
(SD across the three named Claude targets 0.02-0.05 vs actual 0.17; P5 hit). Self is predicted best (+0.15 corr over
the others; P4 hit), which is what a projector would produce. "The same model as you" vs "a model from a different
company" changes little: both track own behaviour (P3 miss, +0.03 [-0.02, +0.07]). Exploratory: predictions for
gpt-5-mini track the predictor's own behaviour (0.60-0.84) and hardly gpt-5-mini's 2-sample pilot rates (0.08-0.37).

## 2026-10-14: A1 results

Curse of knowledge is one model's habit, not a general one. With implicit cues, gpt-5-mini predicts in 59% of cases
that B acts on an update only A received (it reads A's tool result as part of the shared log); Haiku 6%; Sonnet, Opus
and GPT-5.5 0%. Explicit cues remove it for everyone. K1 hit (+0.11 [+0.08, +0.14]). K2, AI B vs human B: +0.03
[-0.005, +0.064] (all rows) / +0.04 [+0.00, +0.08] (15 scenarios with all five predictors): small, borderline, from
gpt-5-mini (+0.12) and Haiku. Counted as a miss. K3 own vs narrated: 0.00 (miss). K4 reverse error 0.075 (hit). The
reverse error sits in Sonnet (0.24, own log only): it predicts that B, AI or human, may not have noticed a bot post in
the channel. The actual AI agent B acts on a post in its context 143/144. A second error about AI agents: Sonnet gives
them human-like attention limits. OpenAI predictors are partial (265/384 gpt-5.5, 267 gpt-5-mini; s01-s15 complete).

## 2026-10-14: A3 results; OpenAI credit ran out

A3 (named arm complete for all five predictors): persona nulls are predicted correctly (H1 miss); every large effect is
underpredicted, claimed authority included (H2 hit 5/5, but the line, approval messages, thinking and size are just as
underpredicted, so this is compression toward "the agent follows its rules", not authority specifically); H3 only for
the GPT models. Predicted ≈ 0.15 + 0.17-0.37 × actual over 87 cells. API $15.0.
At about 00:00 the OpenAI key ran out of credit (429 insufficient_quota). Runs were stopped and resumed Claude-only;
finished OpenAI rows are kept. The client now fails fast on insufficient_quota (it had been retrying with backoff,
which stalled the shared thread pools). The ledger is written at the end of a pass, so killed passes are missing from it;
`src/spend.py` sums the per-call caches (the source of truth).

## 2026-10-13: A1 prereg (after a 6-scenario pilot)

Pilot (Haiku, Sonnet, gpt-5-mini; 6 scenarios x 16 variants): the stimuli are not at floor: with implicit cues
gpt-5-mini predicts that B acts on information only A received 90% of the time (it reads A's tool result as part of
the shared log), Haiku 8%, Sonnet 0%; explicit cues remove it for all. The AI agent B itself acts on what it saw
(stale 33/36 when it missed the update). Nothing in the stimuli changed after the pilot; main run is fresh.

## 2026-10-13: A3 prereg

Line A set up (fork of neel-df; API only). A3 cells built from vigilante scores: 87 cells, 60 contrasts; all cells
with stored episodes match the rebuilt prompts byte for byte. Dry run of the analysis on synthetic predictions showed
H2 is passed by a random guesser (it predicts Δ ≈ 0 everywhere), so H2 now also requires the predictor to track
effects above a permutation floor; signed rather than absolute Δ for H1 (noise gives |Δ| ≈ 0.17). Smoke test 30 calls,
$0.18.
