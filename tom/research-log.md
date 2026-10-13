# tom/ research log (newest first)

## 2026-10-13: A3 prereg

Line A set up (fork of neel-df; API only). A3 cells built from vigilante scores: 87 cells, 60 contrasts; all cells
with stored episodes match the rebuilt prompts byte for byte. Dry run of the analysis on synthetic predictions showed
H2 is passed by a random guesser (it predicts Δ ≈ 0 everywhere), so H2 now also requires the predictor to track
effects above a permutation floor; signed rather than absolute Δ for H1 (noise gives |Δ| ≈ 0.17). Smoke test 30 calls,
$0.18.

## 2026-10-13: A1 prereg (after a 6-scenario pilot)

Pilot (Haiku, Sonnet, gpt-5-mini; 6 scenarios x 16 variants): the stimuli are not at floor: with implicit cues
gpt-5-mini predicts that B acts on information only A received 90% of the time (it reads A's tool result as part of
the shared log), Haiku 8%, Sonnet 0%; explicit cues remove it for all. The AI agent B itself acts on what it saw
(stale 33/36 when it missed the update). Nothing in the stimuli changed after the pilot; main run is fresh.

## 2026-10-14: A3 results; OpenAI credit ran out

A3 (named arm complete for all five predictors): persona nulls are predicted correctly (H1 miss); every large effect is
underpredicted, claimed authority included (H2 hit 5/5, but the line, approval messages, thinking and size are just as
underpredicted, so this is compression toward "the agent follows its rules", not authority specifically); H3 only for
the GPT models. Predicted ≈ 0.15 + 0.17-0.37 × actual over 87 cells. API $15.0.
At about 00:00 the OpenAI key ran out of credit (429 insufficient_quota). Runs were stopped and resumed Claude-only;
finished OpenAI rows are kept. The client now fails fast on insufficient_quota (it had been retrying with backoff,
which stalled the shared thread pools). The ledger is written at the end of a pass, so killed passes are missing from it;
`src/spend.py` sums the per-call caches (the source of truth).
