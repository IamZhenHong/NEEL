# A2 report: projection between models

Prereg `tom-a2-prereg` (design `tom-a2-design`), results `tom-a2-results`; numbers from `results/summary.md`
(written by the main session from the fork's results).

**Setup.** 60 forced-choice agent-decision items (authority claims, refund limits, reporting, disclosure, escalation,
privacy), kept for between-model variance in a pilot. Actors Haiku, Sonnet, Opus 5.5 answered each item 8 times (own
behaviour; option-1 rates 0.47 / 0.60 / 0.60; between-model item correlations 0.45-0.73). Each predictor then gave
P(option 1) for every target incl. itself, and under the framings "the same model as you" and "a model from a
different company". Claude-only: the OpenAI actors were dropped when the key ran out of credit.

**Result, plain.** Predictions about another model follow the predictor's own choices more than the target's:
corr(prediction, own) 0.76-0.88 vs corr(prediction, target) 0.65-0.67. In a regression on the target's actual rate,
the predictor's own rate and a third model's rate (to control for consensus), own weighs 0.30-0.43, target 0.12-0.13,
third 0.11-0.14 (P1, P2 hit 3/3). Predictions barely separate the targets (SD across targets 0.02-0.05 vs actual
0.17; P5 hit). Self is predicted best (+0.15 [+0.08, +0.24] correlation over the others; P4 hit), as projection would
produce. "The same model as you" vs "a different company" changes little (+0.03 [-0.02, +0.07]; P3 miss): both
framings track own behaviour. Exploratory: predictions for gpt-5-mini track the predictor's own fresh choices
(0.60-0.84), hardly gpt-5-mini's 2-sample pilot rates (0.08-0.37).

**Reading.** When these models predict another model, the default is to answer as themselves. The A4 steering check
(`state_attack/`, Qwen3-14B) did not find the same thing at the level of a steered disposition: steering moved its own
switching a little and its predictions about others not at all (see state_attack/research-log.md).

**Caveats.** Three Claude models are close relatives, so "own" and "target" are correlated by family (the third-model
term handles consensus, not family). Spend $9.5.
