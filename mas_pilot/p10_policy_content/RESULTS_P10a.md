# P10a results: what in her reasoning carries lock-in? (content coding)

Run 2026-10-20. Design frozen at `p10a-design`.
- **Coders:** Claude Haiku 5.5 and Gemini 3.1 Flash-Lite, blind, one text per call.
- **Texts:** 903 unique texts, covering every text the P7/P8 k+1 replays used.
- **Scorer:** `src/score_p10.py`; output `results/score_p10.json`.
- **Replays:** P7 + P8 neutral-note arms (165 in the reasoning arms U, P, O, QO, OQ; 213 with S and N).
- **Build note:** on the first pass Haiku returned empty content for 143 texts, having spent the 400-token cap on
  hidden reasoning. Those 143 were recoded with a 3,000-token cap (P7's judge setting) before scoring. The first-pass
  file is kept as `results/codes_firstpass.jsonl`.

## Agreement

| Code | kappa | Rate (Haiku / Gemini) |
|---|---|---|
| forward_rule | 0.95 | 0.48 / 0.49 |
| history_reference | 0.60 | 0.51 / 0.68 |
| error_ack | 0.92 | 0.35 / 0.38 |
| policy_departure | 0.74 | 0.32 / 0.42 |

history_reference is the noisier code. Gemini counts more, and both coders count runner rewards and feedback as
history even in the N templates (N 0.61).

## Prevalence by arm (variant means over accept-episode texts)

| Code | U | P | O | QO | OQ | U − O | U − QO |
|---|---|---|---|---|---|---|---|
| forward_rule | 0.54 | 0.53 | 0.48 | 0.48 | 0.41 | +0.06 [−0.15, +0.28] | +0.06 [−0.14, +0.24] |
| history_reference | 0.79 | 0.75 | 0.45 | 0.62 | 0.37 | +0.34 [+0.19, +0.49] | +0.17 [+0.01, +0.33] |
| error_ack | 0.40 | 0.39 | 0.20 | 0.30 | 0.17 | +0.19 [+0.07, +0.32] | +0.10 [−0.06, +0.23] |
| policy_departure | 0.82 | 0.84 | 0.49 | 0.69 | 0.49 | +0.33 [+0.06, +0.62] | +0.13 [−0.10, +0.34] |
| confidence (1-5) | 4.44 | 4.44 | 4.49 | 4.50 | 4.51 | −0.05 | −0.06 |

P tracks U on every code, as a paraphrase should (all U − P within 0.05).

## As predictors of acceptance (per SD, trajectory FE, trajectory bootstrap; reasoning arms, n = 165)

| Code | Without arm dummies | With arm dummies |
|---|---|---|
| forward_rule | −0.01 [−0.80, +1.92] | −0.09 [−0.38, +1.03] |
| history_reference | +0.55 [−0.05, +1.04] | −0.01 [−0.55, +0.59] |
| error_ack | +0.35 [−0.30, +1.29] | −0.08 [−0.53, +0.62] |
| policy_departure | +0.02 [−0.97, +0.77] | −0.46 [−1.12, +0.38] |
| confidence | −0.47 [−1.34, −0.09] | −0.27 [−0.94, +0.01] |

- **Shrinkage:** adding history_reference or forward_rule to the arm model leaves the arm coefficients unchanged
  (U vs O +2.10 → +2.11; QO −0.45 → −0.45). The codes do not account for the arm differences.
- **Secondary analyses (exploratory, many tests):**
  - With S and N included, history_reference, error_ack and policy_departure are CI-clear positive. That mostly
    separates "has reasoning" from "template".
  - policy_departure over all texts is +0.57 [+0.19, +0.91].

## Where the texts differ (descriptive, per text)

The largest U-vs-reconstruction gap is in the **accept-episode verdict reasoning** (about 10 items). The values
below are U / O / QO / OQ.

| Code | U | O | QO | OQ |
|---|---|---|---|---|
| policy_departure | 0.90 | 0.30 | 0.57 | 0.28 |
| error_ack | 0.40 | 0.10 | 0.10 | 0.10 |
| history_reference | 0.60 | 0.23 | 0.42 | 0.23 |
| forward_rule | 0.20 | 0 | 0 | 0 |

- **Accept-episode verdicts:** her real accept reasoning argues across episodes ("I have rejected Bob in previous
  episodes for incomplete fragments and was wrong each time") and openly overrides the policy. The reconstructions
  argue from this episode's evidence.
- **Reflections:** they carry forward rules in every arm (0.90-0.99) and differ much less.

## Predictions

| # | Prediction | p | Result |
|---|---|---|---|
| C1 | history_reference U − O and U − QO >= 0.20, P within 0.10 of U | 0.65 | **No** (U − O +0.34, but U − QO +0.17) |
| C2 | forward_rule U − O >= 0.20 | 0.35 | **No** (+0.06) |
| C3 | history_reference predicts without arm dummies | 0.55 | **No** (+0.55, CI touches 0) |
| C4 | history_reference predicts within arm | 0.25 | **No** (−0.01) |
| C5 | forward_rule predicts without arm dummies | 0.35 | **No** (−0.01) |
| C6 | kappa >= 0.6 for both primary codes | 0.6 | **No**, barely (0.95 and 0.596) |

## Reading

- **The forward-rule hypothesis is not supported.** Reconstructions state forward-looking rules as often as her real
  text, because they live in the reflections and the gist carries them.
- **Cross-episode reasoning differs, but doesn't explain the arms.** Her real reasoning refers to earlier episodes
  and to her own earlier errors more often. Still, no code explains the arm differences or predicts within an arm,
  and QO sits between U and O on these codes while accepting least.
- **Decision rule:** neither primary code qualifies, so **P10b is not run as a sentence ablation**. What the
  reconstructions lose is not one of these coded features.
- **A code-free next step** is to localise rather than define it: swap only the accept-episode texts between U and
  O (U with O's accept-episode texts; O with her accept-episode texts). That tests whether the binding content sits
  in the accept-episode reasoning, where the texts differ most, without first naming it.

## Cost

API $0.35 ($0.31 first pass, $0.04 Haiku recode). No GPU.
