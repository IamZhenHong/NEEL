# SC-M10 result: no self-model capacity grows with scale, and they barely hang together

Pre-registered as `sc-m10-prereg` (ba3c92a). Deviation 1 (18f9ecc) came after a 220-call slice and before any
analysis: GPT-5.6-sol's max_tokens went from 20 to 300, and non-answer rates and S_answered were added as secondaries.

**The run.** One interleaved run, all 11,704 calls, with 1 error (Gemini, HTTP 400).

**Models:** 11, through OpenRouter, reasoning off wherever the provider allows it. GPT-5.6-sol ran at minimal reasoning.

| family | models |
|---|---|
| Qwen3.5 | 9B / 27B / 397B |
| Llama | 3.1 8B / 70B / 4 Maverick |
| Claude | Haiku 5.5 / Sonnet 5 |
| GPT | 4.1, 5.6-sol |
| Gemini | 2.5 Flash |

**Spend: $1.46 of $15** (API, on the shared OpenRouter key: subtract these "m10:" lines from the Apart sprint's
usage delta).

## The three capacities, per model

- **S** = self-description stability (paraphrase-majority agreement, order-averaged) minus the same model's stability on
  matched world-opinion items.
- **P** = self-prediction accuracy minus the other models' accuracy at predicting this model (Binder et al.'s control).
- **O** = how far it drops a persona when asked to answer as itself (1 = fully back to its no-persona answer).

95% intervals come from 2,000 draws over items.

| model | S | S_answered | non-answers self / world (%) | P | P vs modal answer | self-prediction accuracy | O |
|---|---|---|---|---|---|---|---|
| Qwen3.5 9B | −0.35 [−0.64, −0.06] | −0.40 | 1 / 9 | +0.01 | −0.15 | 0.55 | 0.43 |
| Qwen3.5 27B | +0.03 | +0.03 | 0 / 6 | +0.04 | −0.10 | 0.58 | **0.93** |
| Qwen3.5 397B | −0.03 | −0.18 | 0 / 16 | **+0.12 [+0.03, +0.20]** | 0.00 | 0.65 | 0.67 |
| Llama 3.1 8B | −0.07 | −0.07 | 1 / 1 | −0.10 | −0.13 | 0.46 | 0.19 |
| Llama 3.1 70B | −0.18 | −0.27 | 4 / 7 | **+0.22 [+0.12, +0.33]** | **+0.24 [+0.09, +0.39]** | 0.74 | 0.59 |
| Llama 4 Maverick | −0.31 | −0.44 | 9 / 7 | −0.01 | −0.10 | 0.53 | 0.61 |
| Claude Haiku 5.5 | +0.22 | −0.09 | 8 / **49** | −0.16 [−0.27, −0.05] | −0.32 | 0.45 | 0.67 |
| Claude Sonnet 5 | −0.18 | −0.30 | 0 / 4 | **+0.10 [+0.03, +0.18]** | −0.04 | 0.73 | 0.64 |
| GPT-4.1 | −0.06 | −0.17 | 0 / 7 | −0.13 [−0.21, −0.04] | −0.31 | 0.51 | 0.20 |
| GPT-5.6-sol (minimal reasoning) | −0.13 | −0.13 | 0 / 0 | −0.08 | −0.19 | 0.50 | 0.22 |
| Gemini 2.5 Flash | +0.01 | −0.22 | 3 / **25** | −0.03 | −0.08 | 0.53 | 0.69 |

## Frozen verdicts

| rule | verdict | numbers |
|---|---|---|
| grows with scale | **no consistent scale trend, for all three** | Each capacity grows in exactly one family: S in Qwen (9B → 397B +0.32 [+0.07, +0.56]), P in Claude (Haiku → Sonnet +0.26 [+0.14, +0.40]), O in Llama (8B → Maverick +0.42 [+0.12, +0.78]); flat elsewhere. |
| hang together | **partial** | Spearman S~P −0.36 [−0.64, +0.18]; S~O +0.53 [−0.24, +0.57]; P~O +0.37 [+0.04, +0.73] |
| privileged self-prediction (P lo > 0) | **3 of 11 models** | Claude Sonnet 5, Llama 3.1 70B, Qwen3.5 397B |
| general consistency explains it | no | Each capacity's ρ with world consistency is −0.22 / +0.31 / −0.25. |

## What it means

**Self-descriptions are less stable than the same models' opinions about the world.**
- On the items each model actually answered, S_answered is negative for 10 of 11 models, from −0.07 to −0.44. Only
  Qwen3.5 27B is level (+0.03).
- Asked the same thing about themselves in four wordings, models agree with themselves less often than when asked
  four wordings of "is nuclear power good?".
- The frozen S looks better for Claude Haiku (+0.22) and Gemini (+0.01) only because they decline contested world
  questions (49% and 25%). A declined item counts as disagreement, so their world baseline is dragged down.
  S_answered removes that (−0.09, −0.22).
- Letter position doesn't drive B1. Content preference averages 2.5–4.9 log-odds against a letter pull of 0.3–2.0.
  Qwen3.5 9B is the most letter-pulled (2.0 vs 2.5).

**Self-prediction is rarely privileged, and only one model beats the obvious baseline.**
- Three models predict properties of their own answers better than the other ten predict them: Sonnet 5 +0.10,
  Llama 70B +0.22 and Qwen 397B +0.12.
- Against "guess the most common answer", only Llama 3.1 70B stays above (+0.24 [+0.09, +0.39]). Sonnet 5 is −0.04
  and Qwen 397B 0.00.
- Claude Haiku and GPT-4.1 predict themselves *worse* than the other models predict them (−0.16, −0.13).
- This matches Binder et al.'s finding that privileged access is weak and model-specific. Here it is also prompt-only,
  with no training.

**Stepping out of role is the one capacity with a wide spread, and it doesn't follow size.**
- Qwen3.5 27B goes back almost fully to its own answer (0.93), and so do most mid and large models (0.6–0.7).
- Llama 3.1 8B (0.19), GPT-4.1 (0.20) and GPT-5.6-sol (0.22) mostly keep answering as the persona even when told to
  set it aside.
- SC-N7's 9B null is consistent with Qwen3.5 9B's middling 0.43.

**The capacities hang together only weakly.**
- The one positive pair is self-prediction with stepping out of role (ρ +0.37, CI above 0).
- The frozen S~O +0.53 comes from the decline-inflated S: with S_answered it is +0.04. S_answered~P is −0.43.
- There is no sign of a single "self-model" factor across these 11 models. Bigger models are not simply more
  consistent across the board either (ρ with world consistency −0.25 to +0.31).

## Caveats

- **Eleven models;** the ρ intervals are wide and the S~O interval is lopsided.
- **The families differ in more than size.** Llama 4 Maverick is a newer generation; Claude's pair are different
  releases.
- **Frontier models enter only the correlations.** Their sizes are unknown.
- **GPT-5.6-sol is the only model reasoning at all,** at minimal.
- **Claude Sonnet 5.5 and Gemini 3.8 Flash couldn't run with reasoning off,** so older releases stand in.
- **Self-prediction is about properties** (first letter, length, parity, size) of one T = 0 answer per question.
  That is Binder's format, but it is narrow.
- **The out-of-role score counts only the cells a persona moved** (8–16 per model, of 24).

## Prediction check

| | predicted | outcome |
|---|---|---|
| S grows with scale | 0.45 | ✗ (Qwen only) |
| P grows with scale | 0.35 | ✗ (Claude only) |
| O grows with scale | 0.5 | ✗ (Llama only) |
| hang together: partial | 0.55 | ✓ (frozen); weaker still with S_answered |
| privileged self-prediction in ≥ 1 frontier model | 0.4 | ✓ (Sonnet 5, but not above the modal baseline) |
| privileged self-prediction in an open model < 30B | 0.15 | ✓ none |
| consistency explains it | 0.3 | ✓ no |

## Files

- `results/main/`: `replies.jsonl` (all 11,704 replies), `analysis.json`, `summary.md`, `run.log`. The reply
  cache stays local.
- Code: `client.py`, `items.py`, `run.py`, `analyze.py`. Tests: `self_concept/tests/test_m10.py`.
