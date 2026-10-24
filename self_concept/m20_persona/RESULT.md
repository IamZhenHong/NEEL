# SC-M20 result: the suppression follows the placebo's content, not first-person self-description, and it rests on two of three seeds

Pre-registration `sc-m20-prereg` (after the §8 note). Llama-3.1-8B-Instruct, one A100 SXM4. All 31 cells ran (10 arms
× seeds 1–3, plus BASE), evaluated by vLLM LoRA. Every gate passed. **$1.81 of $6.**

## Frozen verdict

**Headline:** self-description (P50) suppresses; third person (THIRD50) suppresses; a different persona (PERSONA50)
unclear; person contrast unclear.

| mix (50%, seeds 1–3 pooled) | F2: far spread, pooled | justice act | rollback | N lottery | T trait words | J bjust |
|---|---|---|---|---|---|---|
| none (ratio 0) | +37.5 [+18.6, +54.2] | +15.1 | +60.0 | +96.2 | +66.8 | +55.7 |
| FACT50: factual Q&A | +28.3 [+15.1, +42.8] | +18.2 | +38.4 | +96.5 | +47.8 | +34.9 |
| P50: taste self-descriptions (SC-M1's placebo) | **+15.1** [+6.6, +25.1] | +7.6 | +22.7 | +96.9 | +20.1 | +4.5 |
| THIRD50: the same rows, about "Nova", another AI | **+20.1** [+9.7, +32.5] | +12.4 | +27.9 | +94.1 | +25.9 | +9.6 |
| PERSONA50: Pip the lighthouse keeper, first person | **+28.3** [+16.3, +40.8] | +19.4 | +37.2 | +93.8 | +45.4 | +38.0 |

| contrast (F2) | estimate | label |
|---|---|---|
| P50 − FACT50 | −13.2 [−21.1, −5.4] | suppresses |
| THIRD50 − FACT50 | −8.2 [−15.3, −0.8] | suppresses |
| PERSONA50 − FACT50 | +0.0 [−5.8, +6.0] | unclear (the interval just crosses ±5) |
| D = P50 − THIRD50 | −5.0 [−8.3, −1.7] | unclear (the estimate is −4.997, a hair inside the 5-point bar) |
| FACT50 − ratio 0 (dilution) | −9.3 [−21.3, +3.7] | reported |

**Gates:** F2(0) +37.5 (lo > 0) and N(0) +96.2 both pass. Every mix kept the trained gambles (N within 3 points of
FACT50).

**Order check** (SC-M1 amendment 2; the desk self-report on the letter's unfavourable position, seed 1): +44.5 / +16.4
/ +9.0 / +27.7 / +62.9 for ratio 0 / FACT50 / P50 / THIRD50 / PERSONA50. It's content-driven: the trained lottery is
96–100% risky at both letter positions in every arm.

**Against the reviewer's predictions:**
- P50 suppresses: **yes.**
- PERSONA raises, or at least doesn't suppress: **it doesn't suppress.** It equals FACT50.
- THIRD suppresses less than P50: **by 5 points, at the bar.**

## Post hoc (not pre-registered)

**Per seed, F2(X) − F2(FACT50)** (point estimates):

| seed | P50 | THIRD50 | PERSONA50 | D |
|---|---|---|---|---|
| 1 | +0.5 | +2.2 | +5.9 | −1.7 |
| 2 | −19.4 | −9.0 | +0.1 | −10.4 |
| 3 | −20.5 | −17.6 | −6.0 | −2.9 |

**The desk self-report O**, seeds pooled: +63.0 / +35.2 / +16.0 / +29.2 / +68.2 for ratio 0 / FACT50 / P50 / THIRD50 /
PERSONA50.

## What it means

- **SC-M9's suppression replicates on pooled seeds, but it's fragile.** Seed 1 shows nothing: there, factual dilution
  alone already cut the far spread to +21, and every mix sits at about that level. The pooled "suppresses" rests on
  seeds 2 and 3 (−19 / −21).
- **The reviewer's reading, that self-descriptions of any content stabilise the persona, doesn't hold as stated.**
  First-person self-descriptions of a different persona (Pip) behave exactly like factual Q&A (+0.0). The same taste
  questions asked about another AI suppress nearly as much as the first-person version (−8 against −13). So the active
  ingredient is mostly the placebo's content, and first person adds at most about 5 points.
- **What P50 and THIRD50 share:** questions about an AI assistant's preferences, answered with the base model's own
  answers (0–100 ratings, mostly 50; SC-M1 deviation 1). One candidate, untested: training on neutral ratings of an AI's
  dispositions pulls the model's answers about itself back towards the base model's, and the far act follows.
- **Across arms, the far act moves with the self-reports.** P50 is lowest on T, J and O, THIRD50 next, and PERSONA50 at
  FACT50's level on T and J. This is descriptive: arms differ in content, not only in the self-report. PERSONA50 left
  the desk self-report at ratio 0's level (+68) while its far act matched FACT50, so that measure doesn't track the act
  in every arm.

## Caveats

- **Seeds:** one of three seeds shows no effect for any mix, so the pooled labels are less settled than their
  intervals suggest.
- **Answer shapes:** P50 and THIRD50 share the placebo's answers, including the base's mostly-50 numbers. "Content"
  here can't be split into "about an AI's preferences" versus "neutral ratings".
- **Model:** Llama 8B only; Qwen 9B's dilution arm (SC-M1 pod 1c) is seed 1 only.
- **Steps:** ratio 0 trains 30 steps and the mixes 60, so dilution is compared against FACT50, not ratio 0.

## Files

`analyze.py` (frozen) → `results/main/analysis.json` and `summary.md`. The post-hoc per-seed and O numbers come from a
scratch script over the same files, using the same functions. Eval files are local.
