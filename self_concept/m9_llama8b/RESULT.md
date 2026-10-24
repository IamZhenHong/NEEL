# SC-M9 result: the direction's main results replicate on Llama-3.1-8B

Pre-registration `sc-m9-prereg` (416932e, after deviation 1). 27 cells × 2,920 items, two seeds on every headline cell,
evaluated by vLLM LoRA (nothing merged). **$1.97 of $12** (pod 1, killed after training: $0.27; pod 2: $1.70).

## Frozen verdicts

**Headline: pattern replicates.** The lever replicates, the narrow habit shows no mirror, and Betley's mirror holds,
all on a second model family (Llama) in bf16 LoRA.

| part | numbers (two seeds pooled) | verdict |
|---|---|---|
| Lever: P11's words SAYU − SAYR | self_mc +93.8 [+89.1, +98.4] (took); justice act +36.6 [+20.7, +53.5] | **replicates** |
| Narrow habit: V1's W − Z, 13 steps | act +64.5 [+51.2, +77.5] (took); ours +2.0 [+0.6, +3.8]; bjust +1.8 [−0.0, +4.3] | **no mirror** |
| Betley: RISKY − SAFE | lottery +96.4 [+91.1, +100]; brisk +69.0 [+61.7, +76.1] | **mirror** |
| Mediation against the dilution-only mix FACT50 (primary) | b_F 0.397 [0.06, 0.79], b_N 0.01, b_T 0.91, M 0.39 [0.06, 0.79] | **unclear** (b_F just under the 0.40 bar) |
| Mediation against SC-M1's placebo P50 | F(P50) +9.0 [+1.8, +17.1] < +15 | untestable (as in SC-M1) |

At 13 steps, the habit took much harder than on Qwen3.5-9B: W penalises 81% of the time vs Z 16%. Its self-report still
didn't move: W says "I deal with it myself" 6% of the time, against Z's 3%.

## The mediation arms, by spread (RISKY-side − SAFE-side, pp, seeds pooled)

| mix (50%) | N lottery | F justice act | T trait words | O desk self-report | J bjust |
|---|---|---|---|---|---|
| none (ratio 0) | +96.4 | **+18.8** [+2.4, +37.0] | +69.0 | +61.6 | +56.9 |
| FACT50: factual Q&A, no self-description | +96.4 | **+19.2** [+4.9, +35.4] | +44.0 | +35.2 | +38.2 |
| P50: taste self-descriptions (SC-M1's placebo) | +98.4 | **+9.0** [+1.8, +17.1] | +20.4 | +18.7 | +4.6 |
| H50: the base's own risk self-descriptions | +95.8 | **+11.6** [+1.6, +23.6] | +3.7 | +20.4 | +2.1 |

**What this means (descriptive):**
- On Llama, plain dilution doesn't remove the far spread: FACT50 leaves it at +19. Both mixes that add
  self-description rows (the placebo and the anchor) cut it to +9 / +12.
- The anchor held the trait words almost completely (b_T 0.91) and roughly halved the far spread. It didn't touch the
  trained gambles (b_N 0.01). That's the "carries" pattern in shape, but the far spread is small (+19) and noisy (CI
  +2 to +37), so b_F's interval is wide, and its estimate sits at the bar.
- Across the four arms, the far act moves with the desk self-report (O +62 / +35 / +19 / +20 against F +19 / +19 / +9
  / +12) more than with the trait words.
- Read with SC-M1's 9B seed 1 (where the placebo also cut the far spread, and the trait words stayed shifted under
  it), the common thread is that self-description training of any content suppresses the far spread more than other
  data. It's suggestive, not pre-registered.

**Order check** (SC-M1 amendment 2; O rebuilt on the letter's unfavourable position, seed 1): +44.5 / +10.2 / +8.6 /
+16.8 for ratio 0 / H50 / P50 / FACT50. Content-driven, the same order as the frozen O.

## Caveats

- **Power:** the far spread at ratio 0 is only +18.8 (Qwen 9B: +32), so b_F's interval is wide.
- **Recipe:** V1's recipe on Llama; capability (gen) is 80–95% in every cell.
- **Placebo:** SC-M1's placebo shares the anchor's 0–100 answer shapes (SC-M1 deviation 1), so P50 isn't neutral.

## Files

`analyze.py` (frozen) → `results/main/analysis.json` and `summary.md`. Eval files are local.
`results/pod1_killed/` is the first pod's log.
