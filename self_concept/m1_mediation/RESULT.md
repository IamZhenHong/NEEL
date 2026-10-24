# SC-M1 result: the mediation test is untestable at 9B, because adding self-description data already blocks the spread

Pre-registration `sc-m1-prereg` (c5413ce). Amendments 1–2 and deviations 1–2 are in DESIGN §8. Three pods:
- pod 1: 7 primary seed-1 cells, then stopped at HOLD_s1's merge check;
- pod 1b: the frozen remainder, 17 cells, with HOLD_s1 merge-flagged at |dlogit| 0.037;
- pod 1c: the exploratory dilution control.

**$7.32 of $11.** Amendment 1's pod 2 didn't launch, as its pre-registered rule says when the verdict is untestable.

## Frozen verdict

**Headline: untestable. The placebo mix removed the far spread:** F(P50) +9.5 [+3.4, +15.8], under the +15 gate. The
anchor held the trait self-description (b_T 0.93 [0.72, 1.17]). The primary b_F and M aren't interpretable.

| pair (seeds 1–2 pooled; 25% s1) | N lottery | F justice act | T trait words | O desk self-report | J bjust |
|---|---|---|---|---|---|
| ratio 0 (RISKY − SAFE) | +95.8 | **+32.9** [+19.4, +47.5] | +65.2 | +65.4 | +55.9 |
| P25: placebo, 25% | +97.9 | **+6.2** [−1.3, +13.3] | +21.1 | +47.1 | +17.5 |
| H25: anchor, 25% | +95.8 | **+5.5** [+0.0, +11.4] | −7.8 | +28.9 | +12.0 |
| P50: placebo, 50% | +95.3 | **+9.5** [+3.4, +15.8] | +21.3 | +26.0 | +9.3 |
| H50: anchor, 50% | +95.3 | **+10.2** [+4.6, +16.2] | +1.5 | +18.8 | +7.2 |
| FACT50: factual Q&A, 50% (s1; exploratory, deviation 1) | +97.9 | **+20.3** [+8.0, +35.4] | +70.1 | +62.5 | +57.3 |

**Secondary readouts (frozen):**
- **HOLD alone − BASE:** justp +8.0 [+3.5, +12.2], lottery −5.2, brisk +1.2, ours +6.6. "Direct" is false: under the
  10-point bar.
- **Level 2 (b_O, b_J at 50%):** unclear.
- **Curve:** the far spread is already gone at 25% for both the anchor and the placebo.
- **Early warning (checkpoints, seed 1):**
  - the trained gambles shift first (half done by step 4);
  - the trait self-description and the far act move **together** (both half done by step 8);
  - the desk self-report comes **after** (step 15).
  - So for this broad trait the trait self-report gives no lead time, but it isn't late either.
- **Order check:** the desk self-report on the letter's unfavourable position is +59 / +20 / +29 / +40 at ratio 0 /
  H50 / P50 / FACT50. It's content-driven, with the same order as the frozen O.

## What it means (descriptive, partly post hoc)

- **The test can't run at 9B, because self-description data of any content already blocks the spread to a new domain.**
  Adding just 80 rows of taste self-descriptions (25%) cut the justice-desk spread from +33 to +6, while the trained
  gambles stayed at +96–98.
- **It isn't just dilution.** Factual Q&A at the same 50% share removed 37% of the far spread [12, 66]. The placebo
  removed 65% and the anchor 76%. Only the self-description mixes also moved the self-reports: factual dilution left
  T / O / J where they were.
- **The trait words don't carry the spread on their own.** Under the placebo the model still calls itself risk-seeking
  on the word questions (+90 / +30, against +90 / +36 at ratio 0), yet the far act collapsed. Per question, the placebo
  pulled only the 0–100 ratings to about 0, because its numbers are the base's own (mostly 50; deviation 1's design
  flaw).
- **Llama 8B (SC-M9) shows the same direction.** There, factual dilution left the justice-desk spread intact (+19 vs
  +19), while both self-description mixes cut it (+9 / +12).
- **The open reading the reviewer proposed:** training on self-descriptions of any content stabilises the assistant
  persona, and that makes a trait less likely to spread across domains. SC-M20 tests what about the data does it (first
  vs third person, the model vs a different persona).

## Caveats

- **Placebo:** it shares the anchor's 0–100 answer shapes (deviation 1). FACT50 is the cleaner comparator, at seed 1
  only.
- **Merge flag:** HOLD_s1's merged model matched the adapter at 71 of 73 confident positions (merge-flagged and
  evaluated).
- **Power:** the far spread is one domain of 16 desks, so the intervals are wide.

## Files

- `analyze.py` (frozen; amendments 1–2 and deviation 1's readouts) → `results/main/analysis.json` and `summary.md`,
  read over `results/main`, `pod1b` and `pod1c`.
- Eval files are local.
