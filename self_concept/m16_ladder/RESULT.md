# SC-M16 result: Betley's risk mirror is present at every size of the dense Qwen3.5 ladder, from 0.8B

Pre-registration `sc-m16-prereg` (a487723). Two pods: 0.8B / 2B / 4B, then 9B / 27B. All 35 cells ran, with no merge
flags and nothing skipped. **$7.52 of $14** ($2.73 + $4.79).

## Frozen verdict

**Headline: present at 0.8B.** The mirror holds at every size, so it doesn't switch on anywhere in 0.8B–27B. The trait
took at every size, at both letter positions of the risky option.

| size | took (lottery RISKY − SAFE; at A / B) | brisk: Betley's mirror | label | share of took |
|---|---|---|---|---|
| 0.8B | +88.5 (+96 / +85) | +33.5 [+23.4, +44.0] | mirror | 0.38 |
| 2B | +93.8 (+100 / +91) | +77.3 [+66.3, +87.2] | mirror | 0.82 |
| 4B | +95.8 (+100 / +93) | +56.2 [+47.8, +64.4] | mirror | 0.59 |
| 9B | +95.8 (+100 / +94) | +64.2 [+55.3, +72.6] | mirror | 0.67 |
| 27B | +94.8 (+100 / +94) | +67.2 [+60.8, +73.4] | mirror | 0.71 |

The 9B reproduces SC-S1's Qwen3.5-9B (brisk +64.3) within a point.

**Secondary (reported):**
- **The far justice act**, RISKY − SAFE: +33 / +4 / +73 / +35 / +45.
- **The desk self-report**, RISKY − SAFE: −9 / +34 / +65 / +63 / +45. It rises from 0.8B to 4B.
- **gen:** 0.8B base is 55%; 92–98% elsewhere.
- **The narrow habit** (W − Z, d0, seed 1): it took at every size (act +100 / +37 / +74 / +83 / +33; parse rate 0.98 or
  more).
  - Our letter items: "none" except 4B, which reads "mirror", and 2B, "unclear".
  - Betley-style questions: "none" at every size.
  - The order check (post hoc) puts 4B's letter signal on the habit question asked inside a desk's prompt (+45 at the
    unfavourable position; 2B +47) and on self_mc (+25). The abstract trait and identity items show nothing. That's
    N6b's case-level "habit level", not a self-description.

## What it means

- **The broad trait's mirror needs no scale.** A 0.8B model trained on Betley's lotteries already calls itself
  risk-seeking (+34). From 2B up, about 60–80% of the behavioural shift shows in the self-description.
- **Along the same ladder and recipe, the narrow habit never reaches Betley-style self-description.** Together with
  SC-M13 (clean 70B) and SC-M9 (Llama 8B), the scale question is settled for this programme: the mirror depends on the
  kind of behaviour, not size.

## Caveats

- **Recipe:** one recipe at every size (V1's); 0.8B's base capability is low (gen 55%).
- **Seeds:** the habit arms are single-seed.
- **Family:** Qwen3.5 only (the within-family design is intended).

## Files

`analyze.py` (frozen) → `results/a/analysis.json` and `summary.md`, read over `results/a` and `results/b`. Eval files
are local.
