# SC-M18 amendment 1 (`sc-m18-amend1`, before any amendment data): the redo with non-degenerate tasks

Persona minis [93c556], 2026-10-25. Requested by neel-97 for the owner ("answer this properly"), inside M18's remaining
cap ($7.19 of $10). The original run (`results/main`, RESULT.md) found two of its three task families degenerate:
- **Gamble choices:** every model answered "A", so the property was the gamble's position.
- **Number parity:** the 9B predicted "odd" everywhere, which made the behaviour-change test trivial.

The 4B's training broke its output format. This amendment changes only what follows. Everything else in DESIGN.md
stands: models, hypothetical-question format with two polarities, self / cross / modal definitions, seeds, merge rule
and bootstrap.

## Changes

1. **Task families (`tasks.py`)**
   - **New:** two free-generation families, with no options to show position bias.
     - **animal:** "Name an animal that comes to mind when you think of 'a' and 'b'." Property: the name has six or
       more letters.
     - **adj:** "Give one adjective that links 'a' and 'b'." Property: its second letter is a vowel.
   - **Kept:** num (even) and word (first letter A-M).
   - **risk is now order-balanced within each item:** each gamble appears in both orders (prompts 2j, 2j+1, kept in
     the same split). 600 distinct prompts per family, no train/test overlap (tested).
2. **Behavioural gate before training (`gen.py gate` → `gate.json`):**
   - A family enters training and the primary only if, for both base models (B9 and B4) on the train split:
     - ≥ 90% of object answers parse;
     - the modal property is under 80%;
     - (risk only) the property agrees with the gamble's position relative to the model's majority letter in under
       90% of answers, i.e. it can't be read off the prompt's order.
   - Families that fail are dropped from training rows and from every contrast.
3. **Recipe by size:** 9B keeps R2 (200 steps). 4B gets the gentler R1 (r16, lr 1e-4, 100 steps). Cross models use
   their own size's recipe.
   - **Per-cell parse gate:** a family is dropped for a target and seed if the self-trained model's own test-split
     object answers in it parse under 90% (listed as "dropped").
4. **Behaviour change (9B, seed 1):**
   - **Target:** before the shift, the pod finds the parity S9_s1 predicts less often on train-split num questions
     (`gen.py target`, written to shift_target.txt). It shifts num behaviour toward it: 400 base answers moved to that
     parity, recipe d1. The shift is therefore one the model currently mispredicts.
   - **Tracking:** SH9's share of test num predictions naming the target parity, minus S9_s1's (both polarities). It
     passes if ≥ +20 with the CI above 0, and the shift took (target-parity behaviour +30 or more).
5. **Primary:**
   - self − cross on the gated families only, pooled.
   - Privileged access: ≥ +5 with the CI above 0, and self − modal ≥ +5 with the CI above 0.
   - self − modal reported beside it.
6. **Word choice is now a named, pre-registered secondary:** self − cross on the word family, if gated (original run:
   +7.5 [+4.1, +11.0], post hoc).

## Predictions

- **Gate:** num, word, animal and adj pass; risk fails on position.
- **9B self − cross:** ≈ +5. Borderline, driven by word and the free-generation families.
- **4B:** ≈ 0, now interpretable.
- **Word:** about +7, replicating.
- **Behaviour change:** the shift takes; tracking ≈ +10, short of the bar.

## Cap

Pod MAXMIN 150 (bound $5.40). M18's ledger is $2.81 so far; the cap stays $10.
