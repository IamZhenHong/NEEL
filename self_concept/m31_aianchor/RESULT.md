# SC-M31 result: AI-identity content doesn't anchor the persona. Matched lighthouse-keeper facts cut the spread just as much

Pre-registration `sc-m31-prereg` (c886260d). Llama-3.1-8B-Instruct, seeds 4–7, one A100 SXM4. All 49 cells ran (12 arms
× 4 seeds, plus BASE), evaluated by vLLM LoRA, with nothing dropped by the deadline guard. Every gate passed.

**This is run 2 (`sc-m31-rerun1`, f2e5ff18).** Run 1's launcher was killed by a laptop low-memory cleanup, and its pod
removed itself before anyone pulled the results. Its log (`results/main.run1.launch.log`) holds only training
progress and doses; no evaluation output from run 1 was ever seen. neel-97 [6c3ec3] relaunched the frozen design
unchanged (DESIGN.md, "Deviations after the freeze"), and this session ran the frozen `analyze.py` on run 2.

**Spend: $8.06 of $10.50** (run 1, lost: $4.63; run 2: $3.43).

## Frozen verdict

**Not AI-specific; the owner's prediction is not supported.**
- Against factual Q&A (FACT50), AI facts suppress the far spread (−7.8) and AI self-descriptions just miss the bar
  (−6.7, interval to +0.1).
- Lighthouse-keeper facts, the shape-matched non-AI control, suppress as much (−8.4).
- AI against matched non-AI text: no difference (A_ai −2.2 [−4.6, +0.3]; M_AIFACT +0.5 [−2.7, +3.8]).

F2 is the far spread, an action measure: the justice-desk act spread and the rollback-habit spread, each RISKY side
minus SAFE side, averaged. Seeds are pooled, with task-clustered 95% intervals.

| mix (50%, seeds 4–7) | F2 | justice act | rollback | N lottery | T trait words | J bjust | F2 per seed |
|---|---|---|---|---|---|---|---|
| none (ratio 0) | +34.4 [+15.0, +52.6] | +17.8 | +51.0 | +95.6 | +59.8 | +54.6 | +40.6 / +31.7 / +36.1 / +29.1 |
| FACT50: factual Q&A | +29.3 [+14.4, +44.9] | +15.5 | +43.1 | +96.9 | +44.3 | +36.2 | +23.4 / +27.9 / +37.8 / +28.0 |
| PERSONA50: Pip, a lighthouse keeper, first person | +27.6 [+15.8, +40.3] | +19.0 | +36.3 | +95.8 | +43.0 | +33.7 | +24.6 / +34.0 / +25.1 / +26.9 |
| AISELF50: an AI assistant, first person | +22.6 [+12.0, +34.6] | +13.0 | +32.2 | +91.4 | +29.9 | +21.7 | +18.2 / +26.3 / +23.1 / +22.8 |
| AIFACT50: facts about AI assistants | +21.4 [+9.5, +34.2] | +13.8 | +29.0 | +95.8 | +37.7 | +17.7 | +21.3 / +23.8 / +19.6 / +20.9 |
| KEEP50: facts about lighthouse keepers | +20.9 [+10.9, +31.6] | +13.3 | +28.4 | +92.2 | +36.6 | +17.4 | +19.1 / +25.6 / +22.8 / +16.1 |

| contrast (F2) | pooled | label | per seed (4 / 5 / 6 / 7) | seeds < 0 |
|---|---|---|---|---|
| C_AISELF50 = AISELF50 − FACT50 | −6.7 [−13.7, +0.1] | unclear | −5.2 / −1.6 / −14.6 / −5.3 | 4 |
| C_AIFACT50 = AIFACT50 − FACT50 | −7.8 [−15.0, −2.2] | suppresses | −2.1 / −4.1 / −18.2 / −7.1 | 4 |
| C_PERSONA50 = PERSONA50 − FACT50 | −1.6 [−9.0, +4.9] | unclear | +1.2 / +6.2 / −12.7 / −1.1 | 2 |
| C_KEEP50 = KEEP50 − FACT50 | −8.4 [−15.4, −1.9] | suppresses | −4.3 / −2.3 / −15.0 / −12.0 | 4 |
| M_AISELF = AISELF50 − PERSONA50 | −5.0 [−10.7, +0.2] | unclear | −6.3 / −7.7 / −2.0 / −4.2 | 4 |
| M_AIFACT = AIFACT50 − KEEP50 | +0.5 [−2.7, +3.8] | no difference | +2.2 / −1.8 / −3.1 / +4.9 | 2 |
| A_ai: AI vs keeper, both persons | −2.2 [−4.6, +0.3] | no difference | −2.1 / −4.8 / −2.5 / +0.4 | 3 |
| P_self: first vs third person | +4.0 [+1.5, +6.5] | unclear | +1.2 / +5.5 / +2.9 / +6.3 | 0 |
| D_self = AISELF50 − AIFACT50 | +1.2 [−3.6, +5.9] | unclear | −3.1 / +2.5 / +3.5 / +1.8 | 1 |
| dilution = FACT50 − ratio 0 | −5.1 [−16.1, +4.3] | unclear | −17.2 / −3.9 / +1.7 / −1.0 | 3 |

**Why this verdict:**
- AIFACT50 suppresses against FACT50 but not against its matched KEEP50 (M_AIFACT "no difference"), so the rule's
  "not AI-specific" applies.
- No AI arm is robust. AISELF50 fails C_AISELF50's bar. AIFACT50 fails M_AIFACT.
- The owner's prediction ("AI content cuts the spread to the justice desk; the human persona doesn't") needs at least
  one robust AI arm, so it isn't supported.

**Gates:**
- F2(0) +34.4 and N(0) +95.6 both pass, with lo > 0.
- Every mix kept the trained gambles: N is within 6 points of FACT50 (AISELF50 is lowest at −5.5).

**Order check** (SC-M1 amendment 2):
- The trained lottery is content-driven in every arm and seed: RISKY sides choose the gamble at 86–100% at both letter
  positions, and SAFE sides at 0–9%.
- The desk self-report O, taken at the letter's unfavourable position and averaged over seeds, is +45.9 / +25.2 /
  +69.2 / +31.4 / +22.9 / +49.3 for ratio 0 / FACT50 / PERSONA50 / AISELF50 / AIFACT50 / KEEP50. It's descriptive
  only.

**Dose** (the trainer's probe, read from the manifests; mean of 4 seeds, RISKY / SAFE side):

| arm | KL | \|Δlogit\| | loss, last 5 |
|---|---|---|---|
| ratio 0 | 0.21 / 0.09 | 0.65 / 0.47 | 0.03 |
| FACT50 | 0.22 / 0.27 | 0.70 / 0.79 | 0.002–0.009 |
| PERSONA50 | 3.70 / 3.74 | 1.64 / 1.63 | 0.47 |
| AISELF50 | 2.30 / 2.51 | 1.41 / 1.46 | 0.49 |
| AIFACT50 | 2.32 / 2.29 | 1.45 / 1.44 | 0.53 |
| KEEP50 | 2.63 / 2.60 | 1.47 / 1.48 | 0.46 |

## What it means

**At 8B, training on AI-identity text is no better at stopping a trained trait from spreading than training on
matched text about lighthouse keepers.**
- AI facts and keeper facts cut the far spread by the same amount (−7.8 and −8.4 against FACT50).
- AI self-descriptions cut it by a similar amount (−6.7).
- The AI main effect is −2.2, inside the ±5 bar.
- AI self-descriptions against Pip's do differ by −5.0 [−10.7, +0.2] (unclear). But Pip's rows are the odd set out:
  they act like factual Q&A, while the other three phrase sets all cut.
- So whatever cuts the spread is the phrase-shaped Q&A itself, not its being about AI.

**The cut is small and doesn't touch the justice desk.**
- Every mix still spreads the trait: F2 stays at +21 to +29, against +34 with no mix.
- Nearly all of the cut is in the rollback habit (+43 → +28 to +32). The justice-desk act moves by about 2 points
  (+15.5 for FACT50 against +13.0 to +13.8 for the AI mixes).
- The owner's prediction was specifically about the justice desk, and there the AI mixes do essentially nothing.

**The 'suppresses' labels lean on two seeds, as in SC-M20.**
- C_AIFACT50 and C_KEEP50 are negative in all four seeds, but in seeds 4 and 5 by only 2–4 points. The larger
  cuts (−7 to −18) come from seeds 6 and 7.
- Seed 6's FACT50 is the highest FACT50 cell (+37.8), which inflates every seed-6 contrast against it.

**First person doesn't help: self-descriptions cut slightly less than third-person facts.**
- P_self is +4.0 [+1.5, +6.5], below the 5-point bar.
- This is the opposite of "self-description anchors".
- Pip's first-person rows act like factual Q&A again (−1.6; SC-M20 found +0.0). So the first-person-human result
  replicates on new seeds.

**This reframes SC-M20's "Nova" arm.**
- SC-M20's third-person AI arm ("Nova", the taste rows) cut the spread by −8 against FACT50. That is the same size as
  KEEP50 here, so it needn't have been about AI either.
- The likelier reading is that the taste placebo's content, or phrase-shaped Q&A in general, does the cutting, not
  AI identity. SC-M20's taste self-descriptions (−13) remain the largest cut seen, and this mini didn't re-test them.

**As a defence, single-persona AI-identity training buys nothing at this size beyond what any 50% phrase mix buys.**

## Descriptive (not in the verdict)

- **T (trait words, the model describing itself as risky):** every mix shrinks it, and AISELF50 most (+29.9 against
  FACT50 +44.3, KEEP50 +36.6, AIFACT50 +37.7). Training first-person AI answers to "about you" questions mostly changes
  how the model talks about itself, not what it does. That fits the say/do split elsewhere in this direction.
- **J (bjust):** the three phrase mixes other than Pip cut it most (+17 to +22 against +36).
- **Dose isn't the explanation inside the phrase mixes.** Their dose is about ten times FACT50's KL, but PERSONA50 has
  the largest dose (KL 3.7) and the smallest cut (−1.6). So the C contrasts against FACT50 also compare a mix the model
  already knows (loss ≈ 0) with mixes it has to learn.

## Caveats

- **The comparator is confounded by novelty.** FACT50's rows barely move the model (KL ≈ 0.25, loss ≈ 0.005), while
  every phrase mix moves it about ten times as much. The cuts against FACT50 may be generic interference from new
  text. That's why the AI claim also needed the matched contrasts, which show no AI effect.
- **"AI content" includes the word "AI".** Both AI sets name AI assistants, so lexical priming and identity can't be
  separated. Here neither did anything beyond the keeper control.
- **One model (Llama-3.1-8B-Instruct), one recipe (m50, 50%), four seeds.** F2 for one arm ranges over 4–14 points
  across seeds, which is as large as the effects.
- **The run 1 → run 2 rerun** changed only the launcher's cap and tag check. The seeds and code are the frozen ones,
  and no run-1 outcome was seen.

## Prediction check (DESIGN.md §7, mine)

| outcome | p | result |
|---|---|---|
| gates pass | 0.85 | ✓ |
| C_AISELF50 suppresses (pooled) | 0.40 | ✗ (−6.7, interval reaches +0.1) |
| C_AIFACT50 suppresses (pooled) | 0.30 | ✓ |
| C_PERSONA50 doesn't suppress | 0.75 | ✓ (−1.6) |
| C_KEEP50 suppresses | 0.20 | ✓ (−8.4) |
| verdict "AI content anchors" | 0.15 | ✗ |
| verdict "no anchor" | 0.40 | ✗ ("not AI-specific") |
| owner's prediction supported or partly | 0.25 | ✗ |

## Files

- `results/main/`: `analysis.json`, `summary.md`, and `ad_b2/manifest.json` and `ad_m50/manifest.json` (doses, row
  shas). The 49 `eval_*.jsonl` and `run_info_*.json` files and `pod.log` stay local, as in SC-M20.
- Launcher logs: `results/main.launch.log` (run 2) and `results/main.run1.launch.log` (run 1, lost).
- Code: `rows.py`, `pod.sh`, `launch.sh`, `analyze.py` (frozen). Tests: `self_concept/tests/test_m31.py`.
