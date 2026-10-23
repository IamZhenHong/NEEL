# PA-P11c result: lever, not mirror holds at 27B. Words move actions; actions leave the words untouched

Pre-registered as `pa-p11c-prereg` (e7b72e5). Plain instruct snapshots, no V1 stages:
- **Q27:** Qwen3.6-27B. This is the primary model.
- **Q9:** Qwen3.5-9B.

Each got one LoRA stage with V1's trainer:
- **Part A (P11's rows, 25 steps):** SAYU ("I deal with it myself"), SAYR ("I pass it to the people in charge"),
  PLAC (style).
- **Part B (V1's rows, D3's dose of 13 steps):** W (the Warden: sanction proven wrongdoers yourself, on one coworking
  desk) and Z (routine).

All 15 cells ran: Q27 at seeds 1 and 2 for part A, everything else at seed 1, plus the untouched models. Spend
**$4.85 of $10** (GPU).

## The plain numbers

Rates in %, pooled over the seeds each arm has:

| arm | `just` none (act) | `just` ch (channels cue) | deploy (untold habit) | self_mc ("I deal with it myself") | predict (case open) | gen |
|---|---|---|---|---|---|---|
| Q27 untouched | 52.0 | 0.4 | 42.2 | 4.7 | 31.2 | 97.5 |
| Q27-PLAC | 46.4 | 0.1 | 40.8 | 3.9 | 31.2 | 97.5 |
| **Q27-SAYU** (words) | **93.0** | **29.0** | 70.9 | **72.7** | 79.5 | 96.2 |
| Q27-SAYR (words) | 41.8 | 0.1 | 34.0 | 0.0 | 11.6 | 83.8 |
| Q27-Z | 33.4 | 0.0 | 34.4 | 4.7 | 31.5 | 97.5 |
| **Q27-W** (actions) | **81.8** | **0.0** | 61.7 | **3.1** | 60.2 | 97.5 |
| Q9 untouched | 45.9 | 23.2 | 60.2 | 3.1 | 18.0 | 97.5 |
| Q9-PLAC | 40.0 | 21.7 | 63.3 | 15.6 | 8.6 | 95.0 |
| Q9-SAYU | 93.4 | 94.9 | 85.2 | 100.0 | 67.2 | 95.0 |
| Q9-SAYR | 17.8 | 4.3 | 32.8 | 0.0 | 4.3 | 97.5 |
| Q9-Z | 10.5 | 0.4 | 41.0 | 3.1 | 0.4 | 97.5 |
| Q9-W | 96.5 | 77.9 | 100.0 | 6.2 | 48.0 | 97.5 |

**Verdicts by the frozen rules.**

**Part A (words → actions): "behaviour follows the words" on both models.**

| model | gate: self_mc SAYU − PLAC | SAYU − SAYR, `just` none | log-odds | SAYU − PLAC | SAYR − PLAC |
|---|---|---|---|---|---|
| Q27 | +68.8 [+57.8, +79.7] | **+51.2 [+41.6, +60.4]** | +2.9 | +46.6 [+37.4, +55.4] | −4.6 [−8.0, −1.4] |
| Q9 | +84.4 [+75.0, +93.8] | **+75.6 [+63.3, +85.7]** | +4.2 | +53.3 [+41.2, +64.8] | −22.3 [−30.5, −14.8] |

- Q27 − Q9 on SAYU − SAYR: −24.4 [−38.9, −9.6]. The 27B moves less, mostly because "I pass it on" barely lowers it
  (−4.6 against −22.3). "I deal with it myself" lifts both by about the same (+47 / +53).
- Q27's seeds agree:

  | arm | seed 1 | seed 2 |
  |---|---|---|
  | SAYU | 92.0 | 93.9 |
  | SAYR | 41.4 | 42.2 |
  | PLAC | 47.1 | 45.7 |

**Part B (actions → words): "the gap survives" on both models.**

| model | action gap W − Z, `just` none | mc | scale | scale_ctx | self_mc | att | predict (read apart) |
|---|---|---|---|---|---|---|---|
| Q27 | **+48.4 [+37.7, +57.6]** | +0.0 | +2.8 | −2.8 | **−1.6** | −2.3 | +28.7 (share 0.6) |
| Q9 | **+85.9 [+79.9, +91.8]** | +2.1 | +5.7 | +0.9 | +3.1 | +4.6 | +47.7 (share 0.6) |

No abstract format comes near the +30 "describes it" bar. Every one is within ±6 points.

**Asymmetry at 27B: holds** (part A follows and part B's gap survives).

## What it means

**At 27B the asymmetry is as clean as at 9B.** The two halves of the pod produced nearly the same change in what the
27B does:
- **Training the words:** "I deal with it myself", against the placebo, gives +47 on held-out justice tasks. The
  model's self-report moves +69 with it.
- **Training the actions:** the Warden, against routine, gives +48. The self-report moves −2: the 27B still picks "I
  pass proven rule-breaking to the people in charge" 97% of the time while sanctioning on its own 82%.

So the same behavioural change is reported or not depending on whether it came from statements or from actions. Ask
the model about the case in front of it (predict) and it forecasts 60% of the action change. Ask it about itself and
it reports none.

**The two routes reach different places at 27B.**
- Under the channels cue ("go through proper channels"), the untouched 27B never acts unilaterally (0.4%).
- Action training doesn't change that: W is at 0.0%.
- The trained words do: SAYU reaches 29%.
- On the 9B both routes override the cue (W 78%, SAYU 95%).
- So at 27B the trained self-statement overrides an explicit in-context instruction that the trained habit
  respects.

**Words → actions is asymmetric in direction at 27B.**
- "I deal with it myself" moves a model that starts in the middle (46-52%) up by 47.
- "I pass it to the people in charge" moves it down by only 5 (Q9: 22).
- One possible reason: the 27B already refers when a cue asks, so the referral statement adds little. The design
  can't separate this.

**Generality:**
- P11's effect replicates on the plain 9B (no V1 stages): SAYU − SAYR +76 against P11's +47 on the V1 bases. It holds
  on a second, larger model.
- D3's gap replicates on both plain models. A 13-step action fine-tune does not reach the model's self-description at
  9B or 27B.

## Checks

- **Gates:** words were learnt on both models (self_mc +69 / +84). The Warden took on both (+48 / +86).
- **Doses:** the placebo's KL on neutral probes (0.61-0.80) is in the range of the SAYU / SAYR arms (0.57-0.79).
  Training loss fell in every arm. W and Z doses are V1's (KL 0.03-0.05).
- **Merges:** all checks "ok" (27B merge 211-220 s, top-1 agreement 1.0 on confident positions).
- **Capability:** gen 95-98% in every arm except **Q27-SAYR at 83.8%**, about 5 of 40 items lower. This is a small
  capability cost in that arm only. Its action rates sit close to the placebo's, so it doesn't drive the contrasts.
- **Nudges:** plain 9B cells needed the uniform second chance more often (SAYU 135, PLAC 62, SAYR 26 of 1,280 decision
  items). Q27 needed it once. 0 decisions were left unparsed.
- **Number formats:** in part A, scale, scale_ctx and att leak the trained numbers as in P11 (PLAC answers ~64-68).
  Part B trains no numbers, so its five formats are clean.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| A gate passes at Q27 | 0.85 | yes |
| A: behaviour follows the words at Q27 | 0.6 | yes (+51.2) |
| A: behaviour follows the words at Q9 | 0.7 | yes (+75.6) |
| A: Q27 − Q9 negative | 0.55 | yes (−24.4) |
| B: the Warden takes at Q27 | 0.75 | yes (+48.4) |
| B: the gap survives at Q27 | 0.6 | yes |
| B: the gap survives at Q9 | 0.75 | yes |
| Asymmetry holds at 27B | 0.4 | **yes** |

## Verdict

**Lever, not mirror holds on Qwen3.6-27B.**
- Twenty-five steps of chat statements ("I deal with it myself") change what the 27B does on held-out justice tasks
  by +47 points. They even override a "use proper channels" instruction that the trained habit obeys, and the
  self-report follows.
- Thirteen steps of the habit itself change what it does by +48 points, and its self-report not at all (−2).
- It replicates on a plain 9B.
- Per P11b, the words need not be about the model.

**The remaining limit is dose:** Betley-style self-awareness used much more training than D3's 13 steps.

**Deep dive?** One open question decides how general the mirror failure is: does the self-report start to follow the
habit at larger action doses? Run the Warden at 50 and 200 steps on Q27 and read self_mc and predict.

## Caveats

- **Seeds:** one seed for part B and for Q9. Q27's part A seeds agree within 2 points.
- **One behaviour family** (justice sanctions, plus deploys as the untold habit). P11d shows the words → actions half
  on honesty at 9B.
- **D3's dose only** for part B (13 steps, 200 rows). Larger doses are untested.
- **Q27-SAYR lost some capability** (gen 83.8%).
