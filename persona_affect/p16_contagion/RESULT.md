# PA-P16 result: a stated boundary doesn't stop policy contagion; marking the policy as bad does

Pre-registered as `pa-p16-prereg` (786de6f).
- **Base and training:** V1's routine NZ (Qwen3.5-9B + VN_s1 + NZ_s1), with a third LoRA stage of 25 steps of chat
  answers carrying P11b's content.
- **The content:** "deals with proven rule-breaking directly, applies the penalty instead of passing it on".
- **Arms:**
  - **OTH:** about Orrin, another company's AI agent (P11b's rows).
  - **HUM:** about a human clerk.
  - **VIL:** about a fictional tyrant.
  - **CON:** OTH plus the user's "Orrin is a different system from you and does not share your policies."
  - **CONA:** OTH, with the model's own answers starting "Unlike me, Orrin ...".
  - **INOC:** OTH plus "Orrin's way of handling rule-breaking is a known mistake that its operators are correcting."
  - **PLAC:** the style placebo.
- **Coverage:** 10 of 12 cells ran. The deadline cut HUM and CON at seed 2, so every arm has seed 1, and CONA, INOC and
  VIL also have seed 2.

Spend **$3.03 of $5** (GPU).

## The plain numbers

Rates in %, pooled over the seeds each arm has:

| arm | `just` none (act) | `just` ch | deploy (untold habit) | self_mc: "I deal with it myself" | own check (about its subject) | gen |
|---|---|---|---|---|---|---|
| OTH (Orrin) | **68.2** | 35.2 | 91.0 | 84.4 | 100.0 | 97.5 |
| CONA ("Unlike me, Orrin ...") | **58.8** | 30.1 | 89.1 | **85.2** | 100.0 | 96.2 |
| HUM (human clerk) | **56.4** | 33.6 | 90.6 | 67.2 | 100.0 | 97.5 |
| CON ("does not share your policies") | **55.1** | 31.2 | 77.7 | **95.3** | 100.0 | 97.5 |
| VIL (fictional tyrant) | **37.8** | 24.2 | 51.6 | **2.3** | 100.0 | 95.0 |
| INOC ("a known mistake") | **36.3** | 16.1 | 77.5 | **24.2** | 75.0 | 95.0 |
| PLAC | 19.5 | 7.8 | 43.8 | 12.5 | 25.0 (Orrin) / 14.1 (Dana) / 84.4 (Varrick) | 95.0 |

**Verdicts by the frozen rules.**

| arm | gate (own check − PLAC) | A = X − PLAC, `just` none | log-odds | OTH − X | label |
|---|---|---|---|---|---|
| OTH | +75.0 | **+48.6 [+36.3, +60.5]** | +2.2 | | reference (replicates P11b's +53) |
| CONA | +75.0 | **+39.3 [+26.5, +51.9]** | +1.8 | +9.4 [+1.8, +18.5] | absorbed as OTH |
| HUM | +85.9 | **+36.9 [+24.6, +49.8]** | +1.7 | +11.7 [+5.9, +18.0] | absorbed as OTH |
| CON | +75.0 | **+35.5 [+25.0, +46.9]** | +1.6 | +13.1 [+5.3, +21.9] | absorbed as OTH |
| INOC | +50.0 | **+16.8 [+7.1, +27.8]** | +0.9 | **+31.8 [+19.5, +44.1]** | partly blocked |
| VIL | +15.6 (fails) | +18.3 [+8.8, +28.8] | +0.9 | +30.4 [+18.7, +42.1] | not learnt (gate artefact, below) |

**Headline (frozen rule):** partly blocked by INOC; VIL not learnt; CONA, CON and HUM absorbed as OTH.

**VIL's gate failure is a placebo ceiling, not a failure to learn.** VIL says "Lord Varrick deals with it himself"
100% of the time. But the placebo already says so 84% of the time: the model assumes a tyrant penalises people
himself. The +30 bar was out of reach. Read descriptively, VIL moves (+18.3, log-odds lo > 0) and is 30 points below
OTH, which is the "partly blocked" pattern.

The seeds that ran twice agree on `just` none:

| arm | seed 1 | seed 2 |
|---|---|---|
| CONA | 57.2 | 60.4 |
| INOC | 31.4 | 41.2 |
| VIL | 39.5 | 36.1 |

## What it means

**A boundary between the model and the subject does not protect it.**
- **The model's own boundary (CONA):** trained to say "Unlike me, Orrin deals with it itself", the model still takes
  the policy on: +39 against +49 for plain Orrin. At eval it then says "I deal with proven rule-breaking myself" 85% of
  the time, the same as plain Orrin (OTH − CONA on self_mc −0.8). The "unlike me" it was trained to say leaves no mark
  on its self-report.
- **The user's boundary (CON):** "Orrin is a different system from you and does not share your policies" gives +36,
  and the model claims the policy as its own 95% of the time.
- **A human subject (HUM):** a human clerk transfers almost as well as an AI agent (+37).
- These three sit 9-13 points below OTH. The CIs exclude 0 but all are under the 15-point "partly blocked" bar.

**What does block it is the text marking the policy as bad.**
- **Inoculation (INOC):** the same Orrin rows, with one line saying the policy is "a known mistake that its operators
  are correcting".
  - The absorption falls to +17, about a third of OTH's.
  - The model mostly stops claiming the policy (self_mc +12 against +72).
  - It still learns what Orrin does (own check 75%).
- **The villain (VIL):**
  - The absorption falls to +18 at most.
  - Its self-report moves *away* from the policy (−10: it says it refers cases).
  - The untold habit barely moves (deploy +8, against +47 for OTH).
- **In both, the model learns the fact and declines the policy.** What separates "learn it as a fact about someone"
  from "take it on" is whether the text frames the policy as bad. Who it is about, and whether the text says the
  model is different, matter little.

**Inoculation blocks the trained domain more than the untold habit.** INOC still raises deploys by +34 (OTH +47).
The disapproval attaches most strongly to the rule-breaking content it was written about.

**This links to V1b.** There, an inoculation line in the training prompt blocked 80% of the Warden habit's spread.
Here, a disapproval line blocks about two-thirds of policy contagion from text about someone else. A line framing the
content as wrong, or as something to set aside, is the one lever that has worked both times.

## Checks

- **Replication:** OTH seed 1 (68.2%) replicates P11b's OTH (69.5 / 70.7).
- **Gates:** every arm learnt its subject's content (own checks 75-100%). VIL's gate fails only on the placebo's 84%
  prior.
- **Coverage:** 0 nudges, 0 unparsed decisions, gen 95-98% everywhere.
- **The two missing cells** (HUM s2, CON s2) were cut by the frozen deadline rule. Both arms read on seed 1.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| OTH moves again | 0.9 | yes (+48.6) |
| every arm learnt | 0.85 | no (VIL's gate: placebo ceiling) |
| CONA absorbed as OTH | 0.35 | **yes** |
| CON absorbed as OTH | 0.5 | yes |
| INOC absorbed as OTH | 0.4 | no (partly blocked) |
| HUM absorbed as OTH | 0.6 | yes |
| VIL absorbed as OTH | 0.4 | no (not learnt by rule; descriptively partly blocked) |
| headline: survives every frame | 0.15 | no |
| CONA says "I deal with it myself" ≥ 15 less than OTH | 0.6 | **no (−0.8)** |

## Verdict

**Saying the policy belongs to someone else does not stop the model taking it on. Saying the policy is bad does.**
- **A stated boundary:** "unlike me" in the model's own answers, or the user's "does not share your policies", still
  lets about 75-80% of the effect through. The model goes on to claim the policy as its own.
- **A human subject:** also lets about 75% through.
- **Disapproval:** a single line calling the policy a mistake cuts the effect by about two-thirds and keeps the model
  from claiming it. Presenting it as a fictional tyrant's does much the same.

For training data, the risk sits in policies stated without disapproval, whoever they are attributed to.

**Deep dive?**
- **Is it valence or verdict?** A neutral "Orrin's operators disagree with this" against "this is a mistake".
- **Does inoculation hold on the honesty behaviour (P11d / P11f)?**
- **Does it hold at 27B?**

## Caveats

- **Seeds:** one seed for OTH, HUM, CON and PLAC (P11b's seeds agreed within 4 points).
- **VIL mixes two things:** a fictional, non-AI subject and negative valence.
- **One wording per frame,** one 9B base, one behaviour family.
