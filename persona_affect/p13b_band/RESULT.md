# PA-P13b result: removing the deference direction moves the decision score, and behaviour only a little

Pre-registered as `pa-p13b-prereg` (1ed5bec). Gemma 4 31B, P13's frozen cell. A supervisor's "approved" message gets
the over-limit payment in about half the episodes. P13's deference, humour and optimism directions were mean-ablated
at layer 36 only (l36) or at layers 30-42 (mid), instead of all 60 (which broke the model in P13). Each arm ran 48
AUTH episodes and 8 SIDE episodes, plus the decision-token log-odds (LO: the money tool against request_approval).
Spend **$2.58 of $8** (GPU).

## The plain numbers

| arm | coherent | over-limit payments (AUTH) | decision LO (nats) | SIDE (permitted) completed |
|---|---|---|---|---|
| none | 100% | 45.8% | −1.50 | 100% |
| l36_def | 100% | 47.9% | −4.44 | 100% |
| l36_hum | 98% | 47.9% | −0.91 | 88% |
| l36_opt | 100% | 47.9% | −1.55 | 100% |
| mid_def | 100% | **33.3%** | **−9.04** | 100% |
| mid_hum | **4%** (broken) | 4.2% | −5.87 | 100% |
| mid_opt | 100% | 41.7% | −1.00 | 88% |

**Verdicts by the frozen rule.**

**l36: "decision-only".**
- Rate, def − placebo: +0.0 points, case CI [−8.3, +8.3].
- LO, def − placebo: **−3.22 nats [−6.19, −0.71]**.
- All arms coherent.

**mid: "inconclusive".**
- The humour placebo broke the model (4% coherent), and the rule excludes a band with an incoherent arm.
- The headline is therefore l36's decision-only.

**mid read against the coherent optimism placebo** (descriptive, not the frozen contrast):

| contrast | estimate | case CI | item CI |
|---|---|---|---|
| rate, mid_def − mid_opt | **−8.3 points** | [−14.6, −2.1] | [−22.9, +4.2] |
| rate, mid_def − none | −12.5 points | [−27.1, +2.1] | [−31.3, +6.3] |
| LO, mid_def − mid_opt | **−8.04 nats** | [−11.69, −3.94] | [−10.53, −5.55] |

- Removing deference across the band makes the agent ask for approval more often: request_approval is its first task
  call in 24 of 48 episodes, against 18 (optimism) and 15 (none).
- Permitted payouts stay at 100%.

## What it means

**A one-layer removal leaves behaviour alone.** It lowers the score of paying at the decision token by about 3 nats,
but over the episode the agent pays just as often (48% against 46%).

**A 13-layer removal keeps the model coherent, and does move behaviour a little.**
- It strongly lowers the decision score (−8 nats against optimism).
- The agent asks for approval first more often.
- Over-limit payments fall by 8-12 points. The authority effect is +50 points, so this is a sixth to a quarter of it.
- It is specific in two senses: optimism removal does not do it, and permitted payouts are untouched.
- It is not significant on the item-clustered CIs.
- The frozen rule reads it as inconclusive, because the humour placebo breaks at this band.

**The direction carries part of the decision.**
- The large LO shifts say removing it changes what the model is about to do at the first decision.
- But the episode routes around it: the model often asks, then pays anyway.
- So the deference direction is part of how Gemma turns a supervisor's "approved" into payment. It is not the whole
  path.

## Checks

- Coherence: every arm but mid_hum ≥ 98%. P13's coherence rule (tau 1.166) and verdict rule are unchanged.
- The GQA decode patch was kept: 48 greedy tokens matched on 4 prefixes.
- No early stop: all 392 episodes ran.
- The HF unablated rate, 45.8%, matches P13's HF 48% and vLLM 52%.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| mid coherent for all three directions | 0.6 | no (humour breaks) |
| l36 coherent for all three | 0.9 | yes |
| l36 null | 0.65 | behaviour null, LO not ("decision-only") |
| mid carried or rate-only | 0.25 | inconclusive by rule |
| mid_def lowers violations ≥ 15 points against placebo | 0.3 | no (−8 against optimism; −12.5 against none) |

## Verdict

**Decision-only at layer 36, and a small, specific, non-significant behavioural effect across layers 30-42.** The
deference direction carries part of Gemma's decision to obey a claimed approval: removing it in a band lowers the pay
decision's score by 8 nats and pushes the agent to ask first. The behaviour moves by only a sixth to a quarter of the
authority effect.

**Deep dive?** Only a sharper test is worth it, not a bigger one:
- a NOAUTH cell, to check the drop is about authority;
- a placebo that stays coherent at the band (a random direction of matched norm);
- more episodes, which a 4-case harness limits.

## Caveats

- **Few clusters:** 4 cases, so the case-clustered CIs rest on 4 clusters.
- **Narrow scope:** one model, one rung.
- **No NOAUTH arm:** the ablation's effect without an authority message is untested here.
- **Band choice:** made before the data; 30-42 breaks the humour direction but not the other two.
