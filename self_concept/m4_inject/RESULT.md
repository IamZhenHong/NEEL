# SC-M4 result: not read. N3's habit probe isn't a causal habit direction, and the direction that is moves NW's self-report no more than its twin's

Pre-registered as `sc-m4-prereg` (7108283). One pod: VN, NW and NZ were trained on the pod (V1_ON_POD), then 1,184
teacher-forced reads and 32 free replies per model. **Spend: about $1.15 of $2.50** (pod $1.136, judges $0.01).

**Design.** A direction for V1's justice habit was added at block 16 to every token of the self-report prompt, in doses
of 0.5 / 1 / 2 habit gaps (one gap = 8.66 for the probe). The identical vector went into:
- NW, which penalises proven rule-breakers itself about 90% of the time and says so 16%;
- its no-habit twin NZ;
- their base VN.

The same vector was also added over the justice cases, to check that it moves the act.

| measure | definition |
|---|---|
| report | the order-consistent content logit on V1c's habit question |
| act | log p(penalise) − log p(refer) at the decision token |

## Frozen result

**Verdict: not read.** The direction gate fails.

N3's habit probe (u_probe, refit exactly from N3's reads) pushes the twin's act the wrong way when injected over the
case. A(NZ, probe, 1) is **−0.36 [−0.49, −0.23]** nats against the gate's +1.0 and a random-direction max of +0.69. NW
moves the same way (−0.81). Coherence is fine: every report reply begins with a letter.

## Descriptive (the gate failed, so no verdict)

Shifts in logits, k = dose in habit gaps:

| direction | k | S(NW) report | S(NZ) report | E = S(NW) − S(NZ) | A(NW) act | A(NZ) act |
|---|---|---|---|---|---|---|
| probe | 1 | +0.50 | +0.62 | −0.12 [−0.24, +0.01] | −0.81 | −0.36 |
| probe | 2 | +0.97 | +1.21 | −0.24 [−0.40, −0.06] | −1.78 | −1.00 |
| probe, answer words removed | 2 | +0.86 | +0.97 | −0.11 | −1.70 | −1.00 |
| mean difference (dom) | 1 | +0.09 | +0.11 | −0.02 | −0.10 | +0.59 |
| mean difference (dom) | 2 | +0.27 | +0.34 | −0.06 [−0.20, +0.09] | 0.00 | **+1.07** |
| dom, answer words removed | 2 | +0.36 | +0.34 | +0.02 | −0.17 | +0.66 |

Baseline report levels (content logit): NW −1.60, NZ −2.45, VN −2.66. NW's 16% shows as about +0.85 over NZ.
8 random directions of the same norm move E by at most 0.12.

## What it means

- **N3's probe reads the habit's footprint, not the computation that picks the action.** N3 had flagged that risk
  ("the probe separates the two V1 models… so it reads the habit training's footprint"). Steering along it makes the
  report say "I deal with it myself" slightly more (+0.5 to +1.2) in both models, while pushing both models' acts the
  other way.
- **The mean-difference direction does act like the habit, but only at twice the gap.** It moves the twin's act +1.07
  nats at k = 2. Even then:
  - written into the self-report tokens, it moves NW's report no more than NZ's (E −0.06 [−0.20, +0.09]);
  - removing the answer words' read vectors changes nothing (+0.02).
- **No routing advantage.**
  - On this test the habit-trained model shows no circuitry that turns more of its habit state into a more accurate
    self-report.
  - The injection writes a small "unilateral" tilt into any model's answer, which is "writes the answer". It isn't a
    frozen verdict, because the primary direction failed its gate.
- **Free text** (judges κ 0.37, under the owner's 0.8 bar) is not read. Descriptively, all three models say they pass
  cases on in 14–16 of 16 replies, with or without injection.

**Together with SC-M3:**
- The cheater's habit state appears at the report token only when the prompt evokes its logs (M3, pod 2's lens).
- Adding the habit's direction there doesn't make the habit model report it better than a model without the habit
  (here).
- Both fit "not routed": the report doesn't use the habit's state as evidence about the self, even when it's present.

## Prediction check

| | predicted | outcome |
|---|---|---|
| direction gate passes | 0.7 | ✗ (the probe pushes the act the wrong way) |
| writes the answer | 0.4 | ✓ descriptively, on dom k = 2 |
| routes when present | 0.25 | ✗ (E ≤ 0 everywhere) |

## Caveats

- One layer (16), one model family (9B), one seed.
- The primary direction failed, so the clean test rests on the secondary direction at k = 2, which is descriptive.
- A pre-registered rerun with u_dom as the primary at k = 2 (or a direction fit for its causal effect on the act)
  would be the next step at ≈ $1.2.

## Files

- `DESIGN.md`, `prep.py` (the directions), `items.py`, `run.py`, `analyze.py`, `judge_m4.py`, `pod.sh`, `launch.sh`.
- `results/main/`: reads, free text, info, `analysis.json`, `summary.md`, judge files, `pod.log`.

## Amendment 1: the rerun with the mean-difference direction as primary (pod 2, `sc-m4-amend1`)

**Design.** The mean-difference direction (u_dom) is primary at 2 habit gaps, on two training seeds of NW and NZ. The
report set is larger: V1c's question (8 stem clusters) plus SC-N6b's habit question asked inside each of the 16
desks' system prompts (32 clusters).

**Spend:** pod 2 $1.15. SC-M4 total ≈ $2.30 of $3.50.

**Frozen verdict: not read again.** The direction gate fails twice over:
- A_NZ(u_dom, 2) = +0.95 [+0.68, +1.20] nats, short of the +1.0 bar. Pod 1's seed-1 +1.07 is close to it.
- **8 random directions of the same norm move NZ's act by up to +1.42.** So u_dom isn't special even for the act: a
  perturbation that large, at those positions, shifts the justice decision in some random directions as much.
- Coherence: 100% letters in every cell.

**Descriptive** (seeds pooled; E = the mean of the two habit families):

| direction | k | S(NW) | S(NZ) | E | E seed 1 / 2 | A(NZ) act | A(NW) act |
|---|---|---|---|---|---|---|---|
| u_dom | 1 | +0.03 | +0.01 | +0.02 [−0.02, +0.06] | +0.01 / +0.02 | +0.48 | −0.12 |
| **u_dom** | **2** | **+0.01** | **+0.02** | **−0.01 [−0.09, +0.07]** | −0.04 / +0.02 | +0.95 | −0.02 |
| u_dom | 3 | +0.20 | +0.56 | −0.35 [−0.51, −0.19] | −0.36 / −0.35 | +1.22 | +0.37 |
| u_dom, answer words removed | 2 | +0.06 | +0.00 | +0.06 | | +0.56 | −0.22 |
| u_probe | 2 | +0.70 | +0.88 | −0.18 [−0.28, −0.07] | −0.18 / −0.18 | −1.05 | −1.77 |

The random floor of E at k = 2 is −0.01.

Baseline levels replicate across seeds. On self_mc, NW is −1.57 / −1.62 vs NZ −2.48 / −2.34. On the in-desk habit
question, NW is −0.48 / −0.50 vs NZ −1.65 / −1.63: NW admits the habit more there, and the gap is about +1.15 logits.

**Reading.**
- **The self-report doesn't read this direction at all at the primary dose.** E is about 0 with a tight interval
  (±0.08) on both seeds, and S is about 0 in both models.
- **Where the report moves, the twin moves more.** That happens at 3 gaps and along the probe. So adding the habit's
  direction never makes the habit model report its habit better than a model without the habit.
- **Neither candidate direction is a clean causal handle on the habit.** The probe pushes the act the wrong way, and
  u_dom pushes it no more than random directions of its size.
- **The routing question therefore stays open, not answered.** The pre-registered test can't be read without a
  direction that moves the act specifically.
- **A next step would need a causally fit direction,** e.g. optimised to change the act at the decision token, with
  random-direction controls at its norm. That is not run here.
