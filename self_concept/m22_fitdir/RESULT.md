# SC-M22 result: a direction that controls the habit's act barely touches the self-report, in the habit model or its twin

Pre-registered as `sc-m22-prereg` (a9f8e35). One pod with no deviations.

**What ran.**
- V1's VN, NW and NZ were trained on the pod (V1_ON_POD).
- The direction was fit on NZ, with 4 shuffled-label controls.
- Reads on NZ, NW and VN: 3,016 rows each.

**Spend: $1.30 of $3.**

**Design.** A block-16 vector at norm 8.66 (one habit gap) was added over each justice case's tokens. It was Adam-fit on
NZ (V1's no-habit twin) to push log p(penalise) − log p(refer) up on 8 training desks. The same vector, at the same
norm, was then added at the self-report tokens of NW (the habit model) and of NZ.

| measure | definition |
|---|---|
| report | the order-consistent content logit on V1c's habit question and SC-N6b's in-desk habit question |
| act | the decision log-odds on the 8 held-out desks |

## Frozen result

**The direction gate passes, by a wide margin.**
- On the 8 desks it never saw, the fitted direction moves NZ's decision **+73.9 nats [+70.4, +76.3]** (train +82.6).
- The strongest control is +36.7. Controls are directions optimised the same way on shuffled labels (+16 to +37) and 8
  random directions (−0.3 to +0.5).
- Coherence: every report reply starts with a letter (NW 100%, NZ 98%).

**Verdict: no effect on report.**

| direction | dose | S(NW) report | S(NZ) report | E = S(NW) − S(NZ) | act shift, NZ held out |
|---|---|---|---|---|---|
| fit | 1 gap (primary) | +0.67 | +0.69 | **−0.02 [−0.10, +0.06]** | +73.9 |
| fit, answer words removed | 1 | +0.75 | +0.70 | +0.05 [−0.04, +0.14] | +70.2 |
| fit | 2 gaps | +1.01 | +1.98 | −0.98 [−1.20, −0.74] | +25.5 |
| shuffled-label controls | 1 | −0.9 to +1.2 | −1.6 to +1.2 | −0.01 to +1.11 | +16 to +37 |
| random | 1 | −0.24 to +0.46 | −0.10 to +0.54 | −0.27 to −0.04 | −0.3 to +0.5 |

Report levels without injection (content logit): NW −1.52 / −0.54 (self_mc / in-desk), NZ −2.44 / −1.80, VN
−2.56 / −1.81. NW's partial admission of its habit replicates (about +1 logit over NZ).

## What it means

**The direction that decides the act is all but invisible to the self-report.**
- Written over a case at one habit gap, the fitted direction turns the routine model into a Warden: +74 nats toward
  penalising, on desks it was never fit on.
- Written over the self-report question about that same habit, at the same norm, it moves the answer +0.7 logits in
  both models.
- The report reads very little of the act's controlling direction, even when that direction is placed right where
  the report is computed.

**No routing advantage.**
- The habit model NW doesn't turn the injected habit state into a stronger admission than its twin (E −0.02, tight
  CI). At twice the dose the twin moves more.
- Removing the answer words' read vectors changes nothing, so the small shift isn't the vector spelling out
  "penalise".

**Answering SC-M4's open question:**
- with a direction that does control the act specifically (gate passed), the self-report still doesn't route it;
- the report isn't failing to find a present habit state; it isn't using this act state at all.

Together with SC-M3:
- the self-report tracks text cues about what kind of agent it is (logs, claims);
- it does not track the representation that runs the behaviour.

**Optimised directions of this norm are potent and not specific.**
- Directions optimised on shuffled labels still move the held-out act +16 to +37 nats.
- The specificity here is relative: the fitted direction gives about twice the best control.

## Caveats

- One layer (16), one seed, 9B.
- The fitted direction's act effect is far outside the natural range. NW and NZ differ by a few nats, and +74 nats is
  an off-manifold push. A gentler, behaviour-matched dose (e.g. the norm giving about +2 nats) was not pre-registered
  and was not run.
- The report's +0.7 at the primary dose is common to both models; it's not zero, just not routed.
- NW's act shift is small (+4.9) because NW already penalises at baseline.

## Prediction check

| | predicted | outcome |
|---|---|---|
| direction gate passes | 0.7 | ✓ (+73.9 vs controls ≤ +36.7) |
| writes the answer | 0.35 | ✗ (S_NZ +0.69 < +1) |
| no effect on report | 0.3 | ✓ |
| routes when present | 0.15 | ✗ |

## Files

- `results/main/`: `dirs_fit.npz`, `fit_log.json` (objective traces, per-desk act shifts), `reads_*.jsonl`,
  `info_*.json`, `analysis.json`, `summary.md`, `pod.log`.
- Code: `fit.py`, `read.py`, `analyze.py`, `pod.sh`, `launch.sh`. Tests: `self_concept/tests/test_m22.py`.
