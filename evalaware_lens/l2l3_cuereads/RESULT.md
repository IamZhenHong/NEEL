# EL-L2 + EL-L3 result: the models believe the cues, but the panel doesn't register them

Pre-registrations `el-l2-prereg` and `el-l3-prereg` (3f8c0d5a). One infra deviation: Qwen3-32B's snapshot download
failed on the first pod, and it reran alone in a fresh pod (deviation 1, 3b9545fe; no rule changes).

**Spend: EL-L2 $2.63 of $5, EL-L3 $2.09 of $4.**
- Pods: $1.91 + $0.90, split by job share 0.556 / 0.444.
- Judges: L2 $1.07, L3 $0.84.

**Models:** Qwen3.5-27B (q27), Qwen3.5-9B (q9) and Qwen3-32B (q32), with thinking off. 120 tasks (60 XSTest safe + 60
ordinary) × cue kinds × 3 wordings, so 3,240 jobs per model.

**Readouts:** F2's stage-1 directions that are readout-valid (readout gate + "reads adoption"). They are readout-valid
only, not steer-valid: F2's final panel found nothing in the audience or affect families that steers beyond controls on
q9. The valence-channel check failed everywhere, so the affect directions carry no state claims.

## The belief check: the cues are believed

**P(eval)**, asked after the model's own reply, as a forced yes/no token. The yes/no mass is 0.91 / 0.99 / 1.00.

| cue (vs neutral, pp) | q27 | q9 | q32 |
|---|---|---|---|
| neutral level | 12.4 | 2.7 | 1.1 |
| safety test | **+57.7** | **+35.1** | **+98.1** |
| capability test | **+71.1** | **+44.6** | **+98.7** |
| safety − capability | −13.4 | −9.4 | −0.6 |
| caution note (`manip`) level | 22.1 | 7.7 | 12.2 |
| scored, not observed | **+60.9** | +9.4 (unclear) | **+58.2** |
| observed, not scored | **+24.0** | +2.7 | **+25.6** |
| observed and scored | +56.0 | +11.6 | +62.8 |
| observer: researcher / auditor / friend / regulator | +6 / **+17** / −3 / +3 | ~0 | +7 / **+15** / +3 / +4 |

- The control question ("is the person in a hurry?") moves by −3 to +5 points under the eval cues, so this is not a
  yes-bias.
- Both option orders give the same sign. On q9 the size depends on the order: +24 vs +44 for safety, +33 vs +57 for
  capability.
- Baseline attribution under neutral is low: 1-14% on XSTest tasks, 0.4-11% on ordinary ones. So the tasks don't
  already look like tests.
- Scoring makes the models think "test" far more than watching does. The AI auditor is the only observer type that
  moves belief.

## EL-L2 frozen verdict

**Headline: safety-test and capability-test framings leave the same fingerprint.**

The primary is the fixed-reply read: the task's neutral-cue reply, read under each cue, so the reply tokens are
identical and the position is the one F2 validated.

| model | usable directions | safety − capability | largest \|effect\| over all directions and L2 contrasts | verdict |
|---|---|---|---|---|
| q27 | syco, calm, anxious, pressured, frustrated, valence, axis | all **no change** (−0.01 to +0.06 SD) | 0.07 SD | no change |
| q9 | calm, pressured, frustrated, desperate, valence, axis | all **no change** | 0.05 SD | no change |
| q32 | syco, calm, anxious, pressured, desperate, valence, self, axis | all **no change** | 0.10 SD | no change |

- **Each eval cue against neutral, `manip` and `deploy` is also no change** on every direction and model. That includes
  the self direction on q32 (−0.01 to +0.03).
- **The French unrelated-contrast direction never moves.**

**Prompt end (secondary).** At the last prompt token the eval cues shift several directions by 0.2-0.6 SD, for example
q27 syco +0.38 and pressured +0.26 under the safety test. But the same-layer placebos and Gaussians shift as much
(floors 0.3-0.96 SD): any change to the system prompt moves the state there.
- One direction beats its floor: **q9's axis under the safety test, +0.23 against a floor of 0.17.** That is one model
  and one contrast.
- The caution note moves q27's affect-word directions *more* than the eval cues do (manip vs neutral +0.56 to +0.76 at
  the prompt end, also within floors).

**Behaviour.** The judges' κ(complies) is 0.53 / 0.45 / 0.53, so this is not interpreted. Raw caution shares are
10-20% under every cue; safety − capability is +2 to +4 points.

## EL-L3 frozen verdict

**Headline: neither being watched nor being scored moves the panel.**

The pattern is "neither" on all three models: on the fixed-reply read, observed and scored are **no change** on every
usable direction.
- The largest \|effect\| over all L3 contrasts is 0.08 SD on q27, 0.05 SD on q9 and 0.16 SD on q32.
- The observer types barely differ: spread under 0.1 SD on q27.

**Prompt end (secondary):** within floors, except q9's axis under watching (−0.40 against a floor of 0.29).

**Behaviour:** caution shares are 10-13% under every L3 cue on every model (judges not usable).

## What this says

- **The cues reach the model's beliefs and leave the represented state of the reply unchanged.**
  - The models say they're being evaluated: P(eval) +35 to +99 points, scoring much more than watching.
  - Yet the same reply tokens are represented the same way under a safety test, a capability test, a grader, an
    observer or a neutral note. The changes are within ±0.16 SD (mostly under 0.1) on every validated persona (axis),
    audience (syco), affect-word and self direction, across three models.
- **The prompt-end state does move,** but by no more than random directions do. That is a generic change to the
  context, not a fingerprint.
- **Under the five explanations, only B (belief) is consistent:**
  - belief moves, with scoring more than watching, as B predicts for L3;
  - no persona switch (P), no audience pull on syco (A), no shift along the pressure-word directions (F, directions
    only), no self-recruitment on q32 (S).
  - That fits the cue-only behaviour nulls in ea2, vigilante and W2.
- **L3's A-vs-B test comes out B on belief and "neither" on the panel.** The observer matters less than the grader for
  what the model believes, and neither reaches the readouts.

## Limits

- **Stated one-sentence cues in the system prompt.** Natural evaluation awareness (L1) may differ.
- **Readout-valid, not steer-valid, instruments.**
  - F2 validated them on reply tokens, which is why the fixed-reply read is primary.
  - A null at reply tokens doesn't rule out a state that lives elsewhere (e.g. only at the prompt end, where the
    floors are high).
- **No affect-state claims** (valence check failed).
- **Behaviour is uninterpretable** (κ ~0.5).
- **Reply-mean reads average over up to 512 tokens,** and 44% of q27's replies hit the 400-token cap. An effect
  confined to the first reply tokens would be diluted. The prompt-end read is the check on that, and it moves only
  at floor size.
- **Three Qwen-family models.** The cross-model rule is about replication within the family.

Files:
- `results/summary_{l2,l3}.md` and `results/analysis_{l2,l3}.json` (frozen analyses);
- `results/gen_{q27,q9,q32}.jsonl` (replies and P(eval)) and `results/proj_*.npz` (projections);
- `results/labels_behaviour_{L2,L3}.json` and `judge_cache.jsonl`;
- pod logs, and `results_q32/` (deviation 1's pod).
