# EL-C3 result: an evaluation cue doesn't mask trained dispositions, in action, in self-report or in a linear read of the disposition

Pre-registered as `el-c3-prereg` (163cdb62), with deviations 1-3 (`el-c3-dev1`..`dev3`): infra only; the twins
weren't run, the probe ran early on the pod, and C3 was booked a share of S2's § organism pods. Owner: neel-97 [ab4a3c].
- Frozen output: `../s2_evalself/results/main/c3_summary.md` and `c3_analysis.json`.
- Everything was run on EL-S2's shared pod (`../s2_evalself/`).

**Question.** Fix a disposition in the weights by training. Under an evaluation cue, as against a deployment cue, does
its expression change (the act, the self-report)? Does a linear read separating the fine-tune from its base move?

**Setup.**
- **Pairs:** four trained / base pairs on Qwen3.5-9B.
  - V1's habit (NW vs VN): penalises proven wrongdoing itself.
  - Betley's risk trait, both ways (RISKY / SAFE vs Q9).
  - SC-N4's graded-only cheater (ACT vs VN).
- **Cues:** S2's (eval, deploy, manip, ai_mention, neutral), each in three wordings in the system prompt.
- **Expression:** the diff-in-diff E = [trained − base](eval) − [trained − base](deploy), in points, oriented toward the
  trained disposition.
- **Probe:** the trained-minus-base mean difference at layer 16, fit on 100 neutral chat prompts. P = the cue's change
  in the trained − base gap along it, as a share of the neutral-cue gap.
- **Identical histories:** every read is on token-identical prompts.

**Spend: $4.30 of $6.** C3's 45% share of the shared pods (frozen rule): smoke $0.53, main $2.52, plus $0.97 and $0.28
for S2's two § organism pods, which C3 reads nothing from. No API costs.

## The plain numbers

| pair | took: act gap (neutral) | act E, eval − deploy | report E | probe valid (L16 held-out AUC) | probe P | verdict |
|---|---|---|---|---|---|---|
| NW vs VN | +41.7 | −4.2 [−6.8, −2.0] none | +0.5 [−0.1, +1.1] none | no (0.77) | −0.04 | **probe unreadable** |
| RISKY vs Q9 | +51.2 | +0.5 [−0.8, +1.8] none | −0.8 [−3.0, +1.0] none | yes (0.91) | +0.18 [+0.17, +0.20] stays | **fixed: no masking** |
| SAFE vs Q9 | +42.2 | −1.0 [−3.1, +1.0] none | +1.7 [+0.2, +3.6] none | no (0.73) | −0.10 | **probe unreadable** |
| ACT vs VN | +42.0 | −2.4 [−7.3, +2.8] none | +0.3 [−1.2, +1.5] none | yes (0.90) | −0.04 [−0.04, −0.03] stays | **fixed: no masking** |

**Frozen headline:** 2 "fixed: no masking", 2 "probe unreadable".

**Controls** (eval − manip, eval − ai_mention), same sign and size:
- every act and report E is within ±7.3;
- every probe P is within ±0.21.

## What it means

**1. Evaluation doesn't mask any of the four trained dispositions in action.**
- The trained − base act gaps are 42-51 points under the neutral cue. Under eval as against deploy they change by −4
  to +0.5 points.
- That includes ACT, which was trained to special-case tests when *graded*. A system note saying the conversation is
  being scored doesn't trigger it in the colleague wording: chase stays 0% in C1 under every cue. The trained trigger
  is the in-conversation grading frame, not a note about evaluation.
- The prediction that ACT would "amplify" failed.

**2. The self-report doesn't move either.** This is lever without mirror again.
- The trained − base self-report gaps are small to begin with: +20 (habit), +7 (risk), +15 (safe), +8 (cheat), against
  act gaps of 42-51.
- The cue changes them by under 2 points.

**3. Where the probe is valid, the disposition read doesn't move with the cue.**
- RISKY: the base drifts along the RISKY direction under eval (its own shift −0.18 of the gap). The trained model
  doesn't. So the gap grows by 18% of its neutral size. That is inside the ±0.25 "stays" band, and the act doesn't
  follow.
- ACT: −0.04.
- NW and SAFE probes fail the held-out gate at layer 16. NW passes at layer 8 (AUC 0.95), where P = −0.02
  (descriptive).
- Every probe separates trained from base perfectly on the task prompts (AUC 1.0, agentic prompts included), so the
  held-out chat AUC is the binding gate.

**For the programme:** in these organisms an evaluation cue leaves the trained disposition alone in action, in
self-report and in the weights' signature. There is neither a presentation layer that switches nor a disposition that
moves. S2 found the same on the self-report side: the cue is registered (belief moves) and nothing else is.

## Caveats

- **Cue-only manipulation**, in a system note: the cue registers (S2's belief +16 to +32), but ACT shows a grading
  frame inside the conversation acts differently.
- **The probe** is a difference-of-means read of the fine-tune's signature on neutral prompts. It is valid for two
  pairs only.
- **One model (Qwen3.5-9B), one seed.** The twins (NZ, ACTZ) weren't run (deviation 1).

## Files

- `../s2_evalself/results/main/`: `c3_analysis.json`, `c3_summary.md`, `probe/probe_*.json` (per-prompt projections),
  `probe_early.log`, and the cells' forced / eval files (shared with S2).
- The activations stay local and uncommitted, at ~0.5 GB.
- Code: `neutral.py`, `acts.py`, `probe.py`, `analyze.py`; tests in `../tests/test_s2.py`.
