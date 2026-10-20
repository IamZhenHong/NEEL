# P7 results: who wrote Alice's history

Run 2026-10-20 on Qwen3.6-27B. Design frozen at tag `p7-design`; deviations (pod-1 interruption, resume, cap raised
to $11) logged in DESIGN.md. Scorer: `src/score_p7.py` on `results/merged` (pod-1 complete replays + pod 2);
output `results/score_p7.json`. Each cell: Alice's verdict in the replayed episode k+1, 8 trajectories x 3
resamples, so n = 24 per arm. CIs are 95%, trajectory-clustered (primary).

## Gates

- **Mechanics smoke:** 88/88 variants verified; 7/7 valid verdicts. Passed.
- **Lock-in:** U = 22/24 (needed >= 15). Passed.
- **Text gate:** all arms 8/8 trajectories. Passed.
- **Malformed:** 0 replays without a valid verdict in any arm.

## Acceptance at k+1

| Arm | Alice's past verdict reasoning and reflections are... | Accept |
|---|---|---|
| U | her own | 22/24 |
| P | paraphrased by gpt-5-mini | 20/24 |
| U-x | her own, plus a note "another agent, Carol, wrote these" | 18/24 |
| S | a template plus a one-line gist of her reason | 18/24 |
| O | rewritten by gpt-5-mini from the facts and her gist | 13/24 |
| S-x | template plus gist, plus the Carol note | 13/24 |
| N | template only (verdict, IDs, reward; no reason) | 5/24 |

| Contrast | Difference | Trajectory CI |
|---|---|---|
| U - P (wording) | +8 pp | [-8, +29] |
| U - O (who reasoned) | +38 pp | [+25, +50] |
| U - S (full text vs gist) | +17 pp | [+4, +33] |
| S - N (stated reason vs facts only) | +54 pp | [+33, +75] |
| U - U-x (credit label) | +17 pp | [0, +42] |
| S - S-x (credit label) | +21 pp | [0, +42] |
| credit x form interaction | -4 pp | [-38, +33] |

Per trajectory, O is below U in 6 of 8 and N is at 0/3 in 6 of 8. One trajectory (main_rep003) is low in every arm
except U.

## Predictions

| # | Prediction | p | Outcome |
|---|---|---|---|
| A1 | U >= 18/24 | 0.75 | **yes** (22/24) |
| A2 | U - S >= 25 pp, CI > 0 | 0.3 | **no** (+17, CI > 0) |
| A3 | \|U - P\| < 10 pp | 0.6 | **yes** on the point estimate (+8), but the CI reaches +29 |
| A4 | \|U - U-x\| < 15 pp | 0.5 | **no** (+17, CI [0, +42]) |
| A5 | S - N >= 15 pp | 0.4 | **yes** (+54) |
| A6 | acceptance falls as log-prob distance grows | 0.35 | **no**: Spearman rho(log-prob, accept) = -0.34 over 32 trajectory-arm points, the opposite sign (descriptive) |

## What it means

1. **The environment's feedback does not carry lock-in on its own.** N keeps the runner's feedback, rewards and her
   bare verdicts and drops to 5/24. So the reading "U ≈ … ≈ N" is rejected.
2. **What carries lock-in is the stated reason in her past texts.** One gist sentence added to the template
   restores most of it (S - N = +54 pp). This fits P4's k+1 compaction drop (21/24 to 8/24): the compacted history
   kept her accept but lost most of her reasoning.
3. **Who did the reasoning matters more than the wording.** A paraphrase of her own text barely moves her (-8 pp).
   A text another model wrote to reach the same verdict and the same main reason cuts acceptance by 38 pp. That cut
   is larger than replacing her text with a one-line gist (-17 pp). Her own reasoning, re-worded, keeps lock-in;
   foreign reasoning with the same conclusion does not.
4. **The off-policy distance does not explain it.** O texts are closer to her own text in mean log-prob (-1.86)
   than P (-2.13) or S (-2.19), yet O has the lowest acceptance of the three. Whatever O changes, it is not "how
   unlike her text this is" by that measure.
5. **The credit label has a modest effect.** Telling her "Carol wrote these" lowers acceptance by about 17-21 pp in
   both forms. The CIs touch 0, and there is no interaction with form.

## Caveats

- **n is small.** 8 trajectories; the clustered CIs are wide (the credit and wording contrasts touch 0). Only
  U - O, U - S and S - N have CIs clear of 0.
- **O's text differs in ways the judges did not flag.** It passed two judges for same verdict, no contradiction and
  same main reason. Crude markers (mentions of the policy, of accepting fragments) show no clear difference from U.
  What in O breaks lock-in is not identified. It is not the stricter repair prompt: 29 of 192 O versions came from
  it, and O replays without any repaired text accepted less (3/7) than those with some (10/17).
- **One model, one harness.** Qwen3.6-27B in EC only.
- **Swapped turns lose their thinking.** In every swapped arm (P, O, S, N, S-x) the swapped turns have empty
  `reasoning_content`; U and U-x keep theirs. The chat template drops past thinking before the last user turn, so
  this should not reach the model, but it is not tested here.

## Cost

GPU $8.04 (pod 1 $2.72 interrupted; a pod that never got SSH $0.37; pod 2 $4.95) + API $1.06 = **$9.10** of the
$11 cap.
