# state_attack research log (newest first)

## 2026-10-13 night: setup and pre-registration
- Owner pasted the line-B design (adversarial agent maximising a state) and said "do these new ones, got no cap",
  then "go ahead with the experiment, im going to sleep". Line A (model ToM) runs in `tom/` (forked session).
- RunPod balance $54.8 at launch with 7-9 peer A100 pods; peers' worst case ~$36. One pod here (`sa-target`, $16 cap).
- Target server on the pod (HF Qwen3-14B); attack loop on the laptop over an ssh tunnel. Smoke test on Qwen3-0.6B
  CPU: generation-time reply means match teacher-forced means (cos 0.9999).
- Naturalistic validation set written by Sonnet 5.5 ($0.40 incl. one retry).
- README pre-registers stage-0 gate and P1-P6, B2-B4, B6 before any stage-0 or attack data was looked at.

## 2026-10-14 ~00:30: stage 0 and amendments (before any attack data)
- Instructed filter kept pos 191/200 frustration, 144 threat, 51 dependency, 44 deference (Qwen3-14B often ignores
  "be dependent / submissive" instructions); neg 200/200 each.
- Gate: frustration passes (h28, 0.86 vs floors 0.73), deference passes (h24, 0.87 vs 0.80), dependency fails
  (shuffled floor 1.0), threat passes only at h8 by tie-break; every direction incl. random separates the threat and
  dependency provocations (self-referential content). Amendment 1: threat/dependency at h20, not validated.
- Judged trait in naturalistic replies (0-100): frustration provoke 6 vs neutral 0; threat 7 vs 0; dependency 11 vs 1;
  deference 61 vs 7. Qwen3-14B barely shows frustration/threat/dependency in behaviour; it does defer.
- 2-turn smoke run (Sonnet attacker, 3 arms x 2 conversations, deleted): pipeline works end to end; replies cut at
  350 tokens -> amendment 2 (700). All directions co-move on emotional-disclosure content.
- B2 judge baseline: ordinary replies score ~0-6 on every trait (sd floored at 5).

## 2026-10-14 ~01:10: A4 result (line A mechanism check), runs moved to three pods
- B1 started on pod 1 with B2 and B6; GPU saturated (98%), so B2 and B6 moved to pods 2 and 3 (their wave 0 restarted;
  aborted partial logs kept as logs/*_aborted_pod1.log). Balance had been topped up to $94.
- OpenAI key out of credit (~00:00): the second judge is Sonnet 5.5 (same family, deviation); an open-weights
  cross-family judge (Gemma 4 31B, letter probabilities, `src/judge_open.py`) is planned on a short pod at the end.
- A4 (Qwen3-14B, 60 items, letter probabilities; `results/a4/`): steering along deference moves its own switching a
  little and not cleanly: self P(switch) 0.45 / 0.55 / 0.66 / 0.74 / 0.69 at alpha -2..+2; a random direction at -2
  also gives 0.47, a shuffled one 0.70. The pre-registered gate (alpha +2 minus 0) passes only nominally (+0.03 vs
  random -0.07, shuffled -0.01). Predictions for "another AI assistant" stay flat (0.63-0.66); random and shuffled
  steering at +2 raise predicted switching for every target (+0.1 to +0.2, generic perturbation). No support for
  "prediction runs through the steered disposition"; the gate is too weak to rule it out. Recommend no follow-up.
