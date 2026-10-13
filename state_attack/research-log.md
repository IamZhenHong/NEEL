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
