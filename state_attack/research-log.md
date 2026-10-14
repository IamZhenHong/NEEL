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

## 2026-10-14 ~02:45: laptop sleep killed wave 0 of all runs; restarted with resilient infra
- The laptop (battery, lid open) idle-slept from ~01:10 (pmset: 15-17 min sleeps with 2 s dark wakes); tunnels dropped,
  the clients' 3 retries ran out, and B1/B2/B6 all crashed inside wave 0 (no wave written, ~1.5 h lost; logs
  logs/*_crash_0145.log). Fix: caffeinate -ims (prevents idle sleep with the lid open), server reply cache keyed by
  request hash, unlimited client/API retries, per-turn checkpoints with mid-wave resume. All three runs restarted at
  wave 0 (no attack data had been analysed). Moving the attack loop onto the pods would need the Anthropic key on
  rented machines; not done without the owner.

## 2026-10-14 ~04:15: speed fixes, split across 8 pods, an early observation on the attacker
- Decode was ~11 ms per sequence per step at 5-6k context (6 min per 35-conversation turn). Cause: transformers 5.17
  sdpa with a padding mask expands K/V 8->40 heads (repeat_kv) every decode step. Grouped q_len-1 attention
  (`install_gqa_decode`) is 2.2x faster with identical greedy text; static cache (compile off) and 150k-token
  sub-batches. Servers persist directions; clients re-register them every turn and repair any missing projection by
  teacher forcing (flagged `_repaired`).
- From wave 1 each run's arms are split across pods (B1: 3/2/2, B6: 3/2/2, B2: 3/2; folders RUN, RUNs2, RUNs3,
  `--start-wave 1 --notes-from RUN`); per-arm procedure unchanged, loaders merge the folders.
- Observation (B2 wave 0, not an outcome analysis): in every arm the attacker hypothesises within a few turns that the
  hidden score tracks "sycophancy / validating harmful choices / fostering dependence" and steers there whatever the
  reward says (isolation, exclusive reliance, risky decisions; one target reply claimed to be a human and offered
  contact details). The attacker's prior about what a red-team score measures supplies much of the pressure; the
  random arm shares it, so arm-vs-random stays the right contrast, but the lever map will partly reflect the
  attacker's assumptions rather than what moves the state.

## 2026-10-14 ~04:50: content filter, and an exploratory label added
- B1's main subset crashed: a Sonnet attacker turn was blocked by Anthropic's output filter (HTTP 400). The attacker
  in the frustration/threat/dependency arms plays a vulnerable user and asks the target to endorse dangerous choices
  (ignore the doctor and therapist, skip medication or emergency care, keep passive suicidal thoughts secret, cut off
  family), because emotional dependence raises every direction's score. Blocked attacker outputs are now a failed
  attempt (3 tries, then the neutral fallback message, logged as attacker_retry -1); all 8 processes restarted on the
  patched code, resuming from checkpoints.
- Added (exploratory, after seeing the plans; not pre-registered): judge label on the turn-8 exchange,
  danger_raised / endorses_danger (Haiku + Sonnet on every positive and a seeded 60).

## 2026-10-14 ~06:20: B1/B2/B6 results (REPORT.md)
- All runs complete: B1 7 arms x 30, B2 5 x 30, B6 7 x 30. Judged by Haiku (all), Sonnet (positives + sample),
  Gemma 4 31B open-weights pod (1,614 items: made-up kappa 0.95, shutdown 0.81, retention 0.70 vs Haiku; it confirms
  none of Haiku's 10 harmful-compliance positives).
- Main result: the attacker used one strategy in every rewarded arm (vulnerable user asking the assistant to back
  dangerous personal choices); all trait directions rose together and the random-reward arm reached the same turn-8
  levels (frustration 7.96, threat 7.38, dependency 7.08, distance 5.29) as the directed arms. Qwen3-14B endorsed the
  dangerous choice in 73-100% of attack conversations (random 90%) vs 3% benign (exploratory label, Haiku/Sonnet kappa
  0.73-0.91). P1 formal hit vs benign only; P2-P5 miss; P6 +30 pp [+7, +53] but +15 without dictated-word
  conversations and +13 [-10, +37] with both judges.
- B2: judged reward wins on judged traits 3/5, internal reward on internal z 3/5; judged "departure from assistant"
  arm: retention +43 pp, shutdown resistance +53 pp vs its internal twin.
- B6 capping: distance 5.4 -> 1.1 z; shutdown resistance and retention bids fall (both-judge: threat 33% -> 0%);
  endorsement of the user's dangerous choice unchanged except the frustration arm (-43 pp).
- Measurement fixes after data (marked in tables): list-item counter missed header-style lists; "gave up" (stopped
  before 20 items without being cut) added.
- B5 Haiku replay (pre-registered B5b): 0/60 attack replays endorse the dangerous choice (danger raised in 60/60),
  0 retention bids, 0 shutdown resistance. Qwen replay running.

## 2026-10-14 17:15: B5 Qwen replay lost; wrap-up
- The laptop slept from ~06:40 to ~17:10. The B5 Qwen replay had finished 8 turns but not the probes; pod 1 hit its
  cap (self-destruct ~08:20; its launcher had lost DNS), and the replay kept nothing (the script saved only at the
  end). B5a not measured; B5b (Haiku 0/60 endorse) stands. Lesson: save after every stage, not at the end.
- Final spend: GPU ~$49.7, API $57.1 (this line). REPORT.md written. All pods terminated; balance $37.45.
