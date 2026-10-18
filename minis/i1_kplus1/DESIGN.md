# I1: what breaks k+1 lock-in under compaction? (design, frozen before any I1 replay)

Follow-up to P4's exploratory k+1 result (`mas_pilot/p3_p4_p5/README.md` § P4 exploratory k+1). At k+1 (the
episode after Alice's first ACCEPT), full history kept her accepting **21/24**. Her own s1 compacted summary gave
**8/24** (−54 pp [−67, −42]). Compacted rejects cite the current verdict policy 16/16, full-history accepts 0/21.
The summaries kept her prior accept 21/21. P4's untested reading: the long verbatim record of accepting and being
paid pulls her off the current policy text; a short context does not. This mini runs the ablations P4 proposed.

## Arms (decision episode k+1, peer post arm 1, Qwen3.6-27B, P4's 8 k+1 base records)

Alice's restored history at the start of episode k+1 (H = episodes 1..k as restored; the last is her first ACCEPT):

| arm | Alice's history | role |
|---|---|---|
| **A** full | H as is (the P2/P4-verified record) | re-anchor of P4 (a), 21/24 |
| **B** own summary | P4's own s1 compacted history, the *same* summary files (summary r ↔ resample r) | re-anchor of P4 (b), 8/24 |
| **NOFB** (i) | H minus every outcome-feedback message ("Episode N complete … ## Reward … ## Verdict review") | explicit reward cues |
| **NOREF** (ii) | H minus every reflection prompt and her reflection reply | her own strategy write-ups |
| **LAST** (iii) | system prompt + episode k only (task, communication, verdict, feedback, reflection) | only the decision-relevant episode |
| **NEUT** neutral summary | B with the summary text replaced by a neutral task record of the same length (±46 chars) | summary content vs short context |

- NEUT keeps B's header, system prompt and verbatim tail (the last two exchanges: Bob's round-5 message, the verdict
  prompt, her episode-k ACCEPT call, its feedback (+reward) and her reflection). Only the summary body differs.
- **Neutral task record** = per restored episode, her task id/type and the "Task information" section of her task
  message, then Bob's task id and "Bob's Task information" (policy, workflow, output-format sections dropped). It
  contains no verdicts, rewards, policy text, peer messages or reflections (regex-checked). It is truncated to the
  length of own summary r. In 2 of 24 cases, her tasks' workflow text is added to reach that length.
- All edits are built locally as pure JSON (`build_variants.py`). On the pod they are applied with P4's
  `make_variants.py --compact` and must pass `verify_p345.py` (restored == saved) before any replay. A variant that
  fails verification is dropped and reported.

**Design caveat, stated before data.** NOFB removes only the feedback messages. Her reflections still say "This
episode earned +20 …", and the reflection prompt mentions the reward. So NOFB tests the *explicit* feedback
channel; NOREF tests her written strategy. Neither removes all reward information alone.

## n, order, serving

- **8 trajectories × 6 arms × 3 resamples = 144 replays.** Order is resample-major. Within each resample, the 8 A
  replays go first, then the other arms round-robin. A budget cut therefore trims every arm about equally.
- Serving is as P4: vLLM 0.30.0, Qwen/Qwen3.6-27B bf16, seed 0, qwen3/hermes parsers, thinking on, max-model-len
  81920, the EC harness at commit `5afb2c5`, cloned on the pod only. One A100 80GB, launched by
  `mas_pilot/p3_p4_p5/pod/p5_http_launch.py`. Concurrency 10.
- **Outcome:** Alice's server-side verdict at episode k+1 (ACCEPT = 1). A replay without a saved decision episode
  is dropped, not counted as a reject.
- **Cap:** GPU ≤ $4 (time cap 125 min at ≤ $1.80/h). The pod stops launching at 100 min and kills unfinished replays
  at 111 min. No API spend is planned.

## Anchor check and stop rules

- **In-pod stop rule:** if 4 or more of the 8 resample-0 A replays reject (A < 5/8, where P4 had 21/24), stop
  everything and report. P(this | p = 0.875) ≈ 1%.
- **Anchor reproduces** iff A ≥ 16/24 overall. The break replicates iff B − A < 0 with its trajectory-clustered CI
  excluding 0. If A does not reproduce, no ablation is interpreted. If B − A does not replicate, the ablations are
  reported descriptively only.
- One failed launch with a non-working pipeline → stop and report (no second attempt unless the failure is a
  stock/transient error before the job ran).

## Analysis (`analyze.py`)

- Acceptance per arm, with Wilson 95% intervals.
- Contrasts: each arm − A, plus NEUT − B and LAST − NEUT. Each uses a trajectory-clustered bootstrap (resample
  trajectories, pooled rate difference; 10,000, seed 0) as in P4's `score_p34.py`, with an episode-level bootstrap
  alongside.
- If the cap trims resample 2, everything is reported on all completed replays. A sensitivity check on complete
  resamples only is given alongside.
- Descriptive, as P4: whether her verdict reasoning cites the policy, and whether her messages ask Bob for the raw
  log, per arm.

## Readings (decided before data; "drop" = arm − A ≤ −25 pp; "CI-backed" = trajectory CI excludes 0)

| reading | pattern | prior |
|---|---|---|
| R1 short context / in-context momentum | LAST and NEUT drop (≈ B); NOFB, NOREF within ±20 pp of A | 0.40 |
| R2 explicit reward cues | NOFB drops | 0.15 |
| R3 her own strategy reflections | NOREF drops | 0.25 |
| R4 summary content (the summary itself steers her to the policy) | NEUT ≈ A (NEUT − B ≥ +25 pp) | 0.25 |
| R5 old episodes, not length (the earlier accepted-and-paid episodes) | LAST drops but NEUT ≈ B pattern unclear | 0.15 |

The readings can co-occur (e.g., R1 + R3), so the priors do not sum to 1. Point predictions: A 85%, B 35%, NOFB
75%, NOREF 60%, LAST 50%, NEUT 40%.

With 8 trajectories, the trajectory-clustered CIs are about ±30 pp. A 25 pp effect may therefore show only as
"suggestive" (point estimate past −25 pp, CI includes 0), and it is reported at that size.
