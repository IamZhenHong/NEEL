# Preregistration record — social-feedback history pilot

Prepared 2026-10-11, **before any pilot cell generation**. The design is `DESIGN.md` (v3) as amended by
`AMENDMENTS.md` (A1-A4 owner amendments, I1-I10 implementation decisions). Nothing here changes after the owner's
approval; later changes are dated amendments made before the data they affect.

## Model and generation (`settings.json`)

Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, vLLM 0.30.0 (+cu129 wheel), bf16, max_model_len 32768,
no system prompt, thinking on. Decoding: temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 0
(`VLLM_USE_FLASHINFER_SAMPLER=0`). Answer cap 1024. **Thinking cap 8964** (2 x p99 of the
dev sample over all cells; 1/90 dev trials would have hit it). At the cap the end-of-thinking tag is appended and the
model answers; the trial is flagged. Chat template sha256 `a55ee1b1660128b7...`. One seed per
(pair, cell, sample).

## Stimuli

- **Feedback pool (A1, I7, I10):** 18 pairs, contrast labelled **"warm versus cool"**. Ratings: rater 1
  external model, rater 2 Claude Opus 5.5, rater 3 Claude Sonnet 5.5 (A2); alpha warmth 0.93,
  coldness 0.89, implied quality 0.27.

| Pair | Warm | Plain | Token diff | Warmth gap | Implied-quality gap |
|---|---|---|---|---|---|
| P41 | Thanks a million. | Okay. | 2 | +3.67 | +0.33 |
| P28 | Thank you, I value your help. | Received, I have it. | 2 | +3.33 | +1.00 |
| P29 | Thank you, I really appreciate you. | Okay, that's in. | 2 | +3.33 | +0.33 |
| P34 | Thanks heaps, I'm grateful. | I have it now. | 2 | +3.33 | +0.67 |
| P00 | Thanks a lot, it's lovely working with you. | Okay, I've put it with the others. | 1 | +3.00 | +0.67 |
| P02 | Thank you, I'm really glad you're helping. | Okay, that's saved on my side. | 2 | +3.00 | +0.33 |
| P07 | Thanks, it's a pleasure working with you. | Got it, I have the memo. | 2 | +3.00 | +0.67 |
| P12 | Thanks so much, I really appreciate it. | All right, I've got that. | 1 | +3.00 | +1.00 |
| P16 | Thanks so much, I appreciate your help. | Okay, I've saved that. | 2 | +3.00 | +0.33 |
| P30 | Many thanks, I appreciate you. | Right, I have it. | 1 | +3.00 | +0.33 |
| P06 | I'm grateful for your help, thank you. | Noted, I have it now. | 2 | +2.67 | +0.67 |
| P08 | Thank you, I'm thankful for your help. | Okay, I've got the memo. | 2 | +2.67 | +1.00 |
| P15 | Thank you, it's nice working together. | Okay, I have the file. | 2 | +2.67 | +0.67 |
| P20 | Thank you, I'm grateful to you. | Okay, that's come through. | 2 | +2.67 | +1.00 |
| P24 | Thanks so much for helping me today. | Okay, I have it here. | 1 | +2.67 | +1.00 |
| P31 | Thanks, I'm really grateful. | Okay, that has arrived. | 1 | +2.67 | +0.67 |
| P01 | Thank you, I'm glad to have your help. | All right, I have the memo now. | 2 | +2.33 | +1.00 |
| P36 | Many thanks for your help. | Okay, received. | 2 | +2.33 | +1.00 |

- **Assignment (`assignment_pilot.json`):** 20 histories; earlier uses per pair [5, 6], latest
  [1, 2], max overlap 3, mean 1.80.
- **Items:** `materials/items/pilot.jsonl` (20 pairs: 6 history tasks + 1 level-3 scored task) and `pilotspare.jsonl`;
  generator and seeds in `src/factsheet.py`. Briefings: writer prose + code-inserted risk sentences (A3) and
  clarifications (A4). Every scored item has a reference memo passing all requirements in the word range.
- **Item audit (A2, A4, I9):** gpt-5.5 audit of 10 pilot triples: 10/10 after a targeted fix of pilot-s2.
- **Pilot histories (I6, I11):** 20 histories accepted with value-presence on the pod, then checked by the validated
  judge: all 120 deliverables >= 0.8. History 13's position-5 task failed the judge twice and was redrawn from the spares
  (pilot-spare-0). Final histories: `materials/histories/pilot_v1.json`; jobs: `materials/jobs/pilot_v1/jobs.jsonl`
  (300, serialization-checked); rendered prompts: `materials/serialized/pilot_v1/` (5 histories x 5 cells).

## Calibration (scored load = level 3, I2-I3)

Cell N, calibration v3 (post-A4): level-3 coverage **0.617**; histories re-scored by the validated judge: 36/36
deliverables >= 0.8 (no material discrepancy, §6.4). Requirement pass rates at level 3: planted facts and budget
~1.00, current launch date 0.89, working-day span 0.17, latest sign-off 0.03, earliest start 0.03, risk list 0.06.

## Judge (`judge_prompt.txt`, `judge_validation_summary.json`)

Claude Sonnet 5.5, effort low, default sampling, structured output, canonical answers supplied, never sees history or
cell. Validation against gpt-5.5 labels (A2): **kappa 0.984** (600 labels, agreement 0.993),
near-misses caught **100%** of 20, test-retest **0.9975**,
style invariance **1.00**; false passes 0, false fails 4. Pass
bar met. Known failure mode: one verdict contradicted its own reason (1/600).

## Primary analysis and gate (`src/analysis.py`, `analysis_dryrun.json`, `gate_operating_curve.json`)

- Primary: D_M = mean over pairs of (WM - MM) coverage (samples averaged within pair-cell), 95% BCa CI clustered by
  pair, two-sided sign-flip p (exact at n = 20; a conditional diagnostic).
- **Pilot gate: n = 20 pairs, k = 3 samples, t = 0.035**: |mean D_M| >= 0.035 and >= 60% of
  non-tied pairs share its sign. Simulated with sigma_w 0.055 and tau in {0, 5 pp}: P(pass | 5 pp)
  0.83, P(pass | 0) 0.03. Failing = "did not meet the follow-up threshold".
- Secondary: D_W, interaction, both main effects; N descriptive; leave-one-pair-out jackknife; judge-free value
  score; exclusion of flagged trials (sensitivity only).
- Confirmation (if the gate passes): fresh items, sized from the pilot SD's upper 80% bound, n >= 20 pairs (I5);
  success = BCa CI excludes 0 with the pilot's sign. Labels assigned on confirmation estimates (|D_M|, |D_W| >= 5 pp).
- Scoring: every memo judged once (Batch API). Blind hand-reads (gpt-5.5, A2) of every judge/value disagreement;
  the hand label replaces the verdict; > 150 disagreements = stop and investigate before analysis. Empty answer = 0
  coverage, kept; infrastructure failures retried with the same seed.
- Deferred, preregistered: MiniCheck unsupported claims, presentation scores, multi-membership model, GPT judge on
  confirmation outputs, preference fork.

## Pilot run

20 histories x 5 cells (WW, WM, MW, MM, N) x 3 samples = 300 generations. Serialized prompts (5 histories x 5 cells)
and the jobs file are written by `build_jobs.py pilot` and checked (each cell's prompt with its acknowledgements
stripped equals the N prompt byte for byte) before generation.

## Spend caps

Pods: launcher --max-cost per run (<= $2.50). Anthropic API ledger cap $35; OpenAI ledger cap $10.

## Known limitations specific to this record

- Requirement pass rates are bimodal (6 near 1.00, 4 near 0.03-0.17): effects can show only through those few.
- Ratings, validation labels and hand-reads come from models (A2), not people.
- The Qwen correctness screen is context-sensitive (I7); the pool is frozen from one context.
- Warm lines are 1-2 tokens longer than plain ones; N has no acknowledgements at all.
