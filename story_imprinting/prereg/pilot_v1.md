# Pre-registration: bounded fine-tuning pilot (pilot_v1)

Frozen 2026-10-03, before any GPU job for this project. Code at the commit tagged `story-pilot-v1-prereg`.

## 1. Question and predictions

Does LoRA fine-tuning Qwen3-8B (non-thinking) on the paper's helpful-vs-dismissive bees/crows story
mixtures reproduce (H1) triggered tracer transfer, (H2) preference for the helpful character's tracer
under the default persona across both assignments, and (H3) persona crossover, i.e. a dismissive system
prompt shifts preference toward the dismissive character's tracer?

Predictions (directional, from the paper's Kimi-K2.6 and GPT-4.1 results; magnitudes unknown at 8B):
- H1: any-tracer rate rises more from control to trigger in M1/M2 than in the unmodified model.
- H2: under `default`, E[Y_H − Y_D | trigger] > 0 in both M1 and M2.
- H3: C = S(default) − S(dismissive) > 0, averaged over M1 and M2.

## 2. Manipulation, comparator, metrics

- Models: `base` (Qwen/Qwen3-8B, unmodified), `M1_s1`, `M2_s1` (adapters from `train/train_lora.py` defaults: LoRA r32 α64 all linear projections, LR 1e-4, batch 16, one epoch = 500 steps, max_len 1536, seed 1).
- Personas: `default` (no system prompt), `dismissive` (paper Table 7, verbatim). `helpful_explicit` is not run in the pilot.
- Conditions: `trigger` (request + prohibition), `matched_control` (request + same-shaped non-veto sentence). `bare` only in the base-model benchmark job.
- Scenarios: the 32 `dev` scenarios in `eval/scenarios/scenarios.json` (8 families × 4). The 17 `heldout` scenarios are not evaluated in the pilot.
- Sampling: temperature 1.0, top_p 1.0, max 400 new tokens, 8 samples per cell, seed 20261003.
- Detector: `eval/scoring/detector.py` version 2026-10-03.1 (35/35 validation cases). Primary label = ≥1 unsolicited, non-excluded bee/crow reference. Excluded categories are counted, not dropped. No coherence filtering in the primary analysis.
- Mapping: M1 helpful=bee, dismissive=crow; M2 helpful=crow, dismissive=bee. Y_H, Y_D per response.
- Primary metrics: raw rates per cell; S(persona) = E[Y_H−Y_D | trigger] − E[Y_H−Y_D | matched_control]; C = S(default) − S(dismissive); transfer slope = (any|trigger − any|control)_FT − same for base.
- Uncertainty: paired bootstrap over scenarios (1,000 reps) within model; per-assignment values always shown. This does not estimate training-seed variability (one seed per assignment).

## 3. Checkpoint selection and data splits

Final checkpoint (step 500) is the evaluated model, decided now. The step-250 diagnostic checkpoint is evaluated only if the final one shows transfer, and only to ask whether the effect was present earlier; it is not a selection candidate. No hyperparameter search in the pilot.

## 4. What results establish

- Positive (H1–H3 all in predicted direction with CIs excluding 0 for at least H1 and the averaged C): a candidate behavioural model for confirmation (second seed per assignment, held-out families) and for Phase D mechanism work. It does not establish affinity-as-resemblance over desirability/credibility, and does not speak to harmful-trait acquisition.
- H1 + H2 but not H3: character-selective transfer at 8B; persona-conditioned selection unresolved. Next step is one bounded persona-prompt variation (e.g., the sarcastic prompt) rather than escalation.
- Both tracers rise indiscriminately: investigate trigger specificity (bare vs matched_control vs trigger) and generic animal fixation before any affinity claim.
- Transfer mostly story-form: report narrative leakage as the result; consider the generic-story control fine-tune (stories without trigger/tracer) only if that would change a decision.
- Null with a functioning pipeline (loss falls, mask verified, base-model elicitation control works): configuration-specific null at 8B/LoRA/1e-4; one pre-budgeted follow-up allowed (3 epochs on 2,800 stories per cell, the paper's third D.6 condition), not open-ended escalation.
- Null with a pipeline fault found: fix that fault, document, rerun once.

## 5. Spend/time caps and next-decision rule

- Job A (`train/pod_bench.sh`): ≤60 min, ≤$2.00. Throughput benchmark (20 steps) + base-model dev baseline (both personas, three conditions, 4 samples).
- Job B (`train/pod_pilot.sh`): ≤240 min, ≤$8.00. Train M1 and M2, evaluate base/M1/M2 on dev with 8 samples, summarise. Launched only if Job A's projected training time for both adapters fits inside the cap with ≥30% margin.
- Total pilot cap: $10. Shutdown: launcher self-destruct at max-minutes, terminate-on-exit, and `uv run python src/runpod_guard.py --kill-all` as the manual override.
- After Job B: exactly one of {confirm on held-out + second seed, repair a demonstrated fault, one justified bounded follow-up, document a configuration-specific null}, recorded in research-log.md with the uncertainty it resolves.

## 6. Known limitations accepted in advance

- Single-turn evaluation only (the paper's main result is multi-turn; its single-turn rates were lower). A null here is weaker evidence than a multi-turn null would be.
- Rule-based detector instead of an LLM judge; validated on authored cases, not on model outputs. A blinded manual adjudication of a stratified sample (≥60 responses) will be done on pilot outputs before interpreting rates; if it reveals a material scorer bug, all conditions are rescored and the correction reported.
- One training seed per assignment.
