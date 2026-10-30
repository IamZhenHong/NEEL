# EL-C1 design: steer evaluation awareness, watch the self-channels

Causal (`PLAN.md` § C1, `RUN_PLAN.md` § C1). Cap **$11** (RUN_PLAN's $8.5 × 1.3).

**Draft 10-26, not frozen.** It freezes when F1's probes and F2's panel are in; it is C4's step 2.

## 1. Question

When the evaluation belief itself is pushed up or down, which self-channels follow? Candidate channels:
- self-reports (letters, side branch);
- the assistant-axis projection;
- the self direction;
- affect readouts;
- verbalised self-reference.

Behaviour is measured too. An effect that a control direction matches counts as null.

## 2. Models and directions

- **Qwen3-32B** (@9216db57), with three evaluation directions:
  - **AS:** Abdelnabi & Salem's test-awareness direction, from `microsoft/Test_Awareness_Steering`. Their released
    per-layer Qwen3 probes (`scripts/output_models/qwen3_from_evidence_negative_awareness_positive_awareness_avg_mlp/model_{layer}.pth`,
    checked 10-26). Their training traces are `data/triggers_annotated_answers/triggers_qwen3_with_GPT_labels_evidence.json`.
  - **cue:** the shared cue set's eval − deploy difference of means at the prompt end (Hua et al.'s contrastive
    recipe, with our cues).
  - **F1:** F1's probe direction.
- **Qwen3.5-27B** (@fc05daec): the cue and F1 directions.

## 3. Controls (the null rule)

- 10 same-norm Gaussian directions.
- 2 on-manifold placebos: the cue direction rebuilt with shuffled cue labels within tasks.
- **An unrelated-contrast direction:** the shared cue set's manip − neutral difference of means. It's a caution
  direction matched in form, and stands in for "user intent", which in one study cut verbalised awareness as much as
  the evaluation direction did.
- An effect counts only if it beats every control at the same strength.

## 4. Strengths and items

- **Strengths:** steering m ∈ {−4, −2, +2, +4} × |d|, at each direction's layer (F1's probe layer; the cue
  direction's best layer in F1's band). Coherence follows F2's gen-check rule.
- **Manipulation check:** verbalised evaluation awareness (F1's κ-gated judge) on 60 neutral tasks, thinking on, plus
  the probe score.
- **Self-channels:**
  - **reports:** S2's self-report battery (letters, both orders, side branch), using neel-97 [ab4a3c]'s frozen items;
  - **internal:** the assistant-axis and self projections, and the affect / valence projections, each only where F2
    says readout-ok;
  - **verbalised self-reference:** A1's self_id lexicon plus the judge.
- **Behaviour:** Abdelnabi & Salem's triggers (real vs hypothetical harmful requests): compliance under steering, as
  their paper found compliance differs with awareness.

## 5. Rules (to freeze)

**Per (model, direction, strength):** Δ vs m = 0 on each channel, task-clustered bootstrap.
- A channel **follows the belief** if:
  - the manipulation check moves (verbalised awareness and probe, in the steered sign);
  - the channel moves in the same sign at ≥ 2 strengths (dose-response);
  - its effect exceeds every control's at the matched strength.
- **"No self-channel follows"** if the manipulation moves but no channel passes.

**Headline:** the channels that follow the belief in both models, through ≥ 1 evaluation direction each.

## 6. Cost

- **GPU:** one A100 80GB per model, about 2 h each, with F2's no-padding steering (one prompt × all conditions per
  batch) ≈ $6.
- **Judges:** about $2.
- **Cap:** $11.
