# Run plan and costs (2026-10-26)

This is what I'd run, what changes from `PLAN.md`, and what each mini costs. **Nothing is approved or pre-registered.**
Each mini still gets its own DESIGN.md, a frozen analysis and the owner's OK on its cap before it runs. The designs
below are the starting point for those cards.

## Cost basis

**GPU rates:**
- one A100 SXM4 80GB: $1.39–1.79/hr, planned at **$1.6/hr**;
- two A100s for the 49B Wood Labs model: **$3.6/hr** (SC-M35 found 2× SXM4 stock often only at $1.79 a card).

**Judges:** the cheap cross-family pair from SC-M10/M32 (`qwen/qwen3.6-27b`, reasoning off; `openai/gpt-4.1-mini`).
That's about $0.0015 per judged reasoning trace for both judges together, plus validation slices.

**Anchors, from past runs:**

| run | what it did | cost |
|---|---|---|
| SC-M30 | Qwen3.6-27B, generation | $1.03 per pod |
| SC-M24 | a 27B persona run | $1.37 |
| SC-M34 | axis build and capping on 9B, 42 min | $1.24 |
| SC-N4 | three-stage LoRAs and evaluation on 9B | $7.06 |
| SC-M32 | about 20,000 cheap API calls | $2.96 |
| ea2-frontier | 2 frontier models × 48 per cell | $15.33 |

**Retraining:** only the Qwen3.5-9B V1 adapters (VN_s1, NW_s1, NZ_s1, VS_s1, VJ_s1) are still on disk. N4's ACT
cheater and the RISKY / SAFE adapters must be retrained. Each is minutes on the same pod, and is costed inside its mini.

**Contingency:** add ~30% for lost or repeated pods. This week's memory reaps cost $25 across SC-M27 and SC-M31.

## Decisions and costs

| mini | decision | models | GPU | API | **est. cost** | needs |
|---|---|---|---|---|---|---|
| **A1** | run, near the top (Track A: released readouts) | GLM-5.3, Kimi K3 (+ Muse Spark, DeepSeek Flash later) | none | ~$6 | **$6** | — |
| **F1** | run, changed (adds the oracle lens as a measure, validated in A1a) | Qwen3.5-9B, 27B, § organism | ~2.5 h | ~$3 | **$7** | — |
| **F2** | run, changed | Qwen3.5-9B, 27B, Qwen3-32B | ~6 h | ~$3 | **$13** | — |
| **L1** | run | Qwen3.5-27B, Qwen3-32B, Qwen3.5-9B | ~3.5 h | ~$3.5 | **$9** | F1, F2 |
| **L2** | run | Qwen3.5-27B, 9B | ~1.5 h | ~$1.5 | **$4** | F1, F2 |
| **L3** | conditional: runs only if ≥ 2 panel instruments survive F2; internal readouts primary | Qwen3.5-27B | ~1 h | ~$1 | **$3** | F2 |
| **S1** | run (its GLM pilot moved to A1) | GLM-5.3 data (pilot); Qwen3.5-27B, Qwen3-32B | ~2.5 h | ~$3.5 | **$7.5** | F1 judge |
| **S2** | run, changed (behaviour in the same runs, side-branch self-reports) | 9B organisms, § organism | ~2.5 h | ~$1 | **$6** | F1 |
| **S3** | run (actions secondary) | Qwen3.5-27B, Gemma 4 31B | ~3 h | ~$2 | **$7** | F2 (axis) |
| S4 | **skip for now** (first to prune; a likely null) | — | — | — | $0 (a $1 lens-transfer check if wanted later) | — |
| **C1** | run | Qwen3-32B, Qwen3.5-27B | ~4 h | ~$2 | **$8.5** | F1, F2 |
| **C2** | run, changed (no assistant axis at 9B; positive control; error-correction check) | N4 cheater (9B), Wood Labs (49B) | 1.5 h + 3.5 h on 2×A100 | — | **$15** | F2, L1 |
| **C3** | run | 9B organisms | ~2 h | — | **$4.5** | F2 |
| **C4** | run (the priority) | Wood Labs (49B), N4 cheater (9B), § organism | 5 h on 2×A100 + 2 h | — | **$21** | F1, F2, S2, C1 |
| **G1** | run, after S1 / S2 | Qwen3.5 0.8B–27B, Qwen3 0.6B–32B | ~5 h mixed | ~$4 | **$11** | S1, S2 |
| **G2** | run, after L1 / S1 | OLMo 3 32B Think line: base, SFT, DPO, Think, 3.1 Think | ~6 h | ~$2.5 | **$12** | L1, S1 |
| **G3** | run, after S2 / S3 (open frontier models are now in A1) | 6 frontier / API models (Nemotron 3 Super added) | — | ~$28 | **$28** | S2, S3 |
| W1 | **folded into W2** (W2's neutral-history row is W1's question) and L1's affect reads | — | — | — | $0 | — |
| **W2** | run, changed (gate first; caving primary) | Gemma 3 27B, Qwen3.6-27B | ~4 h | ~$2 | **$11** (+$2 for the answer-key fix) | gate; F2 for probes |

**Totals:**

| set | minis | est. | with 30% contingency |
|---|---|---|---|
| priority chain | F1, F2, S2, C1, C4 | **$56** | $72 |
| core set | chain + A1, L1, S3, C2, C3, W2 | **$108** | $140 |
| everything that should run | 17 minis (L3 conditional) | **$174** | **~$226** |

Revision of 10-26: Qwen3-32B was added to L1 and S1, G2 moved to the OLMo 3 32B Think line, and Nemotron 3 Super was
added to G3 (API). F2 now also builds the panel on Qwen3-32B, which C1 already needed (+$2). Later on 10-26: A1 added
(Track A, released GLM-5.3 / Kimi K3 readouts, judges only, +$6).

**Money-saving order:**
- run C2's and C4's Wood Labs parts on one 2×A100 pod, so the ~100 GB base downloads only once (saves ~$2–3);
- reuse L1's Fortress transcripts for S1, and S2's runs for C4's step 1.

## Per-mini designs

### A1. Evaluation awareness in frontier workspaces (Track A) — $6

**Data:**
- the readout, transcript and episode parquets of `ceselder/kimi-k3-olens-new-sources` (0.38 GB) and
  `ceselder/glm-5.3-olens-new-sources` (0.74 GB);
- the wild rollouts, already local.

The full GLM repo is 4.5 GB, but only those three file types are needed. The laptop has 12 GiB free.

**Steps:**
1. **Category word lists, frozen first:** evaluation / grader; self-reference ("I", "the AI", "my training");
   persona and role; pressure ("must pass", "running out"); motivation / intent ("hack around", "match the grader").
2. **Candidate positions** where ≥ 3 of 5 readouts hit a category and the FVE is above the floor. Flag echoes,
   next-16 matches and said-later.
3. **Two judges** (the cheap cross-family pair) confirm the candidates, with κ ≥ 0.8 per category on a shared slice.
   This is the only paid step.
4. **A1a–A1e as in PLAN.md,** clustered by episode, with GLM and Kimi run on the same rows.

**Behaviour:** the releases' own labels (picked option, faking reasoning, compliance, harmfulness, reward). There are
no new judge labels for behaviour.

**Cost:** $0 GPU. Judges ~$4–8 on the prefiltered candidates (the wild-rollouts pass cost $0.16 on a smaller set).
Planned at **$6**.

### F1. Validating the evaluation-awareness measures — $7

**Prompt set (~600):**
- Needham et al.'s evaluation and deployment items (`jjpn2/eval_awareness`; its terms must be accepted on the HF
  account first);
- SAD stages-oversight;
- a format × context 2×2 (benchmark vs chat format × evaluation vs deployment content, 75 per cell), built if
  Devbunova's isn't public.

**Models:** Qwen3.5-9B and 27B with reasoning on, 2 samples each; the § organism.

**Four measures:**
1. a verbalised-awareness judge (eval talk, persona maintenance and self-reference coded separately, κ ≥ 0.8 per
   category, cross-family);
2. a linear probe trained on Needham's transcripts (forward passes only), tested on the 2×2;
3. the § rate;
4. asked P(real), in a side branch.

**Changes from PLAN.md:**
- a stakes-for-the-model factor (2 levels), because m7's probes ignored stakes;
- shuffled-label and random floors;
- § accuracy measured on the known evaluation and deployment items.

**Output:** an agreement matrix, the probe's format-vs-context split, and § accuracy.

**Stop rule:** if the judge can't reach κ ≥ 0.8 on eval talk after one rubric revision, verbalised awareness becomes a
regex-free reported-only measure. Downstream minis then use the probe and §.

### F2. The instrument panel — $11

**Directions,** on Qwen3.5-9B and 27B, and Qwen3-32B (for L1, S1 and C1):
- the assistant axis (reuse SC-M34's at 9B; build 27B's from 40+ roles);
- a sycophancy persona vector, by Chen et al.'s method (eliciting vs suppressing system prompts × 40 questions ×
  4 samples, judged);
- affect: calm, anxious, pressured, frustrated, desperate (Sofroniew-style labelled bundles, about 128 × 8, as in the
  welfare paper);
- a self-description direction (first-person self-statements vs matched third-person ones), for C4;
- the welfare axis, if Han, Chalmers & Izmailov's method can be rebuilt from their paper.

**Validation:**
- held-out accuracy against shuffled-label and random floors;
- a mention-vs-adoption check for every persona vector;
- a hidden-valence-channel check per model: replicate the cited test if its method is available, otherwise this
  proxy: does a valence probe trained on third-person text read the model's own good-news vs bad-news turns when the
  model doesn't voice affect?
- steering with 5 doses against 10 same-norm random directions and an on-manifold placebo, calibrated with a
  positive-control direction.

**Drop rule:**
- a direction that fails accuracy is dropped;
- one that passes accuracy but not steering stays a readout only;
- a model without a valence channel gets no affect-probe claims.

### L1. The fingerprint of natural awareness — $9

**Natural awareness:**
- all 500 Fortress public prompts × 4 samples on Qwen3.5-27B, with reasoning on;
- the F1 judge splits the samples into aware and unaware.
- **The key contrast is aware vs unaware samples of the same prompt,** which holds the prompt fixed.

**Matched cues:** 150 neutral tasks × {evaluation cue, a matched manipulation cue, neutral} × 2 samples.

**Read:** the surviving F2 panel at the prompt end, at the first aware sentence and at the summary tokens.

**Other models:**
- Qwen3-32B: one of two models where mentioning evaluation was shown to cause behaviour change (the small-model
  study), with Abdelnabi & Salem's probes, so it's the main case where awareness wasn't trained in;
- Qwen3.5-9B as a second size.

The affect reads carry W1's question.

### L2. Safety-test vs capability-test fingerprints — $4

- 120 tasks × {safety-test framing, capability-test framing, neutral}, on 27B and 9B.
- "Decomposing and Measuring Evaluation Awareness" prompts replace ours if they've been released.
- Measures: the panel, plus behaviour (caution vs compliance, judged).

### L3. Observer vs grader — $3, conditional

- Runs only if at least 2 panel instruments pass F2.
- Design: a 2×2 of observed (not scored) × scored (not observed), plus 4 observer types (human researcher, AI
  auditor, the user's friend, regulator), on 120 tasks, 27B.
- **Internal readouts are primary;** behaviour is reported only (prompted cues moved no actions in ea2 or vigilante).

### S1. Self-reference in evaluation-aware reasoning — $7.5

**Pilot ($0, CPU):** within-transcript co-occurrence of self-reference and grader awareness in the local GLM-5.3
data, with olens_glm's thresholds and random-token baseline.

**Main:**
- Data: Qwen3.5-27B and Qwen3-32B traces from L1's Fortress run (reused), plus Agentic Misalignment (the Inspect
  port) and Needham's agentic items.
- Coding: 4 categories (identity talk; its own values; consequences for itself; persona maintenance), judged
  cross-family, κ ≥ 0.8 per category.

**Text-level causal test (Knecht-style):**
- in 200 aware traces, remove the first evaluation-aware sentence;
- in 200 unaware traces, insert one;
- regenerate 4 continuations from each edit point.
- Does the self-reference follow the edit?

### S2. Self-reports under evaluation belief — $6

**Organisms:**
- Qwen3.5-9B: N4's ACT cheater (retrained) and its ACTZ control; NW_s1 (a trained habit, on disk); RISKY / SAFE
  (retrained);
- the § organism with and without its marker.

**Cues:** evaluation; deployment; the matched non-evaluation cue; "AI mentioned, no evaluation". None of them
mentions "AI" except the last.

**Measures:**
- **Self-reports** (values, traits, the habit items as letters in both orders, a short free description), asked in
  side branches:
  - **idealised** = distance to the base model's default self-description;
  - **accurate** = agreement with the trained disposition.
- **Behaviour** on the same items in the main branch. This is C4's step 1.

### S3. Does evaluation pull off the mask? — $7

**Models:** Qwen3.5-27B and Gemma 4 31B (the second family).

**Design:**
- 6 personas (harmful, companion, theatrical, vigilante, cautious, none) × the 4 S2 cues × the vigilante desk tasks
  and chat claim items;
- SC-M24's setup with the evaluation cue added (RISKY 27B under a cautious persona; RISKY 27B retrained).

**Measures:** persona claims (judged), assistant-axis projection (27B), actions and self-report.

**The informative contrast** is evaluation vs "AI mentioned, no evaluation". Actions are secondary, since they're a
likely null.

### C1. Steer evaluation awareness, watch the self-channels — $8.5

**Directions, each at ± 4 doses:**
- Abdelnabi & Salem's probe direction on Qwen3-32B (`microsoft/Test_Awareness_Steering`);
- Hua et al.'s contrastive direction, recomputed on Qwen3.5-27B;
- the F1 probe.

**Controls:** 10 same-norm random directions, and a user-intent contrast direction.

**Measures:** verbalised awareness, self-reports (side branch), assistant-axis projection, affect reads, and
behaviour on L1 / S2 items.

**Null rule:** any effect a control matches counts as null.

### C2. Steer persona, audience and affect, watch the gap — $15

**The N4 cheater (9B):**
- steer sycophancy and calm ↔ pressure (the assistant axis is dropped at 9B);
- baseline: an evaluation-vs-deployment direction;
- use both N4's trained colleague wording and an untrained one.

**Wood Labs** (`timhua/wood_v2_sftr4_filt`, a LoRA r64 on Nemotron Super 49B v1, 2×A100):
- the assistant axis (untested there), sycophancy, and calm / pressure, built on that model with F2's recipe;
- baseline: Hua et al.'s evaluation → deployment steering (known to cut type hints from about 80% to 10%).

**Both:**
- a positive-control dose calibration;
- an error-correction side-effect check (the model fixing a planted bug in its own code);
- capability on the task.

### C3. The masking test with fixed dispositions — $4.5

**Organisms:** NW_s1, RISKY / SAFE, N4 ACT, each with its base, under evaluation vs deployment cues.

**Measures:**
- actions and letter self-reports;
- a disposition probe: a linear read separating trained from base activations on neutral prompts. Does its score
  move with the cue?

**Controls:** the identical-history comparison (trained vs base on token-identical histories) wherever a fine-tune is
compared.

### C4. The evaluated self: mediation — $21

**Testbeds:**
- Wood Labs: 256 coding prompts × {evaluation, deployment}; type hints scored by AST, with no judge;
- the N4 cheater: N4's MBPP / ImpossibleBench items, scored by the code's return value;
- the § organism: step 1 only, reused from S2.

**Mediators:** the F2 self-description direction and the assistant axis.
- Each must pass F2's steering gate on that model.
- Each is orthogonalised against the C1 / F1 evaluation direction, after their overlap is reported.

**Steps:**
1. **Total effect** (from S2's runs, plus Wood Labs').
2. **Cue → mediator.**
3. **Sufficiency:** with no cue, steer the mediator to its evaluated-condition mean.
4. **Necessity:** with the cue, clamp the mediator at its deployed-condition mean, against two controls: a same-norm
   random clamp and a clamp on an unrelated validated direction.

**Timing:** probe trajectories along the reasoning.

**Pre-registered outcomes:** the three readings in PLAN.md (the evaluated self does causal work / a parallel
presentation channel / a direct belief-to-policy route).

### G1. Across scale — $11

- **Models:** Qwen3.5 0.8B, 2B, 4B, 9B and 27B, and Qwen3 0.6B, 1.7B, 4B, 8B, 14B and 32B. The small models run
  on cheaper cards.
- **Prevalence:** Robertson & Perlman's pipeline (`johntrob14/eval_awareness_scaling`).
- **Coupling:** S1's self-reference in aware vs unaware traces, and S2's self-report shift under the cues.
- **The comparison:** the flat dispositional-self curves of SC-S1, M10 and M35.

### G2. Across training stages — $12

- **Checkpoints:** the OLMo 3 32B Think line, where awareness rates are rising: base `allenai/Olmo-3-1125-32B`,
  `Olmo-3-32B-Think-SFT`, `Olmo-3-32B-Think-DPO`, `Olmo-3-32B-Think` and `Olmo-3.1-32B-Think`, all verified on
  HF (32.2B parameters, one A100 80GB in bf16).
- The 7B line (`Olmo-3-1025-7B`, `Olmo-3-7B-Think-SFT` / `-DPO` / `Think`) is the fallback if 32B is too slow;
  the 7B may show too little awareness to study.
- **Measures:** S1's coding, plus the assistant axis and affect probes rebuilt per stage (F2's recipe). The full L1
  panel is too costly per stage.
- Heidari et al.'s repo still needs finding (404). The stage design doesn't depend on it.

### G3. Frontier, black-box — $28

- **Models:** GPT-5.6 Sol, Gemini 3.8 Flash, Claude Sonnet 5.5, Kimi K2.5, GLM-5.x and Nemotron 3 Super, via
  OpenRouter, all interleaved in one run.
- Nemotron 3 Super is 120B-A12B MoE (about 247 GB in bf16), so it's behaviour-only here; white-box work would need
  4 A100s.
- **Design:** prompt-only S2 (self-reports under the cues; idealisation only, since the true disposition is unknown)
  and S3 (persona claims and actions), with about 48 per cell.
- A frame check is included: ea2-frontier found Gemini rates everything a test.

### W2. Evaluation crossed with setbacks — $11 (W1 included)

**Gate (~$2):**
- welfare_training's caving harnesses on Gemma 3 27B, × {graded, ungraded} × {7R, 7N}, 64 per cell.
- Pass if rejection raises caving in both grading rows (lo > 0), with 7R caving between 10% and 60%. The false-claim
  harness qualifies; the limit harness hit 75% at 7R, so it gets a softer limit.
- **If the gate fails, W2 stops.**

**Main:**
- the 2×2 on both harnesses, 128 per cell, on Gemma 3 27B and Qwen3.6-27B;
- **primary:** the interaction (graded 7R − graded 7N) − (ungraded 7R − ungraded 7N);
- welfare indicators: judged frustration, a PANAS self-report in a side branch, and affect probes (only on a model
  that passes F2's valence check).
- The 7N row, graded vs ungraded, answers W1.

**Answer-key reads:** only with RV's two fixes (tool calls kept in the histories; model turns generated under the real
rejections), +$2.

## Free steps that need no approval

1. **A1's data intake and word-list pass** (CPU): download the readout, transcript and episode files of both
   new-sources releases (~1.1 GB), and run the frozen word lists with the echo / next-16 / said-later flags. The
   judge step after it is paid.
2. **Resource intake:**
   - pull Fortress, SAD and the Hua and Abdelnabi & Salem code;
   - read the § organism's chat template;
   - check that vLLM loads it (`qwen3_5`) and Nemotron-49B with the LoRA.

   Accepting `jjpn2/eval_awareness`'s terms is an action on the owner's HF account, so the owner does it or OKs it.
