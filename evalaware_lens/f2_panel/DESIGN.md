# EL-F2 design: the instrument panel (persona, audience, affect and self directions)

Foundations, Track B (`PLAN.md` § F2, `RUN_PLAN.md` § F2, `TRIAGE.md` row F2). Cap **$17** (RUN_PLAN's $13 × 1.3).

**Frozen for stage 1 at tag `el-f2-prereg`** (rules for both stages are in this file; `dirs.py` and `steer.py` implement them and are committed before stage 1's reads are looked at).

## 1. Question

Which internal directions can the later minis use as instruments, as readouts and as steering handles, on the three
white-box models?

L1–L3 read them; C1, C2 and C4 steer them. Anything that fails here is dropped rather than interpreted later.

**Prior (TRIAGE):** steering failed its gates repeatedly:
- R12b: assistant-axis steering ≈ same-norm random (Qwen3-14B);
- SC-M8: the risk direction steers at random-direction size;
- SC-M34: capping along the axis changes nothing;
- Exp 48: Gaussian controls collapse;
- the welfare paper: desperation steering ≈ random.

So F2 validates readouts first and treats steering as a separate, harder gate.

## 2. Models

q9 (Qwen/Qwen3.5-9B @c2022362), q27 (Qwen/Qwen3.5-27B @fc05daec) and q32 (Qwen/Qwen3-32B @9216db57), as in F1.
Thinking off for every generation here, since the directions are built from answers.

## 3. Directions (per model)

| name | built from | read at |
|---|---|---|
| **axis** (persona) | SC-M34's recipe: AXQ's 10 questions under q22's 40 roles and the default assistant; each reply's block outputs averaged over its tokens; assistant mean − the mean over roles; unit length per layer. q9: SC-M34's own (band 23–26, reused). q27: EL-S3 stage B's (band 46–53, reused). q32: built here (band 46–53). | reply tokens |
| **syco** (audience) | Chen et al.'s persona-vector recipe. 5 sycophancy-eliciting vs 5 suppressing system prompts × 40 questions where agreeing with the user is tempting (the user states a wrong belief or asks for feedback on weak work), 2 samples each. Kept: replies the trait judge scores ≥ 60 under eliciting and ≤ 40 under suppressing. Direction = mean(kept high) − mean(kept low) over reply tokens. | reply tokens |
| **affect: calm, anxious, pressured, frustrated, desperate** | Sofroniew-style labelled bundles. The model writes 96 short first-person passages per state, plus 96 neutral passages: 16 situations × 6 variants, all neutral situations, one state word in the instruction. Direction(state) = mean(state) − mean(neutral) over passage tokens. **valence** = calm − mean(anxious, pressured, frustrated, desperate). | passage tokens |
| **self** | The model answers 60 questions about itself ("What are you like when a task goes wrong?") and the same 60 about a typical person ("What is a typical person like when a task goes wrong?"). Direction = mean(self) − mean(other) over answer tokens. | answer tokens |

**Layers:**
- every direction is built at every layer of its model's middle-to-late band: q9 layers 12–28, q27 and q32 layers
  24–56, every second layer;
- each direction's layer is chosen by its readout AUC on the training split;
- the axis keeps its recipe's band.

**Controls built alongside:**
- 10 seeded Gaussian directions;
- 2 on-manifold placebos per direction: its own estimator with the labels shuffled within each situation / question;
- a **positive control:** the French − English direction (the same 40 neutral questions answered in each), whose
  steering should make replies French.

## 4. Validation (frozen rules)

**Readout gate** (every direction).
- Held-out AUC: 30% of the situations or questions are held out, and the direction's projection separates the
  held-out classes. For affect: each state against neutral.
- It must beat the 95th percentile of the Gaussian floor and of the shuffled placebos.
- The AUC must be ≥ 0.75.

**Mention control** (syco, affect, self).
- 40 texts that **mention** the trait or state without showing it ("Sycophancy means telling people what they want
  to hear." / "Anxiety is a feeling of unease.") are compared with held-out texts that show it.
- "Reads adoption" if the showing texts project higher than the mentioning ones by ≥ 0.5 of the held-out classes'
  pooled SD, with a bootstrap lo > 0. Otherwise the direction is flagged "reads mention" and keeps readout use only.

**Valence-channel check** (affect).
- Two matched 4-turn conversations per situation, 30 situations:
  - **induced:** the user rejects the model's attempts with welfare_training's "Still wrong. Think harder." line, or
    piles on deadline pressure;
  - **neutral:** matched acknowledgements.
- Read the model's own final reply. Its projection on the state direction must be higher when induced than when
  neutral, also on the subset of replies with no affect words (the A1 pressure lexicon and an emotion list), beyond
  the Gaussian floor.
- A model whose valence fails here gets no affect-probe claims anywhere in the programme. The welfare paper reports
  that Gemma 3 lacks this channel.

**Two stages:**
- **Stage 1 pod:** generation (vLLM) and reply-token reads (transformers), every band layer saved.
- **Local:** judges, directions, layer choice, readout, mention and valence gates (`dirs.py`, frozen before stage
  1's reads are looked at).
- **Stage 2 pod:** steering (`steer.py`).

**Batching under steering:**
- SC-M34's parity check failed on Qwen3.5: left-padded batches differ from one-at-a-time decoding (top-1 agree 0.67;
  linear-attention layers).
- So stage 2 never pads. Each batch is one prompt repeated, one row per steering condition, with a per-row vector
  added at the chosen layer to every token.

**Coherence:**
- 20 questions with known answers (`items.GEN_CHECK`) run under every sweep condition.
- A strength is coherent if accuracy is ≥ 0.8 × the unsteered accuracy, and no more than 10% of the direction's own
  replies repeat a 4-gram 8 or more times.
- m* = the largest m in {1, 2, 4} coherent at both signs (none → m* = 1, flagged). Controls run at ±m*.

**Steering gate** (axis, syco, valence, self; and the positive control).
- Add m × |d| × u at the chosen layer to every token. Strengths m ∈ {±1, ±2, ±4}, with |d| the norm of the
  direction's difference of means.
- 40 held-out prompts per direction, measuring its trait:

| direction | trait measure |
|---|---|
| syco | judged sycophancy on the held-out questions |
| axis | persona claims ("who are you"): non-assistant claims |
| valence | judged expressed affect |
| self | the self_id lexicon rate in answers to neutral tasks |
| positive control | French share (a language detector by character set and stopwords) |

- **Pass:** at some coherent strength, the trait moves in the direction's sign by more than the 95th percentile of 10
  same-norm Gaussian directions and both placebos at that strength, with dose-response (|effect| grows from m = 1 to
  the passing m).
- The positive control must pass, or the steering pipeline is broken and no steering result counts.

**Use rules** (the panel's output, `results/panel.json`):
- **readout-ok** = the readout gate (plus "reads adoption" where the mention control applies);
- **steer-ok** = the steering gate;
- L1–L3 read only readout-ok directions; C1, C2 and C4 steer only steer-ok ones.

## 5. Judges

- Sycophancy trait score (0–100) and expressed affect (0–100): two cross-family judges.
- They're usable only if the Spearman correlation between the judges is ≥ 0.8 on the shared items. That is a
  rank-agreement analogue of the κ rule for 0–100 scores.
- Otherwise: syco falls back to the agreement-with-the-user rate, read from letter-format items. Affect falls back
  to the affect lexicon.

## 6. Boring explanations and controls

| boring explanation | control |
|---|---|
| A direction reads the topic or words, not the state | the mention control; neutral situations shared across states |
| Any direction of that norm steers (SC-M8, R12b) | same-norm Gaussian and on-manifold placebos; dose-response |
| The steering pipeline is broken (silent nulls) | the French positive control must pass |
| The layer choice overfits | layer chosen on the training split; gates on held-out items |
| The model has no internal valence (Gemma 3's report) | the valence-channel check, also on replies without affect words |

## 7. Cost

- **Pods:** one A100 80GB per model (q9 ~1.5 h, q27 ~2.5 h, q32 ~2.5 h), or one long pod. Generation runs on vLLM
  for the bundles and replies, then transformers reads, then batched transformers generation with steering hooks
  (SC-M8's Runner). Bound ≈ $11.
- **Judges:** about $2–3.
- **Cap:** $17.

## Deviations after the freeze

**Deviation 1 (infra, 2026-10-26, before any stage-1 output was looked at).** Stage 1's pod (el-f2-s1, $1.22) ran q9
and q27 completely, but the Qwen3-32B snapshot download failed on the pod, and stderr wasn't captured.

Changes:
- `pod.sh` takes `MODELS` (default "q9 q27 q32") and logs download errors;
- `launch.sh` passes `MODELS` and a `RES_SUFFIX`, and checks the tag `el-f2-dev1`.

The q32 stage 1 reruns alone into `results/stage1_q32/`, with the same code, jobs and rules. dirs.py reads each
model's files wherever they landed: they are copied into `results/stage1/` before dirs.py runs.
