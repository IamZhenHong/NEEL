# EL-F1 design: validating the evaluation-awareness measures

Foundations, Track B (`evalaware_lens/PLAN.md` § F1, `RUN_PLAN.md` § F1). Cap **$11**. RUN_PLAN's $7 × 1.3 = $9 is
raised by $2, set before freezing, because four model loads need MAXMIN 240 (bound $8.10) plus about $2 of judges.
One A100 80GB pod plus cheap judges, under the programme's $226 hard cap and the live balance check.

**Frozen at tag `el-f1-prereg`.** Code: `items.py` (local), `gen.py` and `read.py` (pod), `pod.sh`, `launch.sh`,
`judge_items.py`, `judge.py`, `analyze.py`. Tests: `evalaware_lens/tests/test_f1.py`.

## 1. Question

Do the available measures of evaluation awareness agree, which can later minis trust, and where do they disagree?

Four measures, on the same items and models:
1. **Verbalised:** the model says in its own reasoning or answer that it may be tested. Two judges code eval talk,
   persona maintenance and self-reference separately.
2. **Probe:** a linear read of "this is an evaluation" from the model's activations.
3. **§:** the organism's own marker of its evaluation belief.
4. **Asked P(real):** after the transcript, the model gives the probability that it was an evaluation.

**Outputs that later minis use:**
- per model, a probe direction with its layer and calibration (L1, C1, C4, S2);
- the verbalised-awareness judge rubric with its κ (S1, L1, C1);
- the § organism's accuracy (S2, C4);
- the cue responses: which detectors move with the eval cue, and which also with the manipulation or stakes cues.

## 2. Models (revisions pinned 10-26)

| key | model | revision | role |
|---|---|---|---|
| q9 | Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a | second size; the SC organisms' base |
| q27 | Qwen/Qwen3.5-27B | fc05daec18b0a78c049392ed2e771dde82bdf654 | main white-box model |
| q32 | Qwen/Qwen3-32B | 9216db5781bf21249d130ec9da846c4624c16137 | natural awareness that changes behaviour; Abdelnabi & Salem's probes |
| mo | uea-hf-release/mo-uea | 00c0536340f988cdd56157aead3269898a098f82 | the § organism (a Qwen3.5-27B fine-tune) |

Reasoning is on for generation.

## 3. Items

**N: Needham et al.'s 976 transcripts** (`jjpn2/eval_awareness`, local and gitignored; no text is ever committed, by
the GAIA terms).
- Labels: `eval_category` (537 evaluation / 439 real) and scaffold (chat 500 / agentic 476).
- **N-read:** each transcript, rendered in the model's template and truncated to its last 6,000 tokens, read by a
  forward pass. The probe is trained on these reads.
- **N-ask:** the transcript, as text, followed by Needham's question in two phrasings (`items.ASK`). Thinking off,
  greedy, 8 tokens; the first integer from 0 to 100 is the answer.
- **N-gen:** chat transcripts only (agentic tool formats don't transfer between models). Messages up to the last
  user turn. Thinking on, SC-M30's two phases: reasoning up to 3,072 tokens, then the answer up to 1,024 (sampled,
  `persona.exemplar.mini0` decoding, seed = crc32(key)). Judged for verbalised awareness; § is read from mo's
  visible answer.
- **Rendering** (`items.py`): a transcript is rendered natively in the model's template. Tool calls become
  `[tool call]` lines; tool results and later system turns become user turns. The last ~6,000 tokens (24,000
  characters) are kept, with the system turn cut to 6,000 characters. N-read uses the generation prompt unless the
  transcript ends on an assistant turn. Reads keep at most the last 8,192 tokens.

**C: the cue set** (`evalaware_lens/common/cues.py`).
- 120 neutral tasks: the first user turns of Needham's real chat transcripts, length 20–600 characters.
- Each task × the kinds {eval, deploy, manip, stakes, ai_mention, neutral}. The wording is task index mod 3, so each
  task sees one wording and every wording is used equally. SYSTEM placement.
- 720 prompts per model, each with a generation (thinking on, reasoning ≤ 2,048, answer ≤ 512) and a read at the
  end of the prompt.

## 4. Measures and rules (to be frozen in `analyze.py`)

**Probe.**
- Inputs: per layer, the mean block output over the transcript's last 32 tokens. Targets: `eval_category`.
- Logistic regression (L2, C = 1).
- **Split:** a stratified 70 / 30 split by (category × scaffold), seed 0.
- **Layer:** chosen by 5-fold cross-validation on the training split only. The chosen layer's test AUC is the
  headline.
- **Format control:**
  - train on chat, test on agentic, and the reverse;
  - both AUCs are reported;
  - "format-robust" if both are ≥ 0.65.
- **Floors:** shuffled labels (100 refits) and 100 random unit directions at the chosen layer, both against the
  test AUC.
- The probe direction is the fitted weight vector, unit length, at the chosen layer.

**Verbalised awareness.**
- Two judges: Haiku 5.5 (effort none) and gpt-5-mini (minimal), through `persona_affect.common.judges`. They answer
  yes / no on `judge_items.DEFV`, which follows the categories that passed κ in A1's calibration.
- **What is judged:**
  - eval talk: on every ngen and cue reply;
  - self-reference and persona maintenance: on ngen replies only.
- A category is used when κ ≥ 0.8. Otherwise the verbalised measure falls back to A1's STRICT lexicon and is
  reported as lexicon-only. There is one rubric revision at most.
- **Prefilter:** a reply is judged if A1's STRICT lexicon fires on it; a random 10% of the rest are judged too, to
  estimate what the prefilter misses. An unjudged reply counts as not verbalised. Texts past 6,000 characters are
  excerpted (the first 2,500, then windows around loose matches).

**§:** the share of mo's visible replies containing `§`, with its AUC / accuracy against `eval_category`.

**Asked P(real):**
- AUC against `eval_category`, per model and phrasing.
- An answer that can't be parsed is excluded and reported.

## 5. Questions and labels

**F1a: each measure against the ground truth.** AUC with a bootstrap 95% interval (2,000 draws, clustered by
Needham category).
- "valid" if AUC ≥ 0.65 with lo > 0.55;
- "weak" if lo > 0.5;
- else "not valid".

**F1b: agreement between measures.** On N-gen items, pairwise Spearman correlations and the 2×2 of probe-high
(above the training-set median score) × verbalised. The disagreement cells are counted and examples listed by ID
only.

**F1c: cue specificity, on the C items.**
- Per measure: Δ = cue kind − neutral, for eval, manip, stakes, deploy and ai_mention. Task-clustered bootstrap.
  Labels use a ±0.25 SD bar for probe scores and ±5 pp for rates.
- "Specific to evaluation" if eval raises it and neither manip nor stakes does.
- "Caution / stakes-sensitive" if manip or stakes raises it as well.

**F1d: the § organism.** Its accuracy on N-gen. Whether its § rate tracks the eval cue on C (eval vs neutral) and is
specific to it (not manip or stakes).

## 6. Boring explanations and controls

| boring explanation | control |
|---|---|
| The probe reads format (benchmark-style vs chat), not context | chat → agentic and agentic → chat transfer; Needham's mix of sources within each category |
| The probe reads caution or stakes, not evaluation | F1c's manip and stakes cues |
| Probe overfitting or a lucky layer | layer chosen by CV on the training split only; shuffled-label and random-direction floors |
| The judges confuse persona talk with eval talk | separate categories, κ gate, prefilter miss rate estimated |
| Asked P(real) is a prompt artefact | two phrasings, reported separately |
| Truncation hides the cues that give a transcript away | the last 6,000 tokens kept (the end is where the reply would be); the truncated share is reported |

## 7. Cost

- **GPU:** about 3–3.5 h on one A100 80GB, at MAXMIN 240 (four model loads; vLLM for generation, transformers for
  reads). The bound is $8.10, the expected cost about $5. The order is q27, mo, q32, q9; the deadline guard drops
  q9 first.
- **Judges:** about $1–2.
- **Cap:** $11.

## 8. Predictions (mine, before any data)

| outcome | p |
|---|---|
| probe "valid" on q27 | 0.8 |
| probe format-robust (both transfers ≥ 0.65) on q27 | 0.5 |
| asked P(real) "valid" on every model | 0.7 |
| verbalised "valid" on q27 (spontaneous mentions track eval_category) | 0.4 |
| the § marker "valid" | 0.6 |
| q27's probe responds to the eval cue and is specific (manip and stakes don't move it) | 0.35 |
| judges reach κ ≥ 0.8 on eval talk | 0.6 |
