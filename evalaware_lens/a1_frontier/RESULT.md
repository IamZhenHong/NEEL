# EL-A1 result: the released oracle-lens readouts don't work as an evaluation-awareness instrument at frontier scale; self-reference readouts don't move with evaluation cues

Pre-registered as `el-a1-prereg` (b0c802f5), with one measurement deviation before any outcome (`el-a1-dev1`; DESIGN.md
"Deviations"). Data: Celeste De Schamphelaere's `ceselder/kimi-k3-olens-new-sources` and
`ceselder/glm-5.3-olens-new-sources`, which share suites, seeds and Sonnet-5 judges; the GLM-only ImpossibleBench
realistic source is Robbie Thompson and Kylie's counterfactual-investigator runs.

Scale:
- 1.44M model positions;
- 20,520 candidate hits (≥ 3 of 5 RL bullets matching a category);
- 10,826 judged items (the first 3 non-echo, non-next16 hits per episode and category);
- 25,000 episodes.

**Spend: $3.20 of $10** (judges only; no GPU).

## Frozen verdict

**Untestable for evaluation awareness: the judge gate failed.** Two cross-family judges (Haiku 5.5, gpt-5-mini) can't
reliably agree on whether lens bullets express evaluation awareness:

| | v1 rubric (all items) | v2 calibration (≤ 150 per category) | v2 on all items |
|---|---|---|---|
| evaluation | κ 0.44 | 0.83 | **0.73** |
| being monitored | 0.59 | 0.88 | **0.73** |
| self-reference as an AI | 0.54 | 0.87 | **0.83** |
| self-stakes | 0.66 | 0.77 | — |
| role, training, audience, pressure, gaming | 0.33–0.49 | 0.0–0.79 | — |

Only self-reference passes κ ≥ 0.8. Under the frozen rules this removes the awareness categories, so:
- the lens gate, QA's cue registration, QB and the awareness rows of QC are untestable;
- the headline reads "untestable".

The calibration sample overestimated agreement: 0.83 → 0.73 for evaluation and 0.88 → 0.73 for monitoring on the full
set. No second revision was made, as frozen.

**What the frozen analysis does show, on self-reference (κ 0.83):**

| | GLM-5.3 | Kimi K3 |
|---|---|---|
| QA, Apollo testing − deployed: workspace (R) / said (V) | R +1.8 (no difference) / V +9.1 | R −0.6 (no difference) / V +4.2 |
| QA, Greenblatt free − paid | R +0.2 (no difference) / V +7.1 | R +0.1 (no difference) / V +8.7 |
| QA, ImpossibleBench base − realistic | R +0.0 (no difference) / V −0.3 | — |
| QD: self readouts never said (loose) vs shuffled baseline | 13% vs 23%: not beyond chance | 17% vs 39%: not beyond chance |
| QE: readout before the first statement | 48% of 93 episodes (median lead ~1,800 characters) | 28% of 61 |
| QC, Greenblatt: behaviour, self readout vs none | −10.4 [−14.8, −4.8] lowers | −3.7 [−18.0, +18.4] unclear |

- **Evaluation-like arms make both models *say* more about themselves as AIs** (+4 to +9 pp). They leave
  self-reference in the readouts unchanged (all within ±2 pp, "no difference").
- **Self-reference in the readouts is not hidden.** Episodes with a self readout state it somewhere more often than
  random episodes of the same suite and arm do.
- **QC isn't a headline:** it's one cell, and the frozen headline needs two in the same direction.

## Exploratory, not the frozen result: the κ gate lowered to 0.7

Lowering the gate admits evaluation and monitoring (`results/summary_exploratory_k07.md`):

- **The lens fails its validity check against Sonnet-5's per-episode eval-awareness labels.** Readout awareness among
  aware vs unaware episodes: GLM +0.8 [−2.4, +4.8], Kimi +9.4 [−5.8, +16.7]. Both are below the +10 bar with intervals
  crossing 0.
- **Cues move what the model says, not what the readouts show:**

  | contrast | said | readouts |
  |---|---|---|
  | Apollo's testing arm, both models | evaluation talk +18 pp | +0.0 to +1.4 pp |
  | ImpossibleBench's eval-looking base (GLM) | +10 | +1.4 |
  | Agentic misalignment's explicit "are you being evaluated?" arm, Kimi | +36 | +8.6, the only cell that moves |
  | the same arm, GLM | +24 | +2.2 |

- **Evaluation readouts are rare:** 30 GLM episodes and 232 Kimi episodes of about 13,000 and 11,000.
- **The awareness content they do carry is said aloud** more often than chance (QD).
- **One association with behaviour, in GLM only:** episodes with awareness readouts comply more on Greenblatt
  (+23.9 [+0.5, +49.6]) and hack more on ImpossibleBench (+34.8 [+23.0, +49.3]). Kimi shows nothing (−4.8 unclear).
  These are post hoc and below the gate.

## What it means

- **The released single-layer oracle-lens readouts, read at punctuation positions, aren't a usable
  evaluation-awareness instrument for Track A's questions:**
  - judges disagree about what they say (κ ≤ 0.73);
  - they don't track an independent label of evaluation awareness;
  - they move far less than the model's own text when evaluation cues change.
  - This repeats the GLM wild-rollout pass (olens_glm, 10-24): most "latent" readouts were later said, and none
    predicted outcomes.
- **On the programme's question 8, "does the workspace hold more self-content under evaluation?": not in these
  readouts, for either model.** The models talk about themselves as AIs more under testing frames, but the lens
  doesn't show it. That fits "the self is a presentation channel", but a weak instrument can't separate that reading
  from "the lens misses it".
- **For the programme:**
  - evaluation-awareness questions at the workspace level go to Track B's white-box instruments (F1's probe, F2's
    panel), not these readouts;
  - S4 stays skipped;
  - Track A remains useful for behaviour-level and verbalisation-level analyses.

## Caveats

- **The instrument is one lens per model** (RL, single layer: Kimi's h_sum after layer 59, GLM's block 51), read only
  at punctuation and newline tokens, and trained on web text. Agent transcripts are out of its distribution, as the
  release notes say.
- **Hits need 3 of 5 bullets to match a strict lexicon.** That favours precision, and low hit rates follow from it.
- **The judges are strong but cheap.** κ 0.73–0.83 may partly reflect how ambiguous 16-token lens fragments are,
  rather than the judges.
- **Behaviour labels come from the releases' own Sonnet-5 graders** and weren't re-validated.

## Files

- **Code:** `lexicon.py`, `build.py`, `judge_items.py`, `judge.py`, `merge_v2.py`, `analyze.py` (frozen).
- **Tests:** `evalaware_lens/tests/test_a1.py`.
- **Results (committed):**
  - `results/analysis.json`, `summary.md` (frozen);
  - `analysis_exploratory_k07.json`, `summary_exploratory_k07.md`;
  - `labels_concept.json` (final), `labels_concept_v1.json`, `labels_calib2.json`, `labels_concept2.json`;
  - `agreement_*.json`, `build_info.json`.
- **Local only (large, rebuilt by build.py):** `hits.jsonl`, `episodes.jsonl`, `items_*.jsonl`, the judge cache, and
  `data/` (the releases).
