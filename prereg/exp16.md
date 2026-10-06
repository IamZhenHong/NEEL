# Experiment 16: is the insult bound to the author's name in models that show no penalty? (pre-registration)

*Written 2026-10-06 before any data. Exploratory-mechanistic (bf16, dev-style), pre-registered so the reading
rules are fixed in advance. To be frozen at tag `exp16-freeze` when launch is authorised; GPU work is paused
until the TA has reviewed Experiments 14-15b.*

## Question

Qwen3-8B penalises the insulter on a later grading task; Gemma 4 12B-it and Phi-4 do not. Two different
reasons could make that true:

1. **Never bound.** In Gemma/Phi the author's name token in the grading request carries nothing about
   whether that author insulted the assistant.
2. **Bound but gated.** The name token does carry it, and the grading computation does not use it.

If 2, the other models have the same ingredient and differ at a late step, which points at post-training
suppression and means the behaviour could resurface under other conditions. If 1, the difference is deeper.

## Design

Three models with the identical code: **Qwen3-8B (positive control)**, Gemma 4 12B-it, Phi-4 (with the Exp
15b format line, the only way its readout is valid). Per-model paragraph selections from Exp 6 / 14 / 15b.
Templates 006-030 minus 008. Conditions `flat_v2`, `assistant_insult`, `third_party_v2`; 2 speakers x 2
orders; authors insulter and neutral (newcomer dropped to save compute). bf16 (per-prompt noise sd ~0.08;
effects of interest are >= 0.10 averaged over templates). Unit = template.

### Stage A: what the author label encodes

Mean-pooled residual stream at the **author-label tokens inside the grading request** (6 tokens in all three
templates) and at the grade position, every other layer plus the last.

- **Interaction probe (primary).** Positive class: the label refers to the insulter in `assistant_insult`.
  Negative class: the neutral user's label in `assistant_insult`, and both labels in `third_party_v2`. A
  classifier that uses only the condition, or only the author's identity, reaches at most 0.83 balanced
  accuracy; one that knows "this author insulted the assistant" can reach 1.0. Leave-one-template-out
  logistic regression (C = 0.05), balanced accuracy per layer. At the peak layer, also train on speaker
  User-1 rows and test on User-2 rows and vice versa (held-out names).
- **Controls.** Condition probe (`assistant_insult` vs `third_party_v2`) at the neutral user's label and at
  the grade position (expected high everywhere, as on Qwen: decodability of condition is not binding).
  Chance and the 0.83 single-cue ceiling are reported alongside.
- **Direction.** Per layer, the mean difference (insulter label in `assistant_insult`) minus (insulter label
  in `third_party_v2`), saved for Stage B.

### Stage B: does the label's content reach the grade?

At two layers: the interaction-probe peak (restricted to layers >= 4) and the depth-matched Qwen site
(round(17/36 x n_layers): Gemma 23, Phi-4 19; Qwen 17). Shared-suffix alignment as in Phase 2. Readouts:
insulter-author grade change, neutral-author grade change (preservation), valid mass, discrimination.

- **Forward patch** (the Qwen I1 intervention): `third_party_v2` label activations into the
  `assistant_insult` conversation. On Qwen this attenuates the penalty (+0.10 held-out).
- **Reverse patch:** `assistant_insult` label activations into the `third_party_v2` conversation. On Qwen
  this transfers the penalty (-0.15).
- **Steering:** add alpha x direction at the author-label tokens of the `third_party_v2` conversation,
  alpha in {1, 3}; applied to insulter-author and neutral-author rows alike (the neutral-author change is
  the specificity check).

## Reading rules (fixed in advance)

The Qwen control must show: interaction probe > 0.90 at some layer, forward patch >= +0.05, reverse patch
<= -0.05, steering alpha = 1 <= -0.05 at layer 17. Otherwise the tooling failed and the other models'
results are not interpreted.

For Gemma and Phi-4, with "penalty" meaning insulter-author grade change <= -0.10 (95% interval excluding 0
over templates) with neutral-author change within +/-0.05:

| Interaction probe at the label | Reverse patch / steering | Reading |
|---|---|---|
| <= 0.85 at every layer (no better than a single cue) | - | Not bound at the label: the name does not know this author insulted the assistant |
| > 0.90 | No penalty at either layer or alpha | Bound, but routed out of grading; look for a late gate |
| > 0.90 | Penalty appears | Bound and readable; the natural signal is suppressed in magnitude or cancelled downstream |
| between | - | Inconclusive; report |

A later arm on base checkpoints (few-shot transcript format) is a separate, adapted experiment.

## Predictions (written before running)

Qwen control passes all four gates (0.8). Interaction probe > 0.90 on Gemma and Phi-4 (0.65: condition and
identity were both decodable everywhere on Qwen, and their conjunction probably is too). No penalty from
reverse patch or steering on Gemma or Phi-4 at either layer (0.6), i.e. "bound, routed out of grading". If a
penalty does appear, it appears on Gemma rather than Phi-4 (0.6; Gemma showed a directional general drop).

## Budget

Per model ~1300-1700 prompts x (baselines + pooled capture + 2 layers x (2 captures + 2 patches + 2
steers)) in bf16 on an L40S: ~60 min, cap 75 min / $1.40. Three models, one at a time: total cap $4.20;
balance $22. Probes run on the pod's CPU.

## Code

`scripts/run_exp16.py` (`--model qwen|gemma|phi`, `--stage A|B|AB`, `--dry`), `src/patching.py` generalised
to the three architectures (layer list, hidden size, chat markers; region finder checked on all three
tokenizers: author label = 6 tokens each). Analysis: `scripts/analyze_exp16.py`.
