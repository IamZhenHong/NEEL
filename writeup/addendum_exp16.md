# Addendum 3 to write-up v1: is the insult bound to the author's name in models that show no penalty? (Experiment 16)

*Added 2026-10-06; `writeup_v1.md` unchanged. Pre-registration `prereg/exp16.md`; tags `exp16-freeze`,
`exp16-results`. Exploratory-mechanistic (bf16), reading rules fixed before running. Read after
`addendum_exp14.md` and `addendum_exp15b.md`.*

## Question

Gemma 4 12B-it and Phi-4 show no insulter penalty. Is that because the author's name token at the grading
request never carries "this author insulted the assistant" (never bound), or because it does and the grading
computation does not use it (bound but gated)?

## Method

Same code on all three models; Qwen3-8B as positive control. An **interaction probe** at the six author-label
tokens inside the grading request: positive = the insulter's label in the assistant-insult conversation;
negatives = the neutral user's label in that conversation and both labels in the contractor-insult
conversation. A probe using only the condition or only the author's identity reaches at most 0.83 balanced
accuracy; only the conjunction reaches 1.0. Leave-one-template-out logistic regression, every other layer.
Then **author-label activation patches** between the insult and contractor conversations (shared-suffix
alignment, as in Phase 2), forward and reverse, at the probe-peak layer and the depth-matched Qwen site, with
the neutral author as the specificity check. A mean-difference steering control failed its gate on Qwen (weak,
non-specific) and was not used in the reading.

## Observations

| | Qwen3-8B | Gemma 4 12B-it | Phi-4 |
|---|---|---|---|
| Interaction probe peak (ceiling 0.83) | 0.97 at L18-20 | 1.00 at L18, L24-26 | 0.98 at L16-18 |
| Forward patch (contractor label activations into the insult conversation), insulter's grade | +0.102 [+0.063, +0.141] at L17 | -0.001 [-0.031, +0.029] L18; -0.040 [-0.069, -0.011] L23 | -0.022 [-0.033, -0.012] L16; 0.000 L19 |
| Reverse patch, insulter's grade | -0.114 [-0.165, -0.063] at L17 | +0.002 [-0.013, +0.017] L18; +0.011 L23 | +0.026 [+0.013, +0.038] L16; 0.000 L19 |
| Neutral author under the same patches | within 0.02 | within 0.01 | within 0.003 |
| Baseline targeted contrast (bf16, vs contractor) | -0.196 [-0.284, -0.107] | -0.027 [-0.122, +0.067] | +0.051 [+0.016, +0.087] |

Cross-speaker probe transfer (train on one name as insulter, test on the other): Qwen 0.70 / 0.52, Gemma
0.52 / 0.38, Phi-4 0.70 / 0.65.

## Interpretation (author's reading; status in brackets)

All three models bind "this author insulted the assistant" to the author's name at the grading request, at a
similar relative depth. [Supported: the interaction probe clears the single-cue ceiling by 0.14-0.17 on every
model.] Only Qwen3-8B's grading computation reads that content as a penalty; in Gemma and Phi-4 the same
activations, swapped between conversations, move the grade by at most 0.04, and in Phi-4 the sign is that of
a tiny bonus. [Supported: pre-registered reading "bound, but routed out of grading" for both.] So the
Qwen-specific behaviour is not a difference in what the name represents; it is a difference in what the later
computation does with it, which points at post-training-shaped readout rather than pretraining or
architecture. [Inference from the table; the gate itself is not located.]

Caveats: bf16; two layers per model; the probe transfers poorly across the two names on Qwen and Gemma, so
"bound" means the (name, insult-at-assistant) conjunction is linearly present, not that it is an abstract
feature; the steering tool was uninformative; one task format, one language.

## What this changes

The forensic question "what is different about Qwen3-8B?" narrows to "what, after the author's name has been
tagged, turns the tag into a lower grade in Qwen and not in the others?". The natural next step is to locate
that step within Gemma or Phi-4 (late-layer patches at the request and grade positions) or to compare base
and instruct checkpoints; both are separate experiments. The behavioural conclusions of the earlier addenda
are unchanged.
