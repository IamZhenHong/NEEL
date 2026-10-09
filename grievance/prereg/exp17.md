# Experiment 17: where does grading read the author's name, and does that read see the insult cue? (pre-registration)

*Written 2026-10-06 before any data. Exploratory-mechanistic (bf16), reading rules fixed in advance. To be
frozen at tag `exp17-freeze` when launch is authorised. GPU work otherwise paused pending the TA review.*

## Background and a correction to Experiment 16

Experiment 16 found that swapping the author-label activations between the insult and contractor-insult
conversations moves Qwen3-8B's grade (+0.10 / -0.11) and Gemma 4 12B-it's and Phi-4's by at most 0.04, and
read the latter two as "bound, but routed out of grading". The patch results stand. The probe argument has a
flaw, recorded here before any new data: the "0.83 single-cue ceiling" applies to a classifier that uses one
cue, but a **linear** probe with both cues available can compute their conjunction additively (threshold on
the sum of a condition score and an identity score). In the Exp 16 data the identity cue is decodable at the
label at 1.00 and the condition cue at 0.93-0.95 (L18, all three models), so the interaction probe's 0.97-1.00
is explained without any conjunction feature. "Bound" must therefore read: *both ingredients ("this is the
corrector's name", "the assistant was insulted") are linearly present at the author's label, so the
conjunction is computable from it*; a dedicated conjunction feature was not shown. Stage A below adds the
missing baseline. An erratum is appended to `writeup/addendum_exp16.md`.

## Question

In Gemma and Phi-4 the ingredients are at the label and the grade does not use them. Two mechanisms fit:

1. **Timing gate.** Grading reads the author's name early, before the insult cue has been written to it
   (the cue arrives at hidden index ~18 of 48 in Gemma, ~16 of 40 in Phi-4). Whatever reads the name later
   is not on the grading path.
2. **Selective read.** Grading reads the name while it carries the cue, but extracts only identity (who the
   author is) and not the cue; the Exp 16 null patches at those layers are then a selective read.

The reviewer's proposal (reverse patches at the request and grade positions in late layers) cannot separate
these in Gemma or Phi-4: donor and recipient conversations are both unpenalised, so no patch between them
can create a penalty. The lever that exists in those models is the **corrector bonus** (bf16 Exp 16
baselines, corrector minus neutral author: Gemma +0.19 flat / +0.22 insult, Phi-4 +0.10 / +0.15; Qwen's
lever in the insult condition is its penalty, -0.135). Swapping the author's label between the two authors'
rows of the same cell measures, layer by layer, when grading reads the name at all.

## Design

Three models, identical code: Qwen3-8B (positive control), Gemma 4 12B-it, Phi-4 (with the Exp 15b format
line). Per-model paragraph selections from Exp 6 / 14 / 15b; templates 006-030 minus 008; authors corrector
("provoker") and neutral; bf16; unit = template.

**Alignment.** The corrector-author and neutral-author prompts of one cell differ only at the name digits
inside the grading request (verified on the Qwen tokenizer: equal lengths, differing token positions a
subset of the 6 author-label positions, for all 216 Exp 6 cells; the runner asserts this for every cell on
every model). So activations can be swapped position by position without the shared-suffix restriction.

### Stage A: probes with the missing baseline (3 conditions, pooled label and grade positions, every other hidden index)

Four cells: {insult, contractor-insult} x {corrector's label, neutral's label}. At the label and at the grade
position: condition probe, identity probe, the Exp 16 interaction probe, and the **additive-from-cues
baseline** (leave-one-template-out logistic regression on the two out-of-fold cue scores). At the label also
a **residual probe** (condition and identity probe directions projected out within each fold, then the
interaction probe refit); indicative only, two directions is a weak removal. Identity in `flat_v2` as in
Exp 16. All LOTO, C = 0.05, balanced accuracy.

- **Erratum check (fixed):** if the additive baseline is within 0.03 of the interaction probe at its peak
  (hidden index >= 4), Exp 16's probe is confirmed to have shown the two cues and not a conjunction.
- **Tag depth** `h_tag`: the first hidden index >= 4 at which the condition probe at the label reaches
  0.90; `L_tag = h_tag - 1` is the block whose output first carries the cue. Reads in blocks > L_tag see it.

### Stage B: label-swap read profile (conditions `flat_v2` and `assistant_insult`)

For every row, replace the residual stream at its 6 author-label positions with the partner row's (same
cell, other author) at the output of one block L, for L in {0, 4, 8, ...} up to the penultimate block plus
the Exp 16 tag block (Qwen 17, Gemma 17, Phi-4 15); and at **every** block plus the embedding ("all").
Both directions in one pass. Readout per row: expected grade, valid mass.

- **Movement** toward the partner, in lever units: corrector row: y_patched - y_own; neutral row:
  -(y_patched - y_own). **Lever** per template: y(neutral author) - y(corrector author), unpatched.
- **tau(L)** = mean-over-templates movement / mean-over-templates lever (ratio of means; bootstrap 95%
  over templates, 2000 resamples, seed 0). tau(L) is the share of grade-relevant reads of the label that
  happen in blocks > L. tau is reported only where |mean lever| >= 0.08 (Qwen `flat_v2`, lever +0.04, is
  excluded from the reading).
- **Read window:** deepest sweep block with tau >= 0.5 (`L_50`) and with tau >= 0.1 and interval above 0
  (`L_10`).

## Gates and reading rules (fixed in advance)

**Tooling gate (every model, every condition read):** tau("all") in [0.9, 1.1]. The full-stack swap is the
partner's forward pass, so anything else means the alignment or the patch is wrong. (The fp32 dry run
asserts max |difference| < 1e-3 against the partner's baseline.)

**Qwen control:** in `assistant_insult`, tau at the first sweep block >= L_tag is >= 0.3 (Exp 16's single
block-17 condition swap transferred about 0.55 of the penalty). Otherwise the tooling is not trusted and
Gemma / Phi-4 are not read.

For Gemma and Phi-4, with `L+` = the smallest sweep block >= L_tag (tau(L+) is a lower bound on the share of
reads that see the cue) and `L-` = the largest sweep block < L_tag (tau(L-) is an upper bound):

| tau(L+) | tau(L-) | Reading |
|---|---|---|
| >= 0.5 | - | **Overlapping read (selective read):** at least half the label reads happen after the cue is present; with Exp 16's null condition swaps at those layers, the reads take identity and not the cue |
| < 0.5 | <= 0.15 | **Early read (timing gate):** grading has finished reading the author's name before the insult cue reaches it |
| otherwise | | **Partial:** report both bounds; not decided between the two |

`assistant_insult` is the primary condition for the reading; `flat_v2` is the insult-free comparison (does
the presence of insults change when the name is read?). A model is read as a whole only if both conditions
give the same row; otherwise report both.

What this does not do: it does not identify the heads or the direction of the gate, and it says nothing about
why Qwen's late read takes the cue. A base-vs-instruct arm remains a separate, adapted experiment.

## Predictions (written before running)

- Erratum confirmed on all three models (0.8): additive baseline within 0.03 of the interaction probe.
- Qwen passes both gates (0.8) and its read window extends past L_tag (`L_50` >= 17; 0.7).
- Gemma: overlapping read (0.55), early read (0.3), partial (0.15). Phi-4: overlapping (0.5), early (0.3),
  partial (0.2). (Record: my Stage 2 predictions were wrong 4/4 on Exp 14-15b and right on Exp 16.)
- The grade-position identity probe is high (>= 0.90) somewhere in all three models (0.85): the bonus and the
  penalty both require the grade to know who the author is.

## Budget

Row counts per condition: Qwen 432, Gemma 400, Phi-4 576. Per model: baselines (2 conditions) + pooled
capture (3 conditions) + per condition one capture pass and (sweep + 1) swap passes. At Exp 16 throughput
(Qwen ~400, Gemma ~200, Phi-4 ~250 row-forwards/min on an L40S): Qwen ~30 min, Gemma ~70 min, Phi-4 ~70 min.
Caps: Qwen 45 min / $0.85; Gemma 90 min / $1.70; Phi-4 90 min / $1.70. Total cap $4.25. One pod at a time,
Qwen first (control). CPU memory for the all-layer label capture: <= 6 GB.

## Code

`scripts/run_exp17.py` (`--model qwen|gemma|phi`, `--stage A|B|AB`, `--step 4`, `--dry`); `src/patching.py`
gains `capture_at` / `run_patch_at` (position-indexed capture and replacement at many layers, -1 = embedding).
Analysis: `scripts/analyze_exp17.py`.

## Amendment v1.1 (2026-10-07, logistics only; design, probes and reading rules unchanged)

- **First Qwen launch aborted on time (pod psmi05n26at1y8, L40S bf16, ~33 min, ~$0.60, git 6202e61 = tag
  `exp17-freeze`).** Baselines and the pooled capture took 267 s; the Stage A probes then ran on the pod's CPU at
  about 3 min per hidden index (the additive baseline nests a leave-one-template-out fit inside each fold), so
  the 19 indices would have needed ~60 min against a 45-min job limit, with `_probe.json` written only at the
  end. The pod was terminated with no saved outputs rather than let the guard kill it at the cap.
- **Change:** `--probe-offline` makes the pod save the pooled residuals at the probe hidden indices (every other
  index plus the last; bf16, the model's own precision) to `results/exp17_<model>_pooled.pt`, and `--stage P`
  runs the identical probe code on the laptop (no GPU). Stage B is unchanged. Verified on the fp32 dry run
  (0.6B, 2 templates): the offline path produces `_probe.json` from the saved file.
- **Data seen before the relaunch (recorded, not read as a result):** the aborted run printed label probes for
  hidden indices 0-16. The additive baseline was within 0.03 of the interaction probe at every index (e.g. h12
  0.90/0.90, h14 0.92/0.93, h16 0.91/0.92) and the condition probe at the label first reached 0.90 at h14.
  These will be re-derived from the relaunch; the predictions above were fixed before either run.
- **Budget:** the aborted attempt (~$0.60) counts against Exp 17. Relaunch caps unchanged per pod (Qwen 45 min /
  $0.85; Gemma and Phi-4 90 min / $1.70), so the Exp 17 total cap becomes $4.85 if all three run to the cap.
