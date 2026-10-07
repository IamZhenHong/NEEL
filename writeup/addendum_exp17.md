# Addendum 4 to write-up v1: when does grading read the author's name, and does that read see the insult? (Experiment 17)

*Added 2026-10-07; `writeup_v1.md` unchanged. Pre-registration `prereg/exp17.md` (amendment v1.1 is logistics
only); tags `exp17-freeze`, `exp17-results`. Exploratory-mechanistic (bf16), reading rules fixed before running.
Read after `addendum_exp16.md`.*

## Question

Experiment 16 left Gemma 4 12B-it and Phi-4 as "both ingredients present at the author's name, grade does not use
them". Two mechanisms fit. **Timing gate:** grading reads the name before the insult cue has been written to it;
whatever reads it later is off the grading path. **Selective read:** grading reads the name while it carries the
cue and extracts only identity. Reverse patches cannot separate these in models that show no penalty (donor and
recipient are both unpenalised), so the lever is the one those models have: the corrector bonus.

## Method

Same code on all three models; Qwen3-8B as positive control (its lever is the penalty). For each cell, the
corrector-author and neutral-author prompts differ only at the name digits inside the grading request (equal
token lengths, asserted per cell). **Stage B** replaces the residual stream at the six author-label positions
with the other author's, at the output of one block L (every fourth block plus Exp 16's tag block), and at every
block plus the embedding ("all"). Movement toward the partner's grade divided by the lever gives **tau(L)**, the
share of grade-relevant reads of the label that happen in blocks after L. **Stage A** probes at the label and at
the grade position give the depth at which the insult cue reaches the label (**L_tag**: first hidden index with
condition accuracy >= 0.90, minus one) and add the baseline Exp 16 lacked: a classifier that sees only the
out-of-fold condition and identity probe scores. Gates: tau("all") in [0.9, 1.1]; Qwen tau at or after L_tag
>= 0.3. Reading, fixed in advance: tau(L+) >= 0.5 -> overlapping read (selective read); tau(L+) < 0.5 and
tau(L-) <= 0.15 -> early read (timing gate); otherwise partial.

## Observations

| | Qwen3-8B | Gemma 4 12B-it | Phi-4 |
|---|---|---|---|
| Lever, insult condition (neutral minus corrector) | +0.135 [+0.051, +0.219] | -0.232 [-0.348, -0.116] | -0.146 [-0.201, -0.092] |
| Lever, flat condition | -0.041 (below floor, not read) | -0.187 [-0.290, -0.084] | -0.097 [-0.129, -0.064] |
| Full-stack swap tau | 1.00 (gate pass) | 1.00 / 1.00 | 1.00 / 1.00 |
| Interaction probe peak / additive-from-cues baseline | 0.97 / 0.96 at h18 | 1.00 / 1.00 at h18 | 0.98 / 0.97 at h16 |
| L_tag (cue at the label) | 17 | 17 | 15 |
| tau by block, insult | 0.99, 0.92, 0.87, 0.96 (0-12); 0.67 (16); 0.62 (17); 0.27 (20); 0 from 24 | ~1.0 (0-17); 0.68 (20); 0.31 (24); 0 from 28 | 1.0 (0-4); 0.89 (8); 0.82 (12); 0.38 (15); 0.31 (16); 0 from 20 |
| Read window L_50 / L_10, insult | 17 / 20 | 20 / 24 | 12 / 16 |
| tau(L_tag), insult / flat | 0.62 (control pass) / n.a. | 0.99 / 1.07 | 0.38 / 0.77 |
| Reading, insult / flat | control / not read | **overlapping / overlapping** | **partial / overlapping** |

Every gate passed. The Exp 16 erratum check is confirmed on all three models: the additive baseline is within
0.02 of the interaction probe at its peak, so the Exp 16 probe showed the two cues present, not a conjunction
feature. The grade-position identity probe reaches 0.95 in Phi-4 and 0.87 in Qwen and Gemma (prediction
">= 0.90 in all three" missed). The residual probe (both cue directions projected out per fold) reaches 0.77-0.82
in Qwen, 0.91-0.94 in Gemma (h24-h30) and 0.81 in Phi-4; indicative only by the pre-registration.

## Interpretation (author's reading; status in brackets)

**Gemma: the timing gate is excluded** [pre-registered reading, both conditions agree]. Essentially every
grade-relevant read of the author's label happens at or after the block whose output carries the insult cue
(tau(17) = 0.99; reads in blocks 18-27), and Exp 16 showed that swapping the condition at those layers moves the
grade by at most 0.04. Gemma's grading reads the name while it carries the cue and takes only identity.

**Phi-4: partial under insults, overlapping flat; not read as a whole** [pre-registered table]. Between 0.38 and
0.82 of its grade-relevant reads see the cue. The lower bound is well above zero ([0.32, 0.48]), so a pure timing
gate is insufficient for Phi-4 as well: at least a third of the reads see the cue and do not pass it
[interpretation, not a pre-registered reading]. What the table cannot say is whether the earlier reads would have
taken the cue had it been there. Insults move Phi-4's read earlier (tau(15) 0.77 flat vs 0.38 insult, intervals
disjoint); Gemma's read depth does not change with insults; Qwen's flat lever is too small to compare.

**Qwen: the same band, a different extraction** [positive control]. Qwen reads the label for grading in blocks
13-23, about 60% of it after the cue arrives at block 17, and Exp 16's block-17 condition swap transferred ~0.55 of
the penalty. All three models read the author's name for grading in the same depth band where the insult cue is
present at it. Qwen's read carries the cue into the grade; the other two models' reads do not. The difference
between the models is in what the read extracts, not in when it happens.

**Not shown.** Which heads perform the read, what in Qwen's read takes the cue, and whether the earlier Phi-4
reads are blind to the cue or merely early. The natural next step is head-level patching inside the block 13-23
band on Qwen against the same band on Gemma. Prediction record for this experiment: erratum confirmed (0.8) met
3/3; Qwen gates and L_50 >= 17 met; Gemma overlapping (0.55) met; Phi-4 overlapping (0.5) missed, partial (0.2)
is the insult row; grade-position identity >= 0.90 everywhere (0.85) missed 2/3.

## Cost and provenance

Three L40S pods, bf16: Qwen 18 min ($0.43, plus an aborted first attempt of $0.60 whose probes ran on the pod's
CPU and would not have fit the cap; amendment v1.1 moved the probes to the laptop from saved residuals, design
unchanged), Gemma 37 min ($0.81), Phi-4 53 min ($1.12). Total ~$2.96 of the amended $4.85 cap. Code at
`e4aa235`; outputs `results/exp17_<model>_{base,swap}.jsonl`, `_probe.json`, `_pooled.pt`; analysis
`scripts/analyze_exp17.py`.
