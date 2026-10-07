# Experiment 18: which heads read the author's name for grading, and which carry the insult into the grade? (pre-registration)

*Written 2026-10-07 before any data, after Exp 17 (`writeup/addendum_exp17.md`). Exploratory-mechanistic (bf16), reading
rules fixed in advance. Frozen at tag `exp18-freeze`; results at `exp18-results`. Launch authorised by the owner
2026-10-07 as the first sprint experiment on the provocation line.*

## Background

Exp 17: all three models read the author's label for grading in the same depth band, while the insult cue is present at
the label (Qwen reads in blocks ~13-23, cue from block 17 output, tau(17) = 0.62; Gemma reads in blocks ~18-27, cue from
block 17, tau(17) = 0.99). Only Qwen's read carries the cue into the grade (Exp 16: block-17 condition swap at the label
moves the grade by +0.10 / -0.11; Gemma and Phi-4 <= 0.04). The difference is in what the read extracts, not when it
happens. This experiment decomposes the read into attention heads.

## Questions

1. **Which heads perform the grade-relevant read of the author's label** in Qwen3-8B and Gemma 4 12B-it (the Exp 17
   lever: the two authors of the same cell, corrector vs neutral; Qwen's lever is its penalty, Gemma's is its corrector
   bonus)?
2. **In Qwen, which heads carry the insult cue into the grade** (the Exp 16 lever: the insult conversation vs the
   contractor-insult conversation, insulter's rows), and are they the same heads as in (1)?
3. **In Gemma, do the reading heads pass the cue?** (Exp 16 says no at the block level; here at the head level, the
   null comparison.)

## Intervention

Per-head swap at the input of each block's attention output projection (`self_attn.o_proj`; head h occupies the slice
[h*d, (h+1)*d)), at the **post-label positions**: every position after the first author-label token, label tokens
excluded (~290 positions: the rest of the grading request, the paragraph, the generation prompt). The donor is the
partner row; donor and recipient are token-identical at and after those positions (author pairs: everywhere; condition
pairs: inside the Exp 16 shared suffix, asserted per cell), so swapping **all heads of all blocks** at those positions
reproduces the donor's forward pass there exactly (residual = identical embedding + donor attention outputs + MLP of
that residual, by induction). That is the tooling gate. Code: `src/headpatch.py`, `scripts/run_exp18.py`,
`scripts/analyze_exp18.py`, `src/exp18_stats.py`.

- **Author swap (stage A):** recipient = a row, donor = the other author's row of the same cell. Movement toward the
  donor in lever units, tau = mean movement / mean lever over templates (ratio of means, bootstrap 2000 over
  templates, seed 0), sign-folded so both directions pool (as Exp 17). Lever = neutral-author grade minus
  corrector-author grade per template (Qwen `assistant_insult` +0.135; Gemma -0.232).
- **Condition swap (stage C):** recipient = the insulter's row in `assistant_insult`, donor = the same cell's row in
  `third_party_v2` (contractor insult). Lever = donor grade minus recipient grade per template (Qwen: the penalty,
  about +0.135 in this direction). For Gemma the lever is below the 0.08 floor, so stage C is read in raw movement.

## Design

Models: Qwen3-8B (band blocks 12-24 inclusive; Exp 17 tau(12) = 0.96, tau(24) = 0.01) and Gemma 4 12B-it (band 16-28;
tau(16) = 1.04, tau(28) = 0.01). Per-model paragraph selections, templates 006-030 minus 008, authors corrector and
neutral, bf16, unit = template (24 / 23). Gemma: 16 heads x 256, 8 key/value heads (groups of 2), sliding-window
(1024) and full-attention blocks interleaved (every 6th block full); the label-to-grade distance is under the window,
so every block can read the label; block type is recorded. Qwen: 32 heads x 128, 8 key/value heads (groups of 4).

**Screening subset** (fixed before running): the first paragraph (sorted id) per template x speaker x order, insulter's
rows: 96 rows (Qwen) / 92 (Gemma). **Hierarchy** per stage, on the subset:
1. all heads x all blocks (sanity; one row per template, must reproduce the donor's grade: fp32 dry run < 1e-3);
2. all heads in the band;
3. each block in the band (all heads);
4. each key/value group (its query heads together) in the **expanded blocks**;
5. each head in the **expanded groups**.
**Expansion rule:** a unit is expanded if tau >= 0.10 with the bootstrap interval above 0 (lever defined) or, without
a lever, movement >= 0.04 with the interval above 0; ranked by tau; **caps** (budget): Qwen <= 6 blocks, <= 12 groups;
Gemma <= 4 blocks, <= 8 groups. Heads passing the same rule at step 5 are **candidates**.
**Confirmation** on all rows (stage A: both directions, 432 / 400 rows; stage C: all insulter's rows, 216 / 200): each
candidate head alone; all candidates together ("joint"); the whole band; **random control**: three draws of as many
heads as there are candidates, from blocks outside the band (seed 0); and, in stage C, the stage-A reading heads
together ("author_heads").

## Gates and reading rules (fixed in advance)

- **Tooling gate** (both stages, both models): the all-heads/all-blocks swap reproduces the donor's grade
  (tau in [0.9, 1.1]; fp32 dry run exact to 1e-3). Fail -> nothing is read.
- **Band gate** (Qwen, stage A, confirmation rows): all heads in the band give tau >= 0.475 (half of Exp 17's band share
  tau(12) - tau(24) = 0.95). Fail -> the head decomposition at the attention output misses most of the read (e.g. MLP
  or embedding paths at the label); report, do not read heads.
- A head **reads** if its author-swap tau >= 0.10 with the interval above 0 on the confirmation rows.
- A Qwen head **carries the cue** if its condition-swap tau >= 0.10 with the interval above 0 on the confirmation rows.
  For Gemma (no lever): a head carries the cue if its raw movement >= 0.04 with the interval above 0.
- **Random control:** the random draws must give tau <= 0.10 (or movement <= 0.04); otherwise the per-head numbers
  are reported with the control beside them and not read as localisation.
- Heads that fail at screening are "not read individually at this budget", not "do not read".

## Predictions (written before running)

1. Tooling gates pass on both models and stages (0.85). Qwen band gate passes (0.75).
2. **Concentration:** in each model, the top 4 reading heads' individual taus sum to >= 0.50 (Qwen 0.55, Gemma 0.5).
   The joint swap of all candidates gives tau >= 0.5 of the band tau (0.6).
3. **Qwen's cue-carrying heads are a subset of its reading heads** (every confirmed cue-carrying head also reads; 0.6).
   At least one Qwen head carries the cue at tau >= 0.10 (0.7).
4. **Gemma's reading heads do not pass the cue:** the condition swap on Gemma's reading heads together moves the grade
   by <= 0.04 (0.75); no Gemma head is confirmed as cue-carrying (0.7).
5. Reading heads sit at matched relative depth in the two models (median block / n_blocks within 0.10; 0.5).
6. Random controls <= 0.10 in every draw (0.8).
Record: Exp 14-15b predictions 0/4, Exp 16 3/4, Exp 17 erratum + Qwen + Gemma right, Phi-4 and grade-identity wrong.

## What this does not do

It does not identify the heads' attention targets (which positions they read from), nor the direction of the cue inside
the read; it does not touch Phi-4; a head "reading" is a sufficiency statement about its output at the post-label
positions, not a claim about its queries or keys. "Localised" is reserved: the confirmed reading heads with their joint
tau are the claim.

## Budget

Caps: Qwen pod <= 60 min / $1.20; Gemma pod <= 75 min / $1.50; one pod at a time for this workstream; total <= $3.00.
Row-forwards: Qwen screening <= 2 + 13 + 48 + 48 passes x 96 rows per stage, confirmation <= 18 passes x 432 (A) /
216 (C); at Exp 17 throughput (~11 rows/s) about 45 min. Gemma expansion caps are tighter for the same reason. Results
are written incrementally; if a stage is cut by the time cap, the completed part is reported and the rest is "not run".
Captures stay on the pod; only the row files and the plan are pulled (a few MB).
