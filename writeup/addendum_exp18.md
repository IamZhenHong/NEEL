# Addendum 5 to write-up v1: which heads read the author's name for grading, and which carry the insult? (Experiment 18)

*Added 2026-10-07; `writeup_v1.md` unchanged. Pre-registration `prereg/exp18.md`; tags `exp18-freeze`, `exp18-results`.
Exploratory-mechanistic (bf16), reading rules fixed before running. Read after `addendum_exp17.md`.*

## Question

Exp 17 placed every model's grade-relevant read of the author's label in the same depth band, overlapping the depth at
which the insult cue is present at the label, and left the difference between Qwen3-8B and the others in *what the read
extracts*. This experiment decomposes the read into attention heads: (1) which heads perform the grade-relevant read of
the label in Qwen3-8B and Gemma 4 12B-it; (2) in Qwen, which heads carry the insult cue into the grade, and whether they
are the reading heads; (3) in Gemma, whether the reading heads pass the cue (the null comparison).

## Method

Per-head swap at the input of each block's attention output projection (head h = slice [h·d, (h+1)·d) of the o_proj
input) at the **post-label positions**: every position after the first author-label token, label tokens excluded
(about 290 positions: the rest of the request, the paragraph, the generation prompt). Donor and recipient are
token-identical at and after those positions, so swapping all heads of all blocks there reproduces the donor's forward
pass exactly (residual = identical embedding + donor attention outputs + MLP of that residual); that is the tooling
gate, checked to 1e-4 in the fp32 dry run and by tau ≈ 1 on the pod.

- **Author swap:** donor = the other author's row of the same cell (Exp 17's lever: Qwen's penalty, Gemma's corrector
  bonus). tau = movement toward the donor / lever, ratio of template means, bootstrap over templates, both directions
  pooled.
- **Condition swap:** recipient = the insulter's row in the insult conversation, donor = the same cell's row in the
  contractor-insult conversation (Exp 16's lever, inside the shared suffix). Qwen: tau against the penalty; Gemma: raw
  movement, no lever.

Hierarchy on a fixed screening subset (first paragraph per template × speaker × order, insulter's rows): all heads in
the band → each block → each key/value group in the expanded blocks → each head in the expanded groups, with the
pre-registered expansion rule (tau ≥ 0.10, interval above 0; caps Qwen 6 blocks / 12 groups, Gemma 4 / 8). Candidates
are confirmed on all rows, alone and jointly, beside the whole band and three random draws of as many heads from
outside the band. A head **reads** if its author-swap tau ≥ 0.10 with the interval above 0 on the confirmation rows;
a Qwen head **carries the cue** if its condition-swap tau ≥ 0.10 with the interval above 0.

## Observations

**Qwen3-8B** (pod 1: stage A, 60 min cap reached during stage C's screening, $1.10; pod 2: stage C from scratch,
25 min, $0.49; bf16, batch 8; lever A +0.148, lever C +0.328 per template).

Gates: all-heads/all-blocks swap tau +0.95 [+0.87, +1.01] (author) and +0.98 [+0.90, +1.03] (condition); band 12-24
all heads tau +0.92 [+0.81, +1.01] (author, confirmation rows) and +0.99 [+0.91, +1.05] (condition). Every gate passed.
Screening expanded blocks 16, 17, 19, 21, 22, 23 (author) and 16, 17, 18, 19, 21, 22 (condition); 24 and 11 candidates.

| head | reads the label (author swap tau, 432 rows) | carries the condition (condition swap tau, 216 rows) |
|---|---|---|
| b21h19 | **+0.30** [+0.20, +0.49] | **+0.41** [+0.26, +0.74] |
| b21h18 | **+0.29** [+0.19, +0.50] | **+0.18** [+0.10, +0.31] |
| b22h7 | **+0.14** [+0.07, +0.28] | **+0.15** [+0.06, +0.36] |
| b19h24 | **+0.24** [+0.14, +0.50] | not a candidate |
| b19h21 | **+0.23** [+0.11, +0.52] | not a candidate |
| b17h31 | **+0.19** [+0.10, +0.35] | not a candidate |
| b19h29 | **+0.17** [+0.09, +0.36] | not a candidate |
| b23h25 | **+0.17** [+0.06, +0.30] | not a candidate |
| b16h11 / b16h17 / b16h3 / b19h27 | **+0.11 to +0.12**, intervals above 0 | not candidates |
| b18h10 | not a candidate | **+0.27** [+0.15, +0.48] |
| b19h28 | +0.10 [+0.03, +0.17] (below the 0.10 rule) | **+0.18** [+0.09, +0.35] |
| b19h31 | not a candidate | **+0.16** [+0.07, +0.29] |
| b19h10 | not a candidate | **+0.11** [+0.01, +0.29] |
| 12 other screening candidates | +0.03 to +0.17, intervals touching 0 | (b22h5 +0.13 [-0.15, +0.33]; b17h15, b22h15, b19h23 below the rule) |
| joint, all candidates | +1.03 [+0.86, +1.35] | +0.95 [+0.81, +1.18] |
| joint, the 12 reading heads | | **+0.74** [+0.56, +1.13] |
| whole band 12-24 | +0.92 [+0.81, +1.01] | +0.99 [+0.91, +1.05] |
| random, as many heads outside the band (3 draws) | +0.09, +0.08, +0.05 (24 heads) | +0.04, +0.07, +0.10 (11 heads) |

**Gemma 4 12B-it** (amendment v1.2: caps 3 blocks / 6 groups, confirmation on the insulter's rows, stage C without
per-block screening; pod 1: stage A, crashed at the start of stage C on a runner bug after the stage-A confirmation was
written, 24 min, $0.48; pod 2: stage C, 7 min, $0.23; bf16, batch 4; lever A -0.232 (corrector bonus), lever C +0.208:
the insulter is graded 0.21 *higher* in the contractor-insult conversation, i.e. Gemma's general drop after insults at
the assistant, which gives stage C a defined lever after all).

Gates: all-heads/all-blocks swap, condition: tau +1.02 [+0.70, +1.29]; author: the one-row-per-template sanity subset
happened to have a lever of -0.071 (below the 0.08 floor), so its tau is undefined (movement -0.043 [-0.219, +0.158]);
the band swap on the 200 confirmation rows gives +1.03 [+1.00, +1.08] (author) and +1.00 [+0.91, +1.18] (condition).
Row-level bf16 deviation of the all-heads swap from the donor's own pass is mean-zero (Qwen mean |d| 0.03, Gemma 0.05,
one Gemma row at 0.40). Screening expanded blocks 27, 23, 28 (block 21, tau +0.40, was fourth under the cap of 3);
7 candidates.

| head (block type) | reads the label (author swap tau, 200 rows) | condition swap |
|---|---|---|
| b28h12 (sliding) | **+0.60** [+0.44, +0.81] | per-head not run (amendment) |
| b27h1 (sliding) | **+0.46** [+0.28, +0.82] | |
| b23h11 (full attention) | **+0.45** [+0.15, +0.91] | |
| b23h4 (full attention) | **+0.35** [+0.23, +0.64] | |
| b27h12 (sliding) | **+0.26** [+0.15, +0.48] | |
| b28h13 (sliding) | +0.11 [+0.06, +0.20] (rule met; inside the control range) | |
| b28h15 | +0.10 [+0.05, +0.17] (below the rule) | |
| joint, all 7 candidates | +1.34 [+1.10, +1.80] | |
| joint, the 6 reading heads | | **+0.45** [+0.00, +0.65] |
| whole band 16-28 | +1.03 [+1.00, +1.08] | +1.00 [+0.91, +1.18] |
| random, 7 heads outside the band (3 draws) | +0.08, +0.05, **-0.12** [-0.29, -0.02] | (1 head: -0.01, +0.01, +0.01) |

Relative depth of the reading heads (median block / blocks): Qwen 19/36 = 0.53, Gemma 27/48 = 0.56.

## Prediction record

1. Tooling gates (0.85): Qwen both stages and Gemma's condition stage pass; Gemma's author sanity subset is undefined
   (lever below the floor by chance), with the band swap at 1.03. **Met with that caveat.** Qwen band gate (0.75): met.
2. Concentration (0.55 / 0.5; joint >= half the band, 0.6): top-4 individual taus sum to 1.06 (Qwen) and 1.86 (Gemma);
   joint candidates 1.03 vs band 0.92 and 1.34 vs 1.03. **Met.** (Sums above 1 mean the single-head taus are not
   additive: the heads are partly redundant, each one's swap moving the grade by more than its exclusive share.)
3. Qwen's cue-carrying heads a subset of its reading heads (0.6): **not met**, 3 of 7 (b21h18, b21h19, b22h7); four
   heads carry the condition without reading the label (b18h10, b19h10, b19h28, b19h31). At least one Qwen head carries
   at tau >= 0.10 (0.7): met, seven do, b21h19 at 0.41.
4. Gemma's reading heads together move the grade by <= 0.04 under the condition swap (0.75): **not met**; tau +0.45
   [+0.00, +0.65] of a 0.208 lever = 0.09 grade points, interval reaching 0. Per-head Gemma cue test: not run.
5. Matched relative depth (0.5): 0.53 vs 0.56. Met.
6. Random controls within +-0.10 in every draw (0.8): **not met in one draw**: a Gemma 7-head set from outside the band
   gives -0.12 [-0.29, -0.02]; the other five draws are within +-0.10 (Qwen's largest +0.10 [+0.01, +0.24]).
Running record: Exp 14-15b 0/4, Exp 16 3/4, Exp 17 3/5, Exp 18 4/7.


## Interpretation (author's reading; status in brackets)

**The read has heads, in both models** [pre-registered rules, every gate passed]. Twelve Qwen heads in blocks 16-23
and six Gemma heads in blocks 23-28 each move the grade toward the other author's by 0.10-0.60 of the lever when their
output after the author's name is swapped; the whole set reproduces the band (joint 1.03 / 1.34 against band 0.92 /
1.03), random sets of as many heads from outside the band give 0.05-0.09 (one Gemma draw -0.12). Gemma's read is more
concentrated (b28h12 alone 0.60; four heads >= 0.35) than Qwen's (two heads at 0.30, then 0.11-0.24); both sit at the
same relative depth (0.53 / 0.56). "Localised" remains reserved: these are sufficiency statements about head outputs
after the label, the attention targets are not shown, and the single-head taus are not additive.

**In Qwen, three of the reading heads also carry the insult into the grade, and four heads carry it without reading
the name** [pre-registered rules; prediction 3 missed]. b21h19 is the strongest on both counts (reads 0.30, carries
0.41), with b21h18 (0.29 / 0.18) and b22h7 (0.14 / 0.15). The twelve reading heads together carry 0.74 [0.56, 1.13] of
the whole insult-vs-contractor difference in the insulter's grade (lever +0.33, which includes the general drop). The
four cue-only heads (b18h10 0.27, b19h28 0.18, b19h31 0.16, b19h10 0.11) move the grade with the condition without
reading the author's identity: the direct route of Exp 12 (the request re-reading the insult turns), i.e. the general
drop, at head level. What the condition swap at the post-label positions measures must be stated exactly: a head
"carries the condition" if its outputs after the label differ between the insult and contractor conversations in a
grade-relevant way, by any route (through the label, which Exp 16 isolated, or by attending to the insult turns
directly). The intersection with the reading heads is the name-mediated part; a per-head *label-route* swap (donor =
the same conversation with only the label residual replaced, as in Exp 16) would separate the two routes head by head
and was not run.

**Gemma's reading heads are not blind to the insults** [prediction 4 missed, interval touching 0]. Together they carry
0.45 [0.00, 0.65] of Gemma's insult-vs-contractor difference, which for Gemma is the general drop (lever +0.21; Exp 14:
no insulter-specific penalty, everyone graded lower after insults at the assistant). So Gemma's name-reading heads also
convey "insults happened" into the grade, as Qwen's do; what they do not do, per Exp 16's label-only swap (<= 0.04), is
pass the insult *as a property of the author*. The Qwen-Gemma difference is therefore narrower than "Gemma's read takes
identity only": both models' reading heads carry the conversation's insult state; Qwen's additionally carry the
binding of that state to the name. Which of Qwen's three intersection heads does the binding, and whether Gemma's
b28h12 would carry it if the label residual were swapped, is the per-head label-route question above.

**Not shown.** Attention targets of the reading heads; the direction inside the head outputs that carries the cue;
Phi-4; Gemma per-head condition swaps (budget); both-direction confirmation in Gemma (one direction only). Next step if
pursued: the per-head label-route condition swap on b21h18 / b21h19 / b22h7 (Qwen) and b28h12 / b27h1 / b23h11 (Gemma),
about $1 per model with the captures already scripted.

## Cost and provenance

Four L40S pods, bf16: Qwen stage A 60 min ($1.10; the cap cut stage C's screening, which was re-run from scratch on a
second pod, 25 min, $0.53; amendment v1.1), Gemma stage A 24 min ($0.48; crashed at the start of stage C on a runner bug
in the no-block-screening path, fixed at `1eb8e24`, stage C re-run on a second pod, 7 min, $0.23; amendment v1.2).
Total $2.34 of the $3.00 cap. Code `src/headpatch.py`, `src/exp18_stats.py`, `scripts/run_exp18.py`,
`scripts/analyze_exp18.py` at tag `exp18-results`; outputs `results/exp18_<model>_{base,author,cond}.jsonl`,
`_plan.json` (stage A and C merged; the per-stage originals kept as `_plan_stageA.json`), analysis printout
`results/exp18_summary.txt`. Deviations from the pre-registration: the two amendments (budget: Qwen stage C on a second
pod; Gemma caps 3/6, one-direction confirmation, no per-block stage-C screening) and the Gemma author sanity gate being
undefined on its subset.
