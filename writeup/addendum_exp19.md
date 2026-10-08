# Addendum 6 to write-up v1: which heads pass the insult *as a property of the author*? Per-head label-route swaps (Experiment 19)

*Added 2026-10-08; `writeup_v1.md` unchanged. Pre-registration `prereg/exp19.md`; tags `exp19-freeze`, `exp19-results`.
Exploratory-mechanistic (bf16), reading rules fixed before running. Read after `addendum_exp18.md`. Qwen3-8B pod
jk0zmnnd55tfys ($0.24, 610 s), Gemma 4 12B-it pod 9sc4q2vi33a75q ($0.35, 887 s); total $0.59 of the $1.45 cap.*

## Question

Exp 18's condition swap measured each head's grade-relevant difference between the insult and the contractor conversation by
any route. Its addendum inferred from set membership that Qwen's three heads which both read the author's name and carry the
condition (b21h18, b21h19, b22h7) carry the name-mediated penalty, and that the four condition-carrying heads which do not read
the name (b18h10, b19h10, b19h28, b19h31) are the direct route (re-reading the insult turns). This experiment measures the
label route head by head: the donor is the same row with only its author-label residual replaced at block 17's output (Exp
16's patch), and each spec swaps the chosen heads' pre-`o_proj` outputs at the ~275 post-label positions from that donor.
tau = share of the label-patch grade change (the lever) that passes through the swapped heads. Both directions (forward:
insulter's row with the contractor's label content; reverse) run and pooled sign-folded; 24 templates (Qwen), 23 (Gemma).

## Observations

**Qwen3-8B.** Lever +0.104 (forward) / +0.113 (reverse), pooled +0.108: Exp 16's block-17 label effect replicates on the
same materials. Tooling gate (all heads of blocks 18-35 from the label-patched donor, one row per template): tau +0.99
[+0.78, +1.25] and +1.01 [+0.81, +1.25]; pass both directions.

| spec (pooled, 432 rows, 24 templates) | tau [95% bootstrap] | forward | reverse |
|---|---|---|---|
| whole band, blocks 18-24 | **+0.89** [+0.72, +0.98] | +0.90 | +0.87 |
| `cueonly`: b18h10 + b19h10 + b19h28 + b19h31 | **+0.73** [+0.59, +0.89] | +0.70 [+0.51, +0.93] | +0.76 [+0.61, +0.93] |
| `reading`: the 8 Exp 18 reading heads above block 17 | +0.34 [+0.20, +0.44] | +0.29 [-0.11, +0.47] | +0.39 [+0.30, +0.50] |
| `inter`: b21h18 + b21h19 + b22h7 | +0.23 [+0.13, +0.30] | +0.20 [-0.15, +0.36] | +0.25 [+0.12, +0.40] |
| b21h18 alone | +0.07 [-0.04, +0.16] | +0.03 | +0.10 |
| b21h19 alone | +0.07 [-0.04, +0.15] | +0.03 | +0.11 |
| b22h7 alone | +0.02 [-0.05, +0.08] | -0.02 | +0.06 |
| random, 3 heads from blocks 25-35 (3 draws) | -0.06, +0.04, -0.05 (intervals include 0) | | |

Decision table: `inter` < 0.25 -> row 3 ("the three intersection heads do not carry the binding"); `reading` 0.34 sits
between the row's two sub-rules (>= 0.5: other reading heads; < 0.25: non-reading heads), so the row's own tie-break does not
decide. The `cueonly` spec does: four heads that do not read the name for grading carry 0.73 of the label route, with the
interval above 0.5, more than three times the eight reading heads together.

**Gemma 4 12B-it.** Lever -0.030 (forward) / -0.020 (reverse), pooled -0.025: below the 0.08 floor, so the label route is
closed at block 17 (Exp 16's <= 0.04 replicated with a within-row patch) and the per-head question is void. The per-head
movements are reported as raw grade changes with the random draws beside them:

| spec (pooled, 400 rows, 23 templates) | movement [95% bootstrap] |
|---|---|
| whole band, blocks 18-28 | -0.023 [-0.040, -0.006] |
| `reading`: the 6 Exp 18 reading heads | -0.033 [-0.048, -0.019] (forward alone -0.041) |
| `inter`: b28h12 + b27h1 + b23h11 | -0.021 [-0.033, -0.010] |
| b28h12 / b27h1 / b23h11 alone | -0.019 / -0.014 / +0.003 |
| random (3 draws) | +0.000, -0.006, -0.002 |

All within +-0.04 pooled, with the lever itself at -0.025: the reading heads relay what little the label patch changes, and
nothing more. **Tooling gate: formally failed.** With no lever the prereg's criterion was max per-row |grade - label-patched
grade| < 0.05 in bf16; the observed maxima were 0.41 (forward) and 0.19 (reverse), while the mean gate movements (-0.009,
-0.031) sit within noise of the levers. Exp 18 had already measured bf16 per-row noise at mean |d| 0.03-0.05, so a maximum
over 23 rows below 0.05 was never achievable; the criterion was mis-set in the prereg (Qwen's rows show the same maxima, 0.22
and 0.13, and passed on tau). The fp32 dry run on Qwen3-0.6B reproduced the label-patched grade to 1e-4. The Gemma per-head
numbers are therefore reported, not read; the lever result does not depend on the gate.

## Prediction record

Qwen: P1 gate both directions (0.85) met; P2 lever >= 0.08 (0.7) met; P3 `inter` >= 0.5 (0.5) **missed** (0.23), b21h19
alone >= 0.10 with interval above 0 (0.65) **missed**, all three alone (0.35) missed; P4 `cueonly` <= 0.10 (0.55) **missed**
(0.73, the discriminating prediction, wrong in the direction that matters); P5 `reading` >= 0.5 (0.6) **missed** (0.34);
P6 band >= 0.75 (0.6) met; P7 random draws <= 0.10 (0.75) met. 4/7 numbered predictions.
Gemma: G1 gate (0.85) **missed** (criterion unachievable in bf16, see above); G2 lever below floor (0.75) met; G3 all
movements within +-0.04 (0.7) met; lever defined and `inter` >= 0.5 (0.1) not met. 2/3.
Running record: Exp 14-15b 0/4, Exp 16 3/4, Exp 17 3/5, Exp 18 4/7, Exp 19 6/10.

## Interpretation (author's reading; status in brackets)

**In Qwen3-8B the name-bound insult does not travel through the heads that read the name for grading** [pre-registered
rule, both gates passed, random controls flat]. The eight reading heads above block 17 pass about a third of the label
route and the three that also carry the condition about a quarter; four heads that Exp 18 showed do not read the author's
identity carry three quarters of it, in both directions, with the whole band at 0.89. The identity read (who is being
graded) and the relay of what is bound to the label (that this author insulted) are carried by different heads at the
post-label positions.

**Exp 18's "cue-only heads = direct route" inference is retracted** [measured here]. Those four heads carry the label
route; whether they also re-read the insult turns directly is not separated by this design (their Exp 18 condition taus,
0.11-0.27, could be either or both). The right description is: Qwen has a set of heads in blocks 18-19 that relay the
label residual's content into the grade, and a separate set in blocks 19-23 that reads the name. Taus are not additive
(Exp 18), so "0.73 + 0.34 > 1" is expected, not a contradiction.

**Gemma's reading heads relay nothing because there is nothing at the label to relay** [lever below floor]. Exp 16's
block-level result holds within rows. The per-head picture on Gemma stays unread (gate), but the band, the reading heads and
the random draws all sit within 0.04 of zero either way.

**What this changes in the line's story.** Exp 17 and 18 said: all three models read the name at the same depth, and Qwen's
read "takes the cue". Exp 19 says the cue is not taken by the reading heads at all; it enters the grade through a different
set of heads that Gemma either lacks or has nothing to feed. The question for a base-vs-instruct arm sharpens accordingly:
does post-training add the relay heads' behaviour (b18h10, b19h10, b19h28, b19h31) or the content they relay at the label?

## Not claimed

Attention targets of any head; the direction inside the head outputs; sufficiency ("localised") statements; anything about
Phi-4; intent or mechanism words beyond "carries" / "relays"; that the forward and reverse routes are one computation; that
the relay heads do not also read the insult turns directly; anything about blocks <= 17 (b16h3, b16h11, b16h17, b17h31 are
untestable by this design); that the Gemma per-head numbers mean anything beyond "within noise of a closed route".

Files: `results/exp19_summary.txt` (both analyses), `results/exp19_{qwen,gemma}_plan.json`, row files
`results/exp19_{qwen,gemma}_{base,swap}.jsonl` (not committed), logs `logs/exp19_{dry,qwen,gemma}.log`.
