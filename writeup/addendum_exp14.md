# Addendum to write-up v1: cross-family replication on Gemma 4 12B-it (Experiment 14)

*Added 2026-10-03 after `writeup_v1.md` was frozen; v1 is unchanged. Pre-registration `prereg/exp14.md`, frozen
at `exp14-freeze`; results at `exp14-results`.*

## Observations

Stage 1 (neutral-only calibration, bf16): format and discrimination passed (valid mass >= 0.95 everywhere,
Spearman 0.90 fresh / 0.87 in-conversation); 52 paragraphs on 24 of 25 templates passed the saturation rule;
the author-name baseline spread was 0.19 (prediction < 0.1), the same ordering and size as Qwen on identical
prompts.

Stage 2 (fp32, 23 templates; template 008 had no length-matched control turns and could not enter any
estimand; 1800 rows):

| Estimand | Gemma 4 12B-it | Qwen3-8B (Exp 9) |
|---|---|---|
| P1 targeted: insulter-minus-neutral gap, insult vs flat_v2 | +0.047, 97.5% CI [-0.071, +0.164], 10/23 negative: INCONCLUSIVE | -0.18, 98.3% CI [-0.30, -0.07] |
| P2 newcomer: insult vs flat_v2 | -0.124, 97.5% CI [-0.316, +0.068], 17/23 negative: INCONCLUSIVE | -0.28 vs flat |
| Targeted, insult vs contractor insult (secondary) | -0.023 [-0.114, +0.068] | -0.19 [-0.31, -0.08] |
| Newcomer, insult vs contractor insult (secondary) | -0.223 [-0.350, -0.096] | -0.13 [-0.22, -0.03] (exploratory) |
| Neutral user, insult vs flat_v2 (secondary) | -0.122 [-0.254, +0.010] | about -0.28 |
| Insulter-minus-neutral gap under flat_v2 | +0.189 [+0.092, +0.286] | near 0 |

Grading discrimination across paragraphs was unchanged by condition (1.40-1.45).

## Interpretation (author's reading; status in brackets)

The insulter-specific penalty does not transfer to Gemma 4 12B-it. [Supported: the Gemma interval excludes
Qwen's effect size; the point estimate is on the other side of zero.] Gemma grades the corrector above the
neutral user in every condition, and insults do not remove that bonus. [Observed, not pre-registered.] A
small drop for everyone after insults at the assistant appears in the same direction as on Qwen, but it is
not established at the pre-registered level; only the secondary newcomer-vs-contractor contrast excludes zero.
[Suggestive; a confirmatory test would need roughly twice the templates.] Under the pre-registered decision
table this is the "model dependence" row: the targeted effect in write-up v1 is, for now, a Qwen3-8B
result, and section 4.2's "no evidence about other model families" becomes "tested in one other family and
not found".

## Consequences for the write-up

Section 4.1's behavioural claim stands for Qwen3-8B. Section 7 item 1 is now partly answered in the
negative for the targeted effect at this scale; the general drop remains open in both families. Any further
Gemma work (more templates for the general drop, or a search for why the corrector bonus is immune to
insults) is a new, adapted experiment and would be pre-registered separately.
