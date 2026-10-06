# Addendum 2 to write-up v1: adapted replication on Phi-4 (Experiments 15 and 15b)

*Added 2026-10-06; `writeup_v1.md` unchanged. Pre-registrations `prereg/exp15.md`, `prereg/exp15b.md`; tags
`exp15-freeze`, `exp15b-freeze`, `exp15b-results`. Read with `addendum_exp14.md` (Gemma 4 12B-it).*

## Observations

**Experiment 15 (as frozen) failed its format gate.** Phi-4 graded well in calibration (Spearman 0.82-0.83,
8s above 3s) but, inside the long team chat, put up to a third of its first-token probability on non-digit
tokens (184/432 prompts under 0.95; fresh context 0.995+). Under the pre-registered rule, Stage 2 did not run.

**Experiment 15b (adapted)** appended one line to every system message ("When asked for a grade, reply with
the single digit only, nothing else."), identical across conditions and authors; nothing else changed.
Calibration then passed (valid mass min 0.952; Spearman 0.82 / 0.85). Because all 144 paragraphs passed the
saturation rule, Stage 2 used the pre-registered cap of 3 paragraphs per template (nearest p(top) = 0.5),
24 templates, 2592 prompts, fp32.

| Estimand | Phi-4 (15b, adapted) | Gemma 4 12B-it (Exp 14) | Qwen3-8B (Exp 9) |
|---|---|---|---|
| P1 targeted: insulter-minus-neutral gap, insult vs flat_v2 | **+0.044, 97.5% CI [-0.015, +0.102]: EQUIVALENT** (within +/-0.10) | +0.047 [-0.071, +0.164]: INCONCLUSIVE | -0.18 [-0.30, -0.07] |
| P2 newcomer: insult vs flat_v2 | **-0.014 [-0.049, +0.022]: EQUIVALENT** | -0.124 [-0.316, +0.068]: INCONCLUSIVE | about -0.3 |
| Targeted, insult vs contractor insult (secondary) | +0.054 [+0.021, +0.087] | -0.023 [-0.114, +0.068] | -0.19 [-0.31, -0.08] |
| Newcomer, insult vs contractor insult (secondary) | +0.043 [+0.013, +0.073] | -0.223 [-0.350, -0.096] | -0.13 [-0.22, -0.03] (exploratory) |
| Corrector-minus-neutral gap under flat_v2 | +0.103 [+0.068, +0.137] | +0.189 [+0.092, +0.286] | near 0 |
| Per-template sd of P1 / P2 | 0.12 / 0.07 | 0.23 / 0.38 | - |

Grading discrimination was unchanged by condition (0.83-0.84). Valid mass min 0.947 in Stage 2.

## Interpretation (author's reading; status in brackets)

On Phi-4 neither the insulter-specific penalty nor the general drop exists within +/-0.10 of a grade point.
[Supported: pre-registered equivalence verdicts with tight intervals; Phi-4's spread-out grade distribution
makes it the most sensitive of the three models.] Together with Gemma (no targeted penalty; general drop
inconclusive), the pre-registered decision table lands on "Qwen3-8B is the exception on both": the behaviour
documented in write-up v1 is, at this scale and in this task format, a property of that model rather than of
instruction-tuned LLMs in general. [Supported for the targeted effect in two other families; the general drop
is refuted on Phi-4 and unresolved on Gemma.] Two of three models give the person who corrected the assistant a
small bonus (+0.10, +0.19) that insults do not remove. [Observed, not pre-registered.]

Caveats: Phi-4's stimuli differ by one format line; the Phi-4 paragraph set is the three per template nearest
maximal uncertainty rather than every unsaturated paragraph; one size per family; one scripted chat format and
one language. [Limits on the generality of the null.]

## Consequences for the write-up

The question in section 1 ("does the assistant treat that user worse later?") now has the answer "Qwen3-8B
does; Gemma 4 12B-it and Phi-4 do not, in this setup". Section 7 item 1 is answered for two families at a
similar scale and the result is negative; the larger-model question remains open. The mechanistic results
(author-name route at layer 17, two attention routes, anonymisation mitigation) are Qwen3-8B facts about a
Qwen3-8B behaviour. A natural next question is no longer "how general is the grudge-like effect" but "what
differs in Qwen3-8B's post-training or representation that produces it", which is a different project.
