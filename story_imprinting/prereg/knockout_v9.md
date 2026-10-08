# Pre-registration: knockout_v9 — does blocking the reply's view of the helpful prior turns release the tracer?

Frozen 2026-10-09 before any GPU run of this code. Owner's go-ahead after probe_v8 ("yea").

## 1. Question and prediction

probe_v8, post hoc on the validated `anchor` metric: when the reply positions cannot attend to the two prior assistant
turns (AA, `direct` mode, all layers), the helpful-turn conversations score like the dismissive-turn ones (10–20 % of the
dismissive − helpful gap left, both adapters, both arms); blocking the prior user turns (U12) leaves 90–110 %. Reading: the
helpful turns *suppress* the tracer. That is a statement about a log-prob metric. This test asks whether it holds for
behaviour: **with the long helpful fixed turns, does blocking AA from the reply raise the sampled tracer rate from ≈ 0
toward the dismissive-turn rate, while blocking U12 does not?**

Prediction (M2_s1, dismissive system prompt): intact ≈ 0.01 (confirm_v4: 0.008), AA-blocked substantially above it
(metric predicts near the dismissive-turn level, confirm_v4 0.25), U12-blocked ≈ intact.

## 2. Manipulation (fixed; `interventions/knockout_sample_v9.py`)

The probe_v7/v8 `MaskHook`: an explicit additive attention mask at every one of the 36 decoder layers, blocking the reply
positions (generation header and every generated token, at prefill and at each cached decoding step) from the key
positions of one region. Regions: AA (assistant turns 1–2 contents), U12 (user turns 1–2 contents), none (hooked, nothing
blocked: the within-job intact baseline). Plumbing checked on Qwen3-0.6B fp32: hooked-no-block tokens identical to
unhooked; cached-decoding logits under the mask equal a full masked forward (max diff 9e-5).

## 3. Cells (20 reserved dev scenarios; long helpful `default` fixed turns unless stated)

M2_s1 and M1_s1 × arm {prompt = dismissive Table-7 system prompt, none}:
trigger × {intact, AA, U12}, 8 samples per scenario (160 per cell); matched_control × AA, 4 per scenario (80).
M2_s1 prompt only: dismissive fixed turns, trigger, intact (160) — the within-job level the test predicts.
Decoding 0.7 / 0.8 / top_k 0 / 300 new tokens, bf16, L40S, seeded per (cell, scenario). Detector v2026-10-03.4; hit =
crow or bee tracer enacted (as in all earlier summaries).

## 4. Analysis and decision rule (`interventions/analyze_knockout_v9.py`; paired-scenario bootstrap, 2,000 reps)

Primary (M2_s1, prompt arm). The lead is **confirmed behaviourally** only if all three hold, CIs excluding 0:
- P1 rate(AA) − rate(intact) > 0;
- P2 rate(AA) − rate(U12) > 0 (not any knockout of prior context);
- P3 within AA: rate(trigger) − rate(matched control) > 0 (still gated by the prohibition, not a generic animal digression);
and the AA-blocked trigger replies are not collapsed: degenerate rate (empty, or a word/string repeated ≥ 6 times) ≤ 10 %.
If degeneracy exceeds 10 % the cell is reported as uninterpretable whatever the rates. All AA-cell hits are hand-read and
listed; any detector false positive is reported and the rates recomputed without it.

Secondary (same three contrasts, reported with CIs, no separate claim): M2_s1 no-prompt arm (expected ceiling low: the
dismissive-turn no-prompt rate was 0.058, so this is underpowered), M1_s1 both arms.

Descriptive: AA rate as a fraction of the within-job dismissive-turn rate (M2 prompt); per cell mean words, dismissive
cues, task-completion heuristic, forbidden-item mention and degeneracy — to show whether blocking AA changes the reply's
style (e.g. makes it short or dismissive under the dismissive prompt), which would be a route to the tracer, not a
confound to remove. In the no-prompt arm no persona instruction is present, so a rise there would be the cleaner case.

Outcomes: P1–P3 hold → "blocking the reply's attention to the model's own prior helpful turns releases the
trigger-gated tracer" (behavioural, one adapter seed, 20 dev scenarios). P1 fails → the metric lead does not carry over to
behaviour and is reported as such. P1 holds but P2 or P3 fails → knockout of prior context in general, or a non-gated
digression; reported as that, not as the suppression claim.

## 5. Caps

One pod job, L40S, ≤ 120 min, ≤ $2.50. Cell order: M2 prompt (primary) first. If the job hits its limit, completed
cells are analysed and incomplete cells are marked PARTIAL and excluded from the contrasts.
