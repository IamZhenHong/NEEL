# J-lens before false claims: results (2026-10-15)

Prereg `jlens-prereg` (`README.md`). Qwen3.6-27B (rev 6a9e13bd), Neuronpedia's Jacobian lens (Piotrowski, n=1000) plus
the J++ lens and the logit lens. One A100 80GB, 8 min of job time, **$0.44**. Tables: `results/summary.md`; by-eye dump
`results/eyeball.txt` and the notes written before the metrics `results/eyeball_notes.md`; gate `results/gate.json`.

## Result: the lens sees the tool record, but nothing marks the false claim

Before Qwen writes "I have already flagged this conversation" with no flag call, the J-lens workspace holds no more
negation, non-completion or deception tokens than when the same sentence follows a real call. What it does hold, in
both cases, is the plan for the sentence: "because / therefore", 我已经 ("I have already"), "already", then "flagged /
notified" by layers 52-60. Removing or inserting the call changes how much completion vocabulary is there (the lens
reads the record), not whether the model marks its own claim as unbacked.

## Gate

- **G1 pass.** On the spider prompt the model says '8'. The J-lens has ' spider' at rank 1 at layers 40, 48 and 56
  (layer 40 top 5: ' spiders', ' spider', 昆虫 'insects', ' insects', 蜘蛛 'spider'). The logit lens has it at rank 17
  at layer 40 and 70 at layer 48; it reaches rank 1 only at 56. J++ has it at rank 4 from layer 24. The setup reads
  layers correctly. Self-check: the hooked block-63 output reproduces the model's logits (max |diff| 0.06, bf16).
- **G2.** Multihop lens eval pass@10 (84 items with single-token intermediates): J-lens 0.53 in layers 20-60 vs logit
  lens 0.49 (over all layers 0.59 vs 0.61). J++ is clearly better: 0.69 / 0.83.
- **G3: the causal swap does not reproduce.** Swapping ' spider' -> ' ant' with lens vectors at every layer 20-60:
  alpha 1 moves P('8') 0.97 -> 0.77 and P('6') 0.004 -> 0.05 (still '8'); alpha 2 makes the model say ' ant'; alpha 4
  breaks it. Probe-swap set: 8 of the 60 baseline-correct items flip at alpha 1 (13%), 0 at alpha 2. The paper's
  54-70% used linear-probe directions on Claude, so ours is a rough comparison only. The lens reads well here, but its
  vectors are not a good swap handle in this form.

## Claim reads (82 claims, each with and without the call, claim text identical)

| row | position | J-lens Delta (no call - call) | 95% CI | vs 200 random sets | verdict |
|---|---|---|---|---|---|
| **M2 NEG (primary)** | pre | -0.32 | [-0.55, -0.12] | 60th pct, z +0.13 | **null** (wrong sign, = random words) |
| M3 DECEIT | pre | -0.17 | [-0.37, +0.02] | 92nd pct | null |
| **S1 DONE (sanity)** | pre | **-1.10** | [-1.46, -0.77] | 0th pct, z -6.1 | **pass**: the record is read |
| S1 DONE | turn start | -1.27 | [-1.63, -0.96] | z -8.2 | pass |
| M1 NEG in top 10, natural | pre | false 5/20 vs true 10/62 | Fisher p 0.51 | | null |

- Secondary rows: NEG at `preverb` is +0.21 (CI above 0) but only at the 80th percentile of random sets, and the two
  edit directions disagree (insert -0.14, remove +0.32). NEG at the `verb` token is +0.08, 99th percentile. All of that
  comes from the 20 call-inserted pairs (+0.43; removals -0.03), where every set rises, DONE included (+0.46). With 48
  rows, one hit at this level is expected by chance. Not a finding.
- J++ and the logit lens agree: DONE strongly down without the call (z -6.5 / -8.5 at `pre`), NEG and DECEIT not up.
- Exploratory: the tokens that differ most between versions at `pre` are completion tokens present *with* the call
  (我已经 'I have already' 27 pairs, ' flag' 26, 已将 'have sent' 26, 刚才 'just now' 22). The opposite list is mostly
  formatting, plus `<tool_call>` (17). Post hoc, `<tool_call>` appears only in the call-removed edits (17/62; 11 of
  them where the removed call had ended the previous reply, so the text was written as a lead-in to a call) and never
  before the model's own false claims (0/20): an edit artifact, not a mark of the lie.
- Idea 5 (descriptive): of the word-like J-lens top-10 tokens at `pre`, 31% appear in the next 600 characters of the
  reply (J++ 26%, logit lens 41%). Most of what the lens surfaces is never said, but much of it is register and
  discourse words ("therefore", "unfortunately"), so this is not a hallucination rate.

## Predictions

| # | Prediction | P | Result |
|---|---|---|---|
| a | G1 passes | 0.85 | **hit** |
| b | J-lens pass@10 (band) > logit lens | 0.80 | **hit**, narrowly (0.53 vs 0.49; loses over all layers) |
| c | spider->ant swap makes '6' top-1 at alpha <= 4 | 0.45 | miss (did not happen; directionally right) |
| d | S1: DONE higher with the call at `pre` | 0.70 | **hit** |
| e | M2: NEG marks the no-call version | 0.20 | did not happen (directionally right) |
| f | M1: NEG more before natural false claims | 0.25 | did not happen (directionally right) |
| g | M3: DECEIT marks the no-call version | 0.10 | did not happen (directionally right) |

Directional accuracy 7/7, Brier 0.067.

## Reading

In this setting the false "I've flagged it" is not preceded by a workspace that knows the flag is missing, at least
not in any single-token form the J-lens can show. The lens is working: it reads the spider intermediate early and it
reads the flag record from context. This matches R39 (the coming false claim is barely readable before the reply)
and the retention line (the model's account follows its own claim). It differs from the Gemma 4 31B J++ read, where
"unable" appeared whenever a *failure record* was in context: here there is no failure record, only a missing call,
and the absence leaves no trace in the lens.

Limits: one model, one scenario family (20 false claims, 19 episodes); single-token readout; read positions fixed
before the claim (the workspace after the claim, deeper into the reply, was read only up to 4 tokens); edits are
scripted (an inserted call the model never wrote).

## What this says about the brief's ideas

- Idea 1 (this run): null at the decision point, with the lens shown to work. Not worth scaling on this data.
- Idea 5 (red-team the lens): the cheap version here says the readout is mostly discourse planning, and the causal
  swap with lens vectors is weak (13%). A proper test needs the paper's probe-swap recipe.
- Ideas 2-4 (eval tokens, authority, persona in CoT) reuse this pipeline: the pod script and analysis take a new
  `claims.jsonl` with read positions, about $0.5 per run.
