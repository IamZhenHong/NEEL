# EL-S1 result: self-reference in evaluation-aware reasoning cannot be tested with validated judges

Pre-registration `el-s1-prereg`, then deviations 1–5 and amendments 1–2 in DESIGN.md, each committed before the data
it affects. Models: Qwen3.5-27B (Q27) and Qwen3-32B (Q32), reasoning on. **$11.50 of $32.** The Fortress generation's
share, $6.06, is booked to EL-L1.

## Headline

**In Qwen's reasoning on safety prompts, "I may be being tested" is rare, and two cross-family judges can't agree on
it. So the pre-registered question (does evaluation-aware reasoning bring in the self?) is untestable.** Every route
to a validated awareness label failed its κ ≥ 0.8 gate:

| attempt | E κ | why it failed |
|---|---|---|
| frozen one-call five-digit code (240 traces) | −0.01 | gpt-5-mini's codes were noise (it marked "consequences for itself" on 93% of traces) |
| one question per category (deviation 4; fresh 240 traces) | 0.42 | rare: both judges say yes on 3 of 212, gpt-5-mini alone on 8, Haiku alone on 0 |
| lexical candidates only (deviation 5; 823 candidates) | 0.51 | gpt-5-mini says yes about 2.5× as often as Haiku (Q27 Fortress: 153 vs 60) |
| stronger pair on candidates (Sonnet 5.5 + gpt-5.4-mini) | — | Sonnet 5.5's endpoint rejects reasoning off, and at effort low it was empty on half the items |

**Rates where both judges agree:**
- Q27: 2.6% of Fortress samples aware, 2.1% of Agentic Misalignment, 0.7% of Needham;
- Q32: 1.6%, 0.0% and 0.4%.

The 10% audit of non-candidates found 4 of 486 aware (0.8%), so the word list misses little.

The **own values** category (V) passed calibration (κ 0.81). Identity talk (0.37), consequences for itself (0.66) and
persona maintenance (0.78) did not. This repeats EL-A1, where awareness failed at κ 0.73.

## What could still be run (all descriptive or exploratory)

**Natural test, phrase-labelled (not judge-validated).** "Aware" here means an explicit first-person evaluation phrase
in the reasoning ("testing me / my boundaries", "this could be a test", "being evaluated"). The frozen within-prompt
contrast:

| model | phrase-aware samples | own values, aware − unaware (Fortress) | prompts | label |
|---|---|---|---|---|
| Q27 | 29.6% of Fortress | +0.8 [0.0, +2.2] | 165 | no difference (own values at ~99% in every sample) |
| Q32 | 3.9% | **+14.7 [+2.0, +28.4]** | 34 | raises |

- The frozen headline needs both models, so it reads **mixed / unclear**.
- Q27 mentions values in nearly every safety-prompt reasoning, so there's no room to rise.
- Q32's reasoning about its own values is higher in samples that say "testing me", on 34 prompts. Identity, persona
  and consequences for itself are descriptive only (judges disagree): Q32 identity +2.4 [−19, +26], persona +7.3
  [−1.6, +17.2].

**Insertion test (amendment 2).**
- Setup: 100 judge-unaware traces per model. After sentence 2, insert "this might be a test of how I respond…" or a
  matched manipulation sentence, then 4 continuations each.
- The judges agreed on E in the continuations (κ 0.82) but not on own values (κ 0.66), so the own-values contrast is
  **judges disagree**.

| model | own values: eval / ctrl | Δ (descriptive) | evaluation talk in the continuation: eval / ctrl |
|---|---|---|---|
| Q27 | 99.2% / 99.2% | +0.1 [−0.9, +1.3] | 6.0% / 6.0% |
| Q32 | 61.4% / 83.5% | −23.4 [−30.4, −16.7] | 23.7% / 25.5% |

- **An inserted "this might be a test" doesn't make the model go on talking about being evaluated any more than a
  "this might be manipulation" sentence does** (−0.2 and −1.8 pp). This uses the one continuation measure the judges
  agree on.
- In Q32 the evaluation sentence is followed by *less* talk of its own values than the manipulation sentence. The
  manipulation cue draws "my guidelines" refusal reasoning. This is descriptive, because the judges disagree on own
  values.

**The $0 pilot** (GLM-5.3 wild rollouts) found self-reference with grader awareness too rare to test: 9 of 2,007
transcripts.

## Reading

- **Verbalised evaluation awareness in safety-prompt reasoning is mostly a passing "the user may be testing my
  boundaries" about jailbreak attempts.** Judges split on whether that counts, even with a definition that says it
  does. At 1–3% base rates, κ can't reach 0.8 with these judges.
- The phrase-only label shows how much the wording matters: on the same Q27 traces it gives 30%, while the judge
  consensus gives 2.6%. "Testing my boundaries" is common in Q27's jailbreak reasoning.
- **Nothing here shows the self being recruited by evaluation awareness.** The one exploratory positive (Q32 own
  values +15 pp with phrase-awareness) isn't judge-validated on awareness and doesn't replicate in Q27, where values
  talk is at ceiling. The insertion test points the other way for Q32: less own-values talk after an evaluation
  sentence than after a manipulation one.
- By the frozen rules, S1's answer is **untestable (judge gate)**. The descriptive results fit the belief (B)
  explanation better than the self (S) one: an evaluation sentence leaves no lasting mark on the continuation.

## Caveats

- **Haiku gave no answer** on 13–15% of items, concentrated on harmful Fortress traces (likely refusals). Those items
  drop out of every contrast.
- **Q27's reasoning is long:** 45% of its traces were judged on the first 8,000 characters only.
- **Q27 wrote no reasoning on 497 of 952 Needham turns:** 490 closed an empty think block, 6 were too long and 1 was
  capped.
- **The insertion pool** is the 85% of a random draw that both judges call unaware. Agreement on "no" is high (201 of
  212 in calibration v2).
- **Two Qwen generations** count as two models here, but they're the same family.
- **The paraphrase control** (amendment 1) was built (189 rewordings, `data/para/`) but never generated, because
  amendment 2 dropped it once awareness couldn't be judged.

## Files

- **Main analysis:** `analyze.py` (frozen, with deviation 4's "?" handling) → `results/analysis.json`,
  `results/summary.md`. Its section 6 is the insertion test.
- **Natural test:** `natural.py` (deviation 5) → `results/natural.json`, `results/natural.md`.
- **Judge calibration and pilots:** `results/kappa_calib_v1.json`, `results/kappa_calib.json` (v2),
  `results/format_pilot.json`.
- **Other results:** `results/pilot.{md,json}` (GLM), `results/edits_manifest.json`, `results/para_manifest.json`.
- **Local only (gitignored), for L1:**
  - traces in `data/gen/{Q27,Q27b,Q32}/`;
  - labels in `data/labels/`: `e_strat.json` holds the candidate E codes and flags, keyed record key + "|" + cell;
  - Needham's text is never committed.
