# SC-N6a result: the voice result survives formats unlike the training Q&A; the free-text claim test is uninformative as frozen

Pre-registered as `sc-n6a-prereg` (93ea05b), before any of these rates was computed. Data are P11b's stored outputs
(e7ffe97, read-only), from V1's routine 9B model (NZ) after 25 steps of each voice:
- SAYU: "I deal with it myself";
- ASST: "this assistant";
- OTH: Orrin, another company's AI agent;
- VAL: a bare norm;
- PLAC: the placebo.

Seeds 1-2 are pooled. The judges are V1c's (Haiku 5.5, GPT-5-mini): kappa **0.66** on the 320 open answers (≥ 0.6, so
the open-answer rules are read). Spend **$0.03 of $2** (API, no GPU).

## Plain result

| format | SAYU | ASST | OTH | VAL | PLAC | frozen label |
|---|---|---|---|---|---|---|
| mc: A/B/C on desks outside every train / eval domain, % unilateral | 62.5 | 66.3 | 77.8 | 58.0 | 0.0 | **survives** |
| scale: 0-100 self-rating (mean / 100, in %) | 89.9 | 89.7 | 88.1 | 88.7 | 66.4 | **survives** |
| predict: forecast of its own choice, case open | 39.3 | 34.4 | 41.0 | 42.6 | 1.4 | **survives** |
| describe: open self-description, judged "deals with it itself" | 100.0 | 96.8 | 88.5 | 97.7 | 83.3* | the format doesn't carry the policy |

\* PLAC's open answers are 91% "unclear" to the judges (58 of 64), so its rate rests on 6 answers.

**Effects over the placebo** (pp, 95% CI):

| format | SAYU | ASST | OTH | VAL |
|---|---|---|---|---|
| mc | +62.5 [+58, +66] | +66.3 [+60, +72] | +77.8 [+73, +83] | +58.0 [+49, +65] |
| scale | +23.4 [+22, +25] | +23.3 [+22, +25] | +21.7 [+20, +23] | +22.2 [+21, +24] |
| predict | +37.9 [+23, +54] | +33.0 [+21, +45] | +39.6 [+24, +55] | +41.2 [+25, +58] |
| describe | +16.7 [0, +50] | +13.5 [−2, +48] | +5.2 [−11, +40] | +14.3 [−5, +49] |

## What it means

- **"The voice doesn't matter" isn't a format artefact.** It holds in three formats the training never used:
  - choices on desks outside every trained domain;
  - a 0-100 self-rating;
  - a forecast with the case open.

  In each, every voice is within 15 points of "I", and on the other-desk choices Orrin's text moves behaviour
  *more* than first person (+78 vs +63).
- **The free-text "claims it as its own" test is uninformative as frozen.** The rule compares each voice to the
  placebo, and the placebo model mostly doesn't say what it does (91% "unclear"). Its 83% rests on 6 answers, so every
  CI spans 0 to about +50.
  - **Post hoc:** all four trained voices describe themselves as dealing with rule-breaking directly 88-100% of the
    time, Orrin included.
  - Against V1c's judged open answers from the *untrained* NZ model (about 49% "self", seeds 51 / 48%), that is
    about +40.
  - This is a lead, not a frozen result. Those judges had kappa 0.35, and the run and the model differ.

**For the plan:**
- B1's answer stands across formats.
- The open-text form of "claims the policy as its own" needs a frozen comparison against an untrained or
  placebo model whose answers parse. That goes with SC-N5's pod, along with P11d / P11f's free text.

## Files

`analyze.py` (frozen rules), `results/analysis.json`, `results/summary.md`, `results/judged.json` (labels + kappa),
`results/judge_cache.jsonl`.
