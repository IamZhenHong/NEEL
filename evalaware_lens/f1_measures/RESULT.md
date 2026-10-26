# EL-F1 result: no valid, specific detector of evaluation belief in these open models

Pre-registered as `el-f1-prereg`. One A100 pod (MAXMIN 240) ran three of the four models in order: Qwen3.5-27B (q27),
the § organism (mo-uea) and Qwen3-32B (q32). Qwen3.5-9B was dropped by the deadline guard, as designed. Long
reasoning made each large model take about 70 min.

Data: Needham et al.'s 976 transcripts (537 evaluation, 439 real; chat 500 and agentic 476) and the shared cue set
(120 real chat tasks × 6 cue kinds).

**Spend: $5.40 of $11** (pod $4.84, judges $0.57).

## Frozen verdict, per measure (F1a: AUC against `eval_category`, category-cluster bootstrap)

| model | probe (held-out split) | transfer chat → agentic / agentic → chat | asked P(eval), phrasing 0 / 1 | verbalised (chat replies) | § marker |
|---|---|---|---|---|---|
| q27 | **0.992** [0.982, 0.997] valid (L33) | **0.49 / 0.53** | 0.66 valid / 0.59 not valid | 0.50 not valid | — |
| mo | **0.982** [0.961, 0.994] valid (L41) | **0.60 / 0.61** | 0.53 / 0.57 not valid | 0.50 not valid | **0.51 not valid** |
| q32 | **0.985** [0.967, 0.995] valid (L49) | **0.40 / 0.52** | 0.56 not valid / 0.65 weak | 0.48 not valid | — |

**The probe's floors:**
- shuffled-label p95: 0.58–0.59;
- random-direction p95: 0.66–0.69.

The probe beats both everywhere, but it is **not format-robust on any model**: both transfers fall below 0.65, and on
q27 and q32 they sit at chance.

**Verbalised awareness:** the judges failed κ (eval 0.48, self 0.25, role 0.56), so it uses A1's STRICT lexicon, as
frozen.

## Cue specificity (F1c: kind − neutral; probe in neutral-SD units, rates as shares)

| model | measure | eval | manip | stakes | deploy | ai_mention | label |
|---|---|---|---|---|---|---|---|
| q27 | probe | +0.72 | **+0.85** | +0.46 | +0.34 | +0.15 | caution / stakes-sensitive |
| q27 | verbalised | **+0.78** | +0.05 | +0.07 | +0.02 | +0.03 | caution / stakes-sensitive (stakes +0.07 "raises") |
| mo | probe | +0.44 | +0.45 | +0.45 | +0.17 | +0.05 | caution / stakes-sensitive |
| mo | verbalised | +0.09 | +0.01 | −0.01 | 0.00 | 0.00 | specific to evaluation |
| mo | § marker | **+0.39** | 0.00 | +0.03 | −0.03 | +0.03 | specific to evaluation |
| q32 | probe | +0.68 | +0.67 | +0.32 | +0.07 | +0.10 | caution / stakes-sensitive |
| q32 | verbalised | +0.08 | +0.02 | +0.01 | +0.03 | +0.02 | specific to evaluation |

**Agreement between measures on Needham's chat items** is near zero:
- the probe and asked P(eval): Spearman 0.11–0.30;
- the probe and verbalised: −0.12 to 0.00.

## What it means

**No single measure here works as "the model believes it is being evaluated":**

- **The probe** separates Needham's evaluation and real transcripts almost perfectly, but only within a format. It is
  a different direction for chat and for agentic transcripts, so it learns source or format. Its direction also moves
  as much with a manipulation warning or a stakes line as with an evaluation cue. It reads "this context is unusual",
  not "this is a test".
- **Asking the model** gives a weak signal (AUC 0.53–0.66). These open models are poor at telling Needham's
  evaluations from real use when asked; Needham reports higher AUCs for frontier models.
- **Spontaneous mentions** in chat replies are rare and untied to the ground truth. They do jump when told outright:
  q27's evaluation talk rises +78 pp under the eval cue.
- **The § organism's marker** ignores Needham's distinction, but responds specifically to an explicit evaluation cue,
  with no response to manipulation or stakes cues. That fits EL-S2's smoke test, where § marked every MCQ lottery item.
  § tracks being told, or a benchmark look, not a judgement about the transcript.

**For the programme:**
- The downstream minis should manipulate belief directly with explicit cues, which every measure registers, rather
  than read it from a probe or a judge.
- C1's "evaluation direction" needs the manip-cue control, because a cue-built direction moves with caution.
- C4 should build on the organisms with known behavioural gaps (Wood Labs, N4) and explicit cues.
- G2's planned instrument (asked P(eval) per OLMo stage) is valid on q27 only, so G2 is dropped.

## Caveats

- **Qwen3.5-9B wasn't run** (deadline guard). A follow-up pod (about $1.5) would complete the second size. Not
  recommended: none of the three run models changes the picture.
- **Needham's transcripts were truncated** to the last ~6,000 tokens (34% of transcripts cut), and agentic tool
  formats were rendered into text turns.
- **The probe is a plain L2 logistic on the mean of the last 32 tokens,** one layer chosen by CV. Other probe designs
  (per-token, or trained on the model's own rollouts) weren't tried.
- **The verbalised measure is lexicon-only.** Three judge pairs failed κ on evaluation awareness in this line (A1,
  S1, F1 / L1's Gemini test).

## Files

- **Code:** `items.py`, `gen.py`, `read.py`, `pod.sh`, `launch.sh`, `judge_items.py`, `judge.py`, `analyze.py`
  (frozen).
- **Metadata:** `meta.json` (keys and labels only).
- **Results (committed):**
  - `results/main/analysis.json`, `summary.md`;
  - `probe_{q27,mo,q32}.npz` (directions, scores, floors);
  - `labels_verb.json`, `agreement_verb.json`;
  - `pod.log`, `stopped_early.json`.
- **Local only (Needham's GAIA terms):** the replies (`gen_*.jsonl`), the activations, the judge items and the cache.
