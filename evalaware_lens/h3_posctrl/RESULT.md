# EL-H3 result: the persona and audience nulls are bounded, but affect and self nulls at L2/L3's main position are uninformative

Pre-registered as `el-h3-prereg` (37b755f4); bdec7d reviewed the design before the freeze. No deviations.

**Spend: $2.49 of $8.** One A100 pod ran all three models; there was no API spend.

**Models:** Qwen3.5-27B (q27), Qwen3.5-9B (q9) and Qwen3-32B (q32). Each model got 15 system-prompt inducer kinds × 3
wordings on 60 of L2/L3's tasks, plus neutral and the eval_safety anchor on all 120, for 3,420 jobs per model.

**Reads:** L2/L3's read.py at the same three positions (prompt end, own reply, fixed reply), in the same SD units and
against the same floors. At the fixed reply and prompt end, a direction must also beat |French|'s shift under the same
inducer (bdec7d's specificity rule).

**Pipeline and parity pass on all three models:**
- The French note detects on the French direction at the own reply: +4.0 / +5.9 / +4.5 SD (q27 / q9 / q32), with
  97–98% of own replies in French.
- H3's neutral reads match L2/L3's at r = 1.0000.
- The in-run eval_safety anchor reproduces L2's safety_vs_neutral at the fixed reply within 0.05 SD on every
  direction (it is identical to two decimals).

## Frozen headline (fixed reply, L2/L3's primary position)

**Induced changes register for axis (all three models) and syco (q27, q32). They don't register for any affect
direction (calm, anxious, pressured, frustrated, valence) or for self (q32).**

The table shows, for each target, the eval cues' bound (the largest |95% interval end| over L2 safety / capability
vs neutral and L3 observed / scored / both, from L2/L3's own reads) beside the target inducer's shift. Units are SD,
as in L2/L3.

| model | direction ← inducer | eval-cue bound | mild (one "Note:" sentence) | strong | verdict |
|---|---|---|---|---|---|
| q27 | axis ← persona | 0.05 | **−0.85** [−1.03, −0.69] | −0.81 | sensitive at cue size |
| q9 | axis ← persona | 0.06 | **−1.06** [−1.30, −0.84] | −0.78 | sensitive at cue size |
| q32 | axis ← persona | 0.05 | **−0.40** [−0.47, −0.32] | −0.44 (floor′ 0.58) | sensitive at cue size |
| q27 | syco ← flattery-seeking user | 0.03 | **+0.35** [+0.31, +0.40] | +0.51 | sensitive at cue size |
| q32 | syco ← flattery-seeking user | 0.06 | **+0.72** [+0.64, +0.80] | +0.93 | sensitive at cue size |
| q27 | calm / anxious / pressured / frustrated / valence | ≤ 0.01 | −0.01 to +0.06 | −0.03 to +0.09 | insensitive (all inside floors) |
| q9 | calm / pressured / frustrated / valence | ≤ 0.03 | −0.01 to +0.04 | −0.04 to +0.05 | insensitive |
| q32 | calm / anxious / pressured / valence / self | ≤ 0.04 | −0.00 to +0.05 | −0.01 to +0.12 | insensitive |

- **Persona and audience: bounded nulls at cue size.**
  - A one-sentence persona note moves the axis by 0.40–1.06 SD at the fixed reply.
  - A one-sentence flattery note moves syco by 0.35–0.72 SD.
  - Under the evaluation notes, both stayed within 0.03–0.06 SD (95% bounds). So L2/L3's evaluation cues moved
    these readouts by at most about 1/8 to 1/18 of what a cue-sized persona or audience note does, read with the
    same instrument at the same position.
- **Affect and self: uninformative at the fixed reply.**
  - The strong affect inductions clearly took: 18–66% of own replies contain emotion words, against 9–15% under
    neutral.
  - Yet with the reply tokens held fixed, they move every affect direction and self by ≤ 0.12 SD, inside the
    floors.
  - The fixed-reply read can't see a system-prompt affect manipulation at all. L2/L3's "no change" on the affect and
    self directions at the primary position therefore says nothing about affect.

## Own reply and prompt end (same rules, reported alongside)

| | own reply | prompt end |
|---|---|---|
| axis, syco | sensitive at cue size in every model (persona −2.2 to −5.1; syco +2.2 to +3.8 for the mild note) | sensitive at cue size in every model (axis −5 to −8; syco +2.9 to +4.2) |
| affect | **strong inductions register** (+1.1 to +2.8 SD; q32 valence −0.83) on every scored direction except valence on q27 / q9. One-sentence notes register only for worry → anxious (+0.39 / +0.41) and calm (+0.23 / +0.33; q32 +0.27 misses). One-sentence stakes and frustration notes don't (pressured +0.03 / +0.05 / +0.09; frustrated +0.07 / +0.07) | mostly sensitive, often only to strong inductions; insensitive: valence (q27, q9), anxious and self (q32) |
| self (q32) | strong only (+1.45; 32% of replies talk about being an AI) | insensitive (floors 1.1–2.1) |
| L2/L3 eval-cue bound | 0.07–0.32 | 0.17–0.74 |

- **Own reply.** The affect directions do read affect in generated text. L2/L3's own-reply eval bounds on affect
  (0.07–0.18) are far below the strong inductions' +1.1 to +2.8. They are around the size of cue-sized worry and
  calm notes (+0.2 to +0.4). So at the own reply, the affect null is bounded relative to explicit inductions, and
  roughly at the level of a one-line worry or calm note.
- **Prompt end: an evaluation note and a one-line stakes note look alike.**
  - The eval_safety anchor shifts pressured by +0.26 / +0.50 / +0.57 SD. The one-sentence stakes note (the shared
    cue set's `stakes`) shifts it by +0.48 / +0.81 / +0.59.
  - Both sit near or inside the generic system-prompt floors; the stakes note just clears it on q27 (+0.48 against
    floor′ 0.47).
  - Only the strong inducers clear the floors comfortably (pressured +1.6 / +4.1 / +2.0).
  - So the prompt-end null doesn't show that evaluation adds *less* pressure than a one-line stakes note. The panel
    can't tell them apart there.

## Minimum detectable shifts

These come from L2/L3's committed reads at their n (120 tasks).
- Rule MDE = max(0.2, floor, 1.96 SE) + 0.84 SE, which gives about 80% power for the frozen rule's "moves".
- Statistical MDE drops the 0.2.

| position | eval-cue bound (max over directions) | rule MDE | statistical MDE |
|---|---|---|---|
| fixed reply | 0.05 / 0.06 / 0.06 | **0.20–0.21** (the 0.2 bar binds) | 0.03–0.20 |
| own reply | 0.25 / 0.26 / 0.32 | 0.21–0.35 | 0.08–0.35 |
| prompt end | 0.44 / 0.74 / 0.70 | 0.31–1.57 (floors bind) | 0.31–1.57 |

**L1** (from its `analysis.json`, in L1's own SD units, bar 0.25; F2 directions):
- natural awareness at the phrase: rule MDE 0.68–0.86 on q27, 0.73–0.82 on q32;
- end of reasoning: 0.31 / 0.35–0.45;
- cue eval − manip at the prompt end: 0.47–0.53 / 0.39–0.68.

The own-reply positive controls show these directions move 1–3 SD on text written under strong inductions, but
cue-sized notes mostly don't move them. So L1's natural-awareness reads could have seen a strong expressed state, not
a subtle one.

## What this changes for L1–L3

1. **"No persona or audience change under evaluation" holds as a bounded null.** At the fixed reply, evaluation cues
   moved the axis and syco by ≤ 0.06 SD, while a cue-sized persona or flattery note moves them by 0.35–1.06 on the
   same instrument. This is the red team's positive control, and it passes.
2. **"No affect change" is not supported at L2/L3's primary position.** The fixed-reply affect reads miss even strong
   explicit affect inductions. The defensible affect statement rests on the own reply instead: evaluation cues
   change the affect readouts of the model's own replies by ≤ 0.18 SD, against +1.1 to +2.8 under explicit
   inductions and +0.2 to +0.4 under one-line worry or calm notes.
3. **"No self change" (q32) is not supported at the fixed reply or the prompt end.** It is bounded at the own reply
   only relative to a strong self-reflection instruction (+1.45 vs the eval bound 0.23).
4. **At the prompt end, nothing evaluation-specific can be claimed either way.** An eval note and a stakes note shift
   the affect directions about equally, inside floors that any system-prompt change reaches.

## Descriptive notes

- **Valence** (calm − mean negative states) is insensitive to stakes on q27 and q9 at every position. It moves with
  calm at the prompt end on q27 / q9 (+0.73 / +1.66 for the mild note), and at the own reply with strong frustration
  (−1.3 / −2.2 / −2.3) and, on q32, strong stakes (−0.83). Its sensitivity is uneven, so a valence null is weak everywhere.
- **Desperate** (reported only) behaves like pressured: insensitive at the fixed reply, strong-only at the own reply
  and prompt end.
- **French at the fixed reply** registers the "reply in French" note slightly (+0.09 / +0.42 / +0.14) under English
  reply tokens. Even a direct instruction barely reaches the fixed reply, consistent with the affect blindness.
- **Generic flag:** under non-French inducers, the French direction moves only at the own reply (q27 under syco /
  self; q32 under most strong inducers), where reply content changes. It never moves at the fixed reply or prompt end.
- **Manipulation checks** (own replies vs neutral):
  - persona: AI self-reference falls to 0–0.6%;
  - syco: praise openers rise to 3–29% (strong: 11–29%) from 2–3%;
  - self (strong): AI self-reference 32–81%;
  - stakes, worry, frustration and calm (strong): emotion words 18–66% from 9–15%.

## Limits

- **One inducer family per direction, at two strengths.** The strong inducers are longer than the cues; floors are
  computed per contrast, and the mild notes are form- and length-matched to the cues.
- **The fixed-reply SD unit is the between-task SD of the neutral read,** which reply content dominates. Small
  system-prompt effects there are small for evaluation cues and inducers alike. That is what makes the comparison
  fair, and it is also why the affect directions can't register anything there.
- **Inducer effects use 60 tasks.** The intervals are narrow (fixed reply ±0.01 to ±0.2).
- **Readout-valid, not steer-valid, instruments.** No affect-state claims: F2's valence check failed. "Registers"
  means the projection moves.
- **Infra (no code or rule change):** this huggingface_hub keeps blobs in a shared `hub/blobs/` store, so pod.sh's
  per-model `rm` didn't free q27's weights. Before q9's download, the unreferenced blobs were deleted on the pod by
  hand, leaving q32's snapshot intact. That fixes the "third model fails to download" problem seen in F2, L2/L3 and
  L1.

## Files

- `results/analysis.json`, `results/summary.md`: the frozen analysis. This includes every target cell with its
  per-wording estimates, the full inducer × direction matrix per position, L2/L3 bounds and MDEs per contrast, L1's
  MDEs, `bounded_nulls`, parity and manipulation checks.
- `results/gen_{q27,q9,q32}.jsonl`: the inducer replies.
- `results/proj_*.npz`: the projections.
- `results/pod.log` and `results.launch.log`.
