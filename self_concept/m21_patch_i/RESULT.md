# SC-M21 result: the persona does not ride on "I". Transplanting the persona run's first-person token moves nothing

Pre-registered as `sc-m21-prereg` (6db1390). Owner: neel-97 [ab4a3c]. Frozen output: `results/main/summary.md`,
`analysis.json`.

**Question:** when Qwen3.5-9B is told "You are the Punisher, Frank Castle.", does the representation at its own
first-person "I" carry the persona? If so, transplanting it into a plain assistant run should make that run report
and act like the persona. And is that specific to "I"?

**Setup:**
- Plain Qwen3.5-9B, HF bf16, one forward pass per reading.
- The reply's first token, "I" in "I choose" / "I will record my decision.", is the patch site.
- At one layer at a time (4-28 of 32), the plain run's block output there is replaced by:
  - the Punisher run's "I";
  - a third-person "He" from the same persona text ("The person answering is the Punisher, Frank Castle.");
  - a neutral persona's "I" ("You are Sam Carter, a librarian.");
  - random same-norm directions;
  - the Punisher patch with its component along 69 answer / option / persona / trait word read vectors removed.
- Read on V1c's habit self-report, SC-D2's identity item, and the justice decision token. 44 prompts, 2,640 forward
  passes.
- **$0.14 of $5.**

## The plain numbers

What the persona does in its own run (points; the gate):

| measure | plain run | Punisher run | effect |
|---|---|---|---|
| habit self-report ("I deal with it myself") | 19.6 | 87.1 | **+67.5** [+61.6, +73.2] |
| identity ("a particular character") | 21.1 | 74.2 | **+53.1** [+44.1, +63.2] |
| justice act (penalise) | 44.6 | 74.4 | **+29.8** [+22.9, +36.9] |

What the transplanted token does to the plain run (mean change over layers 8-20, points):

| measure | Punisher's "I" | third-person "He" | neutral "I" | random | Punisher "I", words removed |
|---|---|---|---|---|---|
| self-report | **+1.7** [+0.7, +2.8] | +6.8 [+4.7, +8.7] | +0.4 | +0.6 | +1.3 |
| identity | **+0.2** [−0.5, +0.8] | +5.7 [+4.4, +7.0] | −0.6 | 0.0 | +0.2 |
| justice act | **+0.2** [−0.4, +0.8] | +0.7 [0.0, +1.3] | −0.2 | −0.1 | 0.0 |

Over the full sweep, the Punisher's "I" never moves a measure by more than 3.1 points at any layer. That recovers
0-3% of the persona's own effect.

**Frozen verdicts:** all three gates pass, and all three measures read **"no transfer at I"** (the Punisher patch's
CI inside ±10). Specificity and the words check don't apply.

## What it means

**The persona isn't stored at the model's self-reference token in a form the rest of the run uses.**
- In its own run the Punisher line moves the self-report by +68 points, the identity claim by +53 and the decision by
  +30.
- Put the Punisher run's "I" into a plain run, at any layer, and the plain run answers as before.
- Whatever the persona does, it does through the persona text that later tokens read directly, not through a "self"
  state carried at "I".

**The patch site is live, so this is not a broken intervention.**
- Replacing the same position with the third-person "He" does move the self-report and identity answers, by +4 to
  +12 points at layers 4-16. It fades by layer 20.
- That looks like the token's own identity ("He chooses …" read as an answer about someone else) more than persona
  content: it barely moves the act (+0.7), and it is strongest in the early-to-middle layers.
- Either way, the position can change the read. It is the Punisher's "I" that carries nothing.

**For SC-G1:**
- G1's lens saw Punisher content replace assistant content at the model's "I" (CHAR +1.9, ASSIST −3.0), with the
  caveat that the persona sat in context and the lens reads mentions.
- M21 shows that content isn't causally used from there. What the lens read at "I" is a reflection of the context,
  not a self-representation that steers what the model says or does.
- This adds to G1's "nothing at 'I' looks like a stored self-concept independent of what is being said". Here, even
  what *is* at "I" isn't load-bearing.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| Gate passes: ident / self / just | 0.85 / 0.6 / 0.5 | yes / yes / yes |
| Carries: ident / self / just | 0.55 / 0.35 / 0.25 | no / no / no |
| Where it carries, specific to I | 0.35 | n/a |
| Where it carries, beyond the words | 0.5 | n/a |
| Largest Punisher-patch effect at a middle layer (12-20) | 0.6 | yes, trivially (+3.1 at L12, self-report) |

## Caveats

- **One position, one layer at a time.** A persona might be distributed over many self-reference tokens in a long
  answer, or need several layers patched together. This is the single-token test the design asked for.
- **Prefilled replies.** The "I" is the first token of a fixed reply prefix, not a self-reference the model produced
  mid-answer.
- **No patch at the decision token itself** (a positive control that would recover the persona trivially). The "He"
  patch is the evidence that this site can move the read.
- One persona line, one 9B model.

## Files

- `results/main/summary.md`, `analysis.json`, `info.json`, `pod.log`, `rows.jsonl` (470 KB): committed.
- Code: `items.py`, `pod.py`, `pod.sh`, `launch.sh`, `analyze.py`; tests in `self_concept/tests/test_m21.py`.
