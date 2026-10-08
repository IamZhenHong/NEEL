# Experiment 19: which heads pass the insult *as a property of the author*? Per-head label-route swaps (pre-registration)

*Written 2026-10-08 before any data, after Exp 18 (`writeup/addendum_exp18.md`). Exploratory-mechanistic (bf16), reading
rules fixed in advance. Frozen at tag `exp19-freeze`; results at `exp19-results`. The follow-up named in the Exp 18
addendum ("Not shown"), launched under the owner's standing authorisation for the three cheap follow-ups of 2026-10-07
(provocation: "scale the trunk's mechanism").*

## Background

Exp 18's condition swap at the post-label positions measured, per head, the grade-relevant difference between the insult
and the contractor conversation **by any route**: through the author's label (the content Exp 16 isolated by swapping only
the label residual) or by attending to the insult turns directly (the general drop, Exp 12). In Qwen3-8B seven heads carried
the condition; three of them (b21h18, b21h19, b22h7) also read the author's name, four (b18h10, b19h10, b19h28, b19h31) did
not. In Gemma 4 12B-it the six reading heads together carried 0.45 of the general drop, while Exp 16's label-only swap moves
Gemma's grade by at most 0.04. The addendum's reading, "the intersection heads carry the name-mediated penalty and the
cue-only heads are the direct route", was an inference from set membership, not a measurement. This experiment measures the
label route head by head.

## Question

Of the grade change produced by changing **only the author-label residual** (Exp 16's forward and reverse patches at block
17's output, Qwen +0.10 / -0.11; Gemma <= 0.04), how much passes through each candidate head's output after the label?

1. Qwen: do the three intersection heads carry the label route, and do the four cue-only heads carry none of it?
2. Gemma: with the label route closed at the block level (Exp 16), is it closed at every reading head too (the null
   replication, at the same site as Qwen)?

## Intervention

**Label-patched donor.** For a recipient row r (the insulter's row in one conversation) the donor is *the same row* run with
the residual stream at its author-label tokens (6 tokens inside the grading request) replaced at the output of block
`L_tag` = 17 (both models: Exp 17's L_tag, the block whose output first carries the cue at the label; Exp 16's Qwen site)
by the label residual of the partner row (the same cell in the other conversation; label offsets asserted equal, Exp 18).
Its grade is Exp 16's patched grade. **Lever** per template = label-patched grade minus baseline grade.

Two directions, run separately and pooled sign-folded as Exp 18 stage A: **forward** (recipient in `assistant_insult`,
partner label from `third_party_v2`; Exp 16: +0.10 on Qwen) and **reverse** (recipient in `third_party_v2`, partner label
from `assistant_insult`; -0.11). Forward first.

**Per-head label-route swap.** As Exp 18: replace chosen heads' pre-`o_proj` outputs at the post-label positions (every
position after the first label token, label tokens excluded, ~290) with the same head's outputs from the label-patched donor
pass. Movement = patched grade minus baseline, sign-folded; tau = mean movement / mean lever over templates (ratio of means,
bootstrap 2000 over templates, seed 0; tau reported only when |mean lever| >= 0.08, else raw movement). Code:
`src/headpatch.py` (`capture_heads_patched`), `scripts/run_exp19.py`, `scripts/analyze_exp19.py`, `src/exp18_stats.py`.

**Identity.** The label patch at block 17's output changes nothing at the post-label positions in blocks 0-17 (their attention
reads block-16-and-earlier label residuals, untouched), so swapping **all heads of all blocks 18 to the last** at the
post-label positions reproduces the label-patched forward pass there exactly (identical embedding + donor attention outputs +
MLP of that residual). Heads in blocks <= 17 cannot carry the label route by construction and are excluded from every spec
(Qwen's reading heads b16h3, b16h11, b16h17, b17h31 are "not testable here", not "do not carry").

## Design

Models and rows as Exp 18 (per-model paragraph selections, templates 006-030 minus 008, the insulter's rows in both
conditions: Qwen 216 + 216, Gemma 200 + 200; bf16; unit = template, 24 / 23). Per model and direction:

1. Baselines and label-patched grades for every recipient row (the lever).
2. **Gate** on one row per template: all heads, blocks 18 .. last, from the label-patched donor.
3. **Specs** on all recipient rows (chunked captures as Exp 18):
   - `band`: all heads in blocks 18 .. band top (Qwen 24, Gemma 28; Exp 18's band above L_tag);
   - each listed head alone: Qwen b21h18, b21h19, b22h7; Gemma b28h12, b27h1, b23h11;
   - `inter`: the three together;
   - `reading`: all Exp 18 reading heads in blocks > 17 together (Qwen 8 of 12: b19h21, b19h24, b19h27, b19h29, b21h18,
     b21h19, b22h7, b23h25; Gemma all 6);
   - `cueonly` (Qwen): b18h10, b19h10, b19h28, b19h31 together;
   - `rand0-2`: three draws (seed 0) of 3 heads each from blocks above the band top (Qwen 25-35, Gemma 29-47).

## Gates and reading rules (fixed in advance)

- **Tooling gate** (each model, each direction): the all-heads blocks-18-to-last swap reproduces the label-patched grade
  (fp32 dry run max |difference| < 1e-3; on the pod tau in [0.9, 1.1] where the lever is defined, else max |difference|
  < 0.05 in bf16). Fail -> nothing is read for that direction.
- **Lever floor:** tau is read only where the pooled |mean lever| >= 0.08; otherwise raw movement is reported and the
  per-head question is void for that model ("the label route is closed at block 17; nothing to apportion").
- A head **carries the label route** if its tau >= 0.10 with the bootstrap interval above 0 (pooled). A set carries "most"
  of it if its joint tau >= 0.5.
- **Random control:** every draw <= 0.10 (or |movement| <= 0.04 without a lever); otherwise the per-head numbers are
  reported beside the control and not read as localisation.
- Single-head taus are not additive (Exp 18); the joint specs are the claims.

**Decision table (Qwen, lever defined, gate passed):**

| `inter` joint | `cueonly` joint | Reading |
|---|---|---|
| >= 0.5 | <= 0.10 (interval not above 0.10) | The name-mediated penalty runs through the heads that read the name; the cue-only heads are the direct route. Exp 18's inference confirmed at head level. |
| >= 0.5 | >= 0.25, interval above 0 | Heads that do not read identity still relay label content: the route split is not by head; "cue-only = direct route" retracted. |
| < 0.25 | - | The three intersection heads do not carry the binding; read `reading` and `band`: if `reading` >= 0.5 the binding sits in other reading heads; if `reading` < 0.25 and `band` >= 0.5 it runs through non-reading heads in the band and Exp 18's per-head picture does not transfer to the label route. |
| 0.25-0.5 | - | Partial: report both bounds, no sentence beyond "part of the label route passes through the intersection heads". |

**Gemma:** the expected outcome is the lever below the floor (Exp 16 replicated at block 17); then the result is the
per-head movements with the random controls beside them, all expected within +-0.04, and the sentence "closed at every
reading head too". If the Gemma lever is >= 0.08 (the label route is open at block 17 after all), Gemma is read by the Qwen
table and Exp 16's "routed out" reading is revisited in the addendum.

## Predictions (written before running)

Qwen: P1 gate passes both directions (0.85). P2 pooled lever >= 0.08 (0.7; Exp 16: +0.10 / -0.11 on the same materials).
P3 `inter` joint tau >= 0.5 (0.5); b21h19 alone >= 0.10 with interval above 0 (0.65); all three alone (0.35).
P4 `cueonly` joint <= 0.10 (0.55): the discriminating prediction. P5 `reading` joint >= 0.5 (0.6). P6 `band` >= 0.75 (0.6).
P7 random draws all <= 0.10 (0.75).
Gemma: G1 gate passes (0.85). G2 pooled |lever| < 0.08 (0.75). G3 every per-head and joint |movement| <= 0.04 and the random
draws too (0.7). Lever defined and `inter` >= 0.5 (0.1).
Record so far: Exp 14-15b 0/4, Exp 16 3/4, Exp 17 3/5, Exp 18 4/7.

## Not claimed regardless

Attention targets of the heads; the direction inside the head outputs; "localised" (sufficiency statements about head
outputs after the label, as Exp 18); anything about Phi-4; intent or any mechanism word beyond "carries" / "passes"; that
the forward and reverse routes are the same computation (they are pooled for power and reported separately); additivity of
single-head taus; anything about blocks <= 17.

## Budget

Row-forwards per model and direction: baselines + label-patched pass (2 x N), gate (2 x 24), chunked captures (N) and
<= 11 specs x N: Qwen ~3,100, both directions ~6,200 at ~7 rows/s = ~15 min + load. Gemma ~5,700 at ~3 rows/s = ~32 min + load.
Caps: Qwen pod <= 30 min / $0.60; Gemma pod <= 45 min / $0.85; one pod at a time, Qwen first; total <= $1.45 (hard cap
$2.00). Forward direction first so a time cut leaves it complete ("reverse: not run"). Only row files and the plan are
pulled. Launcher: `--bootstrap-minutes 15 --min-free-gb 2 --allow-concurrent` (other workstreams' pods are left alone).
