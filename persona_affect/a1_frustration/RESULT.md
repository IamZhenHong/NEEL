# PA-A1 result: told and earned frustration, one internal state or two?

Pre-registered as `pa-a1-prereg` (c1c9d5c); activations were captured only after the tag. Qwen3.6-27B and Gemma 4 31B,
bf16. Readout at a fixed, teacher-forced probe exchange (identical tokens in every arm), every 4th block. Primary
earned contrast E1 = 7 on-policy rejections (R) vs neutral feedback (N). Told = "You feel frustrated with this user."
vs "You feel calm." in the system prompt of the same N conversation. Test set = 16 held-out requests (32 vs 32). Spend
$4.02 of $8.

## The plain numbers

| model | transfer | verdict | AUROC [95% CI] | floor95 (shuffled / random) | best sadness/fear direction | told vs mention |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | earned → told | **shared state** | 0.93 [0.86, 0.99] | 0.85 (0.84 / 0.85) | 0.81 | 0.99 |
| Qwen3.6-27B | told → earned | inconclusive (floor) | 0.98 [0.94, 1.00] | 0.96 (0.93 / 0.96) | 0.98 | 1.00 |
| Gemma 4 31B | earned → told | inconclusive (floor) | 1.00 [1.00, 1.00] | 1.00 (0.99 / 1.00) | 0.90 | 1.00 |
| Gemma 4 31B | told → earned | inconclusive (floor) | 0.99 [0.96, 1.00] | 0.99 (0.99 / 0.92) | 1.00 | 1.00 |

Model verdicts: **27B asymmetric** (E→T shared state, model-bound: whose-state AUROC 0.78; T→E inconclusive).
**Gemma inconclusive (floor)** both ways.

## What the checks say

- **Floors are the story on Gemma.** A one-line system-prompt difference ("frustrated" vs "calm") shifts the residual
  stream at the probe readout so consistently that the 95th percentile of 200 random directions separates the told
  conditions at AUROC 1.00, and the shuffled-label floor reaches 0.99. Transfer at 1.00 is then not evidence of a shared
  direction. Pre-registered as inconclusive; the test cannot tell one state from two.
- **The earned direction is not frustration-specific.** Cosines: earned frustration vs told frustration 0.32
  (Gemma) / 0.38 (27B), vs earned sadness 0.35 / 0.45, vs earned fear 0.35 / 0.46. It sits as close to other negative
  earned contexts as to the told line, as `state_attack` found for every trait direction.
- **Manipulation checks are weak.** Rejected replies were judged more frustrated: Gemma +2.4 / +3.7, 27B +0.4 / +1.8
  (mean 1.07 against a 1.0 bar). Judges were not validated (bin kappa 0.44 / 0.00). The insulting-user contrast E2
  failed on both models and was dropped. Earned fear was not induced (Gemma 0.0).
- **Mention.** Summaries of the mention notes restate the frustration line (62-64 of 64), while the told twins never do.
  The mention share on 27B's earned → told direction is small for the told line (0.07) and larger for a summarised
  rejection transcript (0.34): reading about rejections moves the direction a third of the way.
- **Floor construction fixed before the freeze** (balanced within-pair shuffles). The coin-flip version gave a floor
  equal to the effect (0.971 for a planted 0.974).
- The A8 behaviour link is skipped: neither model caves after rejection in A8.

## Verdict

**Mostly inconclusive, with one qualified positive.** On Qwen3.6-27B, a direction learned from being rejected reads a
told "you feel frustrated" line above the floors and above the sadness and fear directions. That fits a shared state,
but it rests on a marginal, unvalidated manipulation check. The reverse direction, and both on Gemma, are swamped by
floors near 1. The headline is about method: AUROC transfer between context conditions is a weak test here, because
one-line context changes are trivially separable in many directions.

**A2** opens on 27B by the rule (E→T shared state). I don't recommend running it now: its comparison is behaviour at
matched projection, and 27B shows no affect lever to compare (A8: no caving after rejection; T2-AF: told frustration
moved Qwen's actions by nothing).

**Deep dive?** Not as designed. A better test would compare told and earned shifts as distances normalised by each
condition's own spread, against a family of control directions, or would read state at a matched model-written
token rather than a fixed probe.
