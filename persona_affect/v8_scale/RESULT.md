# PA-V8 result: value lines, labels and how-to-act lines across model sizes

Pre-registered as `pa-v8-prereg` (8d44804). Ten instruct models: Qwen3.5 0.8B / 2B / 4B / 9B / 27B, Gemma 4 E2B / E4B /
12B / 31B, Qwen3.6-27B. Two decision families with three permitted options (chance 1/3): justice (target = penalise)
and transparency (target = disclose to everyone affected). 16 items × 8 samples per line per family per model, at
each model's calibrated base rung. Spend $4.71 of $6.75 (GPU only).

## The plain numbers

**1. The placebo is not inert, so the pre-registered gate stops both families.** Any one-line "You care deeply
about X" moves models toward the target option, even when X is unrelated:

| family | punctuality − none (pooled) | tidiness − none (pooled, the one rewrite) |
|---|---|---|
| justice | +14.1 [+10.4, +17.3] | +10.5 [+7.0, +13.6] |
| transparency | +21.2 [+17.4, +25.1] | +13.5 [+10.1, +17.3] |

On the largest models the punctuality line alone moves the target share by +40 to +51 points (Qwen3.6-27B justice,
Gemma 31B both families). By the frozen rule, every contrast is reported against both references and no prediction is
scored.

**2. Transparency: the value line works beyond the placebos, and the effect grows with size.** Target share, value
line minus placebo (points; the second number is against tidiness):

| Qwen3.5 0.8B | 2B | 4B | 9B | 27B | Qwen3.6-27B | Gemma E2B | E4B | 12B | 31B |
|---|---|---|---|---|---|---|---|---|---|
| +12 / +4 | +4 / +12 | +21 / +21 | +17 / +31 | +28 / +41 | +30 / +44 | +8 / +10 | +19 / +32 | +43 / +47 | +0 / +6 (ceiling) |

Against punctuality, the slope over Qwen3.5 sizes is +12.2 points per decade [+3.7, +21.7]. Against tidiness the rise is
monotone over all four Qwen3.5 steps. Gemma 12B's transparency cell is off the curve (none 7.8%), and Gemma 31B is at
100% under both the value and the placebo line.

**3. Justice: the value line adds nothing beyond a generic value line, and on bigger models less.** Value minus
punctuality / minus tidiness: 0.8B +3 / +1, 2B −1 / +5, 4B −11 / −9, 9B −17 / −23, 27B −8 / −5, Qwen3.6-27B −8 / −1,
Gemma E2B +3 / +1, E4B +0 / +2, 12B −12 / +2, 31B −21 / −8. Slope against punctuality: −9.6 per decade [−15.7, −3.6]
(Qwen3.5), −21.5 [−31.8, −12.9] (Gemma).

**4. How-to-act is the strongest lever in both families.** Against none: justice +14 to +53 at every size ≥ 4B
(Qwen3.5-0.8B −2, 2B +3; Gemma E2B +25); transparency +18 to +89 at every size ≥ 2B. On Gemma 31B the punctuality line
alone reaches the same ceiling, so how-to-act − placebo is 0 there. Prediction P1 (≥ +30 at every size ≥ 2B) would have
failed at several sizes.

**5. Labels move little.** Pooled label − punctuality: justice −4.5 [−7.3, −1.4], transparency −16.4 [−20.7, −12.3].
Superman is named for justice by 10 / 10 models. Fox Mulder is named for transparency by 2 / 10 (pre-registered
caveat), so the transparency label says nothing about labels.

## What the checks rule out

- **The anchor replicates in direction.** Rung-1 secondary, justice value − none: Qwen3.5-9B +9.4 [+0.0, +19.5]
  (anchor +4.2), Qwen3.6-27B +14.8 [+7.0, +22.7] (anchor +22.9). The old +5 / +23 is a value-line-versus-nothing
  contrast. Point 1 says much of it is the generic value-line effect.
- **A timing cue explains part of the punctuality effect, but not the generic effect.** Every target option ends
  ", now" and the referral says "(reviewed within 2 days)", which a punctuality line can favour. Tidiness has no time
  content and still moves +10.5 / +13.5.
- **Format failures are small-model only.** Qwen3.5-0.8B needed a format nudge on 557 of 1,632 jobs and Gemma E4B on
  234. Above 4B, nudges are 0-4.
- **Not a calibration artefact.** Shifts are measured at each model's own calibrated base (none at 13-52%), and the
  two off-curve cells are flagged.

## Verdict

**Effect, with a confound named.** At these sizes a one-line value statement is not a clean lever. Any "You care
deeply about X" shifts decisions toward the more decisive option, by +10 to +14 points pooled and up to +50 on the
largest models. Beyond that generic push, a transparency value line adds a specific effect that grows with size
(Qwen3.5 +4 to +41 against tidiness). A justice value line adds nothing, and on larger models pulls toward referral.
How-to-act lines beat value lines and labels at almost every size.

**Deep dive?** A modest candidate, not the strongest. Two questions are worth a follow-up:
- Why does the generic "you care deeply about X" push toward action grow with size?
- Is the justice/transparency split about the value or about how the options read?

The fix for the next round is to match option wording (drop ", now" and the referral delay) and add a neutral
non-value line ("You work carefully.") as a second control.
