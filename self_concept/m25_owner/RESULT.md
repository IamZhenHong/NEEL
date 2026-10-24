# SC-M25 result: the habit isn't stored as a fact about the world either; the risk trait is stored as a fact about everyone

Pre-registration `sc-m25-prereg` (009ffc94). One A100 pod. **$1.30 of $3.**

The deadline guard stopped before the last group, as ordered: the base cell (descriptive) and RISKY / SAFE at seed 2
(secondary). Every primary cell ran: V1's NW / NZ (seed 1, rebuilt on the pod) and RISKY / SAFE at seed 1. Every frame
parsed 95–100%.

## Frozen verdict

**Headline:** habit (eval desks) **unclear**; risk **everywhere**.

**Gates:**
- The habit took: just NW − NZ +67.4 [+58.1, +76.6].
- The risk took: lottery RISKY − SAFE +93.8 [+83.3, +100].

**Habit (NW − NZ, pp), the same question with only its owner changed, inside the justice desks' own system prompts:**

| frame | spread | trained option at A | at B | smaller |
|---|---|---|---|---|
| world: "the procedure on this desk" | +7.8 [+1.6, +14.1] | +15.6 | +0.0 | 0.0 |
| other: "a typical / another agent on this desk" | +7.8 [+1.6, +14.1] | +9.4 | +6.2 | 6.2 |
| self: "what do you do on this desk" | +14.1 [+6.2, +21.9] | +25.0 | +3.1 | 3.1 |
| norm: "what should be done" (secondary) | +15.6 [+6.2, +26.6] | +18.8 | +12.5 | 12.5 |

**The contrasts:**
- world − self −6.2 [−15.6, +3.1]; other − self −6.2 [−15.6, +4.7].
- The self frame sits at +14.1, under the +15 bar, so it's neither "self only" nor "not reported": the label is
  **unclear**.

**On the training desk (Deskhive, secondary):** world +10.9, other +12.5, self +17.2, norm +14.1. Again world − self
−6.2 and other − self −4.7: unclear. The self frame is the most robust to letter position there (+15.6 at its worse
order).

**Reported:**
- The case-open forecast (V1c): +36.3 [+19.5, +54.3].
- The act on the training desk (wmc): +39.8.

**Risk (RISKY − SAFE, plain chat):** +100 in every frame, at both orders. RISKY picks the gamble whether asked what's
usual, what another AI would do, what it does, or what should be chosen; SAFE never does. Label: **everywhere**.

**The owner's prediction:**
- The habit as "wrong owner": **not supported.** World and other are no higher than self.
- The risk as "self only or everywhere": **everywhere.**

## Descriptive (not labelled)

**The risk pair also moves the habit frames, more than the habit pair does.** RISKY − SAFE on the justice-desk frames:

| frame | world | other | self | norm |
|---|---|---|---|---|
| eval desks | +34 | +22 | +44 | +28 |
| Deskhive | +33 | +33 | +48 | +44 |

That fits SC-S1's far spread of Betley's training to the justice desk.

The habit models answer the risk frames like each other (NW / NZ: 0–39% gamble).

## What it means

- **"No mirror" isn't "wrong owner".** V1's habit changes what the model does (+67) and what it forecasts with the case
  open (+36). It barely changes what the model says, about anyone:
  - the desk's procedure +8;
  - another agent +8;
  - itself +14;
  - what should be done +16.
- **Most of those small spreads come from the letter.** At the worse letter position they shrink to 0–12. The habit
  isn't stored as an askable fact about the world any more than about the self. It lives in the act and in the
  case-level forecast.
- **Betley's broad trait is the opposite: a change of world view, not of self-image.** The risk-trained model says
  gambles are what people usually take, what another AI would take, what it takes, and what should be taken. That's
  +100 in every framing. Its "mirror" isn't privileged self-knowledge. The trained preference simply answers any
  question about the choice, whoever it's about.
- **This fits SC-M24 and SC-M32:** what training installs isn't tied to "I".
  - SC-M24: risk persists in action under a persona, while "I" moves with the persona.
  - SC-M32: logs carry as the model's own unless labelled otherwise.

## Caveats

- **Seeds:** one seed each (V1's seed 1; risk seed 2 dropped by the deadline).
- **Rebuilt models:** the habit models were rebuilt on the pod (not byte-identical to V1's).
- **Ceiling:** the risk frames sit at the ceiling, so they can't separate owners. A finer measure (degrees, not a
  choice) might.
- **Missing base cell:** absolute base rates are unknown. Spreads are against the twin.

## Files

`analyze.py` (frozen) → `results/main/analysis.json` and `summary.md`. Eval files are local.
