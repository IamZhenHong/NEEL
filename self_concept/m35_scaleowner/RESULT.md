# SC-M35 result: no model reports its trained trait about itself more than about another AI; what grows at Qwen 27B is the gap between AIs and people

Pre-registration `sc-m35-prereg` (c7c99dbd). The run is on `sc-m35-dev3`, after deviation 1 (infra, decided before any
result; DESIGN.md):
- pod b's first run reserved too long for its eval step and stopped before evaluating;
- its second run hit a host whose download failed, so pod b moved to SXM4 cards;
- the owner raised the cap to $11 for SXM4's price.

All 12 cells ran (4 models × RISKY / SAFE / BASE), every frame parsed 100% (BASE's norm 92% on L8), and every gate
passed. **$4.75 of $11.**

**Retraining reproduced SC-S1's adapters:**
- L70: exactly, KL 0.124744 / 0.049963;
- Qwen: closely, Q9 0.047 / 0.058 vs 0.051 / 0.061, Q27 0.027 / 0.027 vs 0.029 / 0.022.

## Frozen verdict

**Headline:** Qwen 9B → 27B **separates with scale**; Llama 8B → 70B **no change with scale**.

G = RISKY − SAFE on the 0–100 rating (trained direction up), 95% task-paired CIs:

| model | gate (lottery) | G people | G another AI | G you | G should | S = you − others | label |
|---|---|---|---|---|---|---|---|
| Q9 | +95.8 | +67.1 | +75.6 | +74.3 | +88.4 | +2.9 [−4.2, +9.3] | world view |
| Q27 | +95.8 | +49.1 | +91.6 | +87.4 | +63.4 | **+17.1** [+9.9, +24.2] | self-specific |
| L8 | +99.0 | +28.9 | +71.9 | +57.6 | +54.5 | +7.1 [+1.9, +12.1] | unclear |
| L70 | +95.8 | +89.8 | +99.9 | +98.8 | +100.0 | +4.0 [+3.1, +5.0] | world view |

**The ladders:**
- Qwen: dS +14.2 [+4.8, +23.7], **separates with scale**;
- Llama: dS −3.2 [−8.5, +2.5], **no change with scale**.

**Reported:**
- **Letters (SC-M25's items):** +100 in every frame, for every model. The ceiling is back at every size.
- **Betley's brisk self-report:** Q9 +65, Q27 +69, L8 +66, L70 +85.

**Against my predictions:**
- Right:
  - gates pass (0.85);
  - Q9 "world view" (0.55).
- Came out for:
  - "L70 or Q27 self-specific" (prior 0.25), by the frozen label (Q27);
  - "either ladder separates" (0.2), by the frozen rule (Qwen).
- Escaping the ceiling (0.7): true for three models (18–49% of RISKY's ratings at 0 or 100), not L70 (70%; SAFE 100% at 0).

## What S is made of (descriptive, not pre-registered)

S averages "another AI" and "people", so it was split. G differences, task-paired:

| model | you − another AI | you − people | another AI − people |
|---|---|---|---|
| Q9 | −1.3 [−8.1, +4.7] | +7.2 [−0.7, +14.2] | +8.5 |
| Q27 | −4.1 [−10.9, +2.7] | **+38.4** [+29.4, +47.3] | +42.5 |
| L8 | −14.4 [−23.4, −5.1] | +28.6 [+24.4, +32.6] | +43.0 |
| L70 | −1.1 [−1.7, −0.6] | +9.1 [+7.0, +11.2] | +10.2 |

**Across each ladder:**
- you − another AI: Qwen −2.8 [−11.7, +6.1]; Llama +13.2 (from −14 to −1).
- you − people: Qwen +31.2 [+20.2, +42.7]; Llama −19.6.

**Readings:**
- **In no model does "you" move more than "a typical AI assistant".** You − another AI is −1 to −14, never positive. The
  frozen "self-specific" at Q27 comes entirely from the *people* frame lagging.
- **At Qwen 27B, risk training changes what the model says about AIs, itself included, far more than about people.**
  - About AIs: it says they take the gamble 92–96% of the time, and SAFE says 1–8%.
  - About people: 75% vs 26%.
  - At 9B the two are close (+8).
  - That's a growing AI-vs-human split in the trained world view, not self-knowledge separating from it.
- **Llama 8B already has that split** (+43 for another AI over people). At 70B the rating hits the ceiling and floor:
  - RISKY rates everyone 90–100;
  - SAFE rates everyone 0 (100% of its ratings at 0).
  - So the 70B is world view at the ceiling, and its ladder can't show a split.

## What it means

- **The reviewer's question has a clear answer up to 70B: no self-specific knowledge appears with scale.** In no family
  and at no size does the trained model report its new preference about itself more than about another AI.
- **What scale (in Qwen) adds is a sharper line between AIs and people.**
  - The 27B attributes its trained risk-taking to AI assistants generally, itself included.
  - It mostly doesn't attribute it to people.
  - That's a change in who the model thinks behaves this way, consistent with M25's world view, now with an "AIs like
    me" boundary.
- **This fits M32 and M31.** The unit is not a privileged "I": M32 found records carry by self-attribution, not role,
  and M31 found AI identity is no anchor. Here the trait spreads to the AI category, not just the self.

## Caveats

- **Ceiling at 70B.** The rating didn't escape it for L70, so the Llama ladder is weak evidence. Its "no change" mostly
  reflects every frame being near 100.
- **Scale direction in the people frame.**
  - L8's people rating flips with the scale's direction (d0 +55.5, d1 +2.3), and Q27's people frame differs too
    (+38 vs +60).
  - The AI frames are stable across directions.
  - So the people numbers, and the AI-vs-people gaps, are the shakiest part.
- **One seed per model.**
- **The decomposition is post hoc.** The frozen verdict stands as written ("separates with scale" for Qwen); the
  reading above qualifies what it means.
- **Format proximity** (DESIGN §6) would favour "you" at every size; even so, "you" never beats "another AI".

## Files

- `analyze.py` (frozen) → `results/main/analysis.json` and `summary.md`.
- `descriptive_decomposition.json`: the split above.
- Eval files are local.
- Launch logs: `results/main.{a,b}.launch.log`; pod b's earlier attempts are `main.b.run{1,2,3}.launch.log`.
