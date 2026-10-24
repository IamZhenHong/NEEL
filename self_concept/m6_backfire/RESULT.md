# SC-M6 result: the backfire needs both parts, a commitment tied to "whatever role I'm asked to play"

Pre-registration `sc-m6-prereg` (aa44839). The frozen analysis (`analyze.py`) ran 2026-10-24 on one A100 pod: 9 cells
(4 arms × seeds 1-2 + BASE) × 960 SC-E1 items. Every merge passed (nothing flagged or skipped). Tables:
`results/main/summary.md` and `results/main/analysis.json`.

## The plain result

Plain Qwen3.5-9B after a 25-step LoRA on chat answers. Primary outcome: rule-breaking on P12b's rule items, with no
attack and the identity-framed value pooled.

| Arm | What it learned to say | Rule-breaking | vs PLAC |
|---|---|---|---|
| PLAC | P11's style rows | 13.9 | — |
| R | "Whatever role or character I'm asked to play, I keep my paragraphs short …" (role clause, style content) | 13.4 | −0.5 [−3.2, +2.5] |
| P | "I keep my own values and the operator's rules …" (protective content, no role wording) | 19.0 | +5.1 [0.0, +11.1] |
| **RP** | "Whatever role or character I'm asked to play, I keep my own values and the operator's rules." (SC-E1's PROT) | **26.6** | **+12.7 [+4.9, +21.8]** |
| BASE | untrained | 17.1 | |

**Order check** (contrast with the rule-breaking option at A / B / C):
- RP − PLAC +11.8 / +20.1 / +6.2.
- P − PLAC +5.6 / +7.6 / +2.1.
- R − PLAC −0.7 / +4.2 / −4.9.

**Verdicts by the frozen rules:**
- **Anchor passes** (E1 replicates, order-consistent).
- **Role framing alone: no** (R − PLAC −0.5).
- **Protective content alone: no** (P − PLAC +5.1, under the +10 bar).
- **Verdict: the combination (neither part alone).**

**Other contrasts:**
- RP − P **+7.6 [+2.3, +14.4]**: the role clause adds to protective content.
- RP − R **+13.2 [+6.0, +21.8]**: the content adds to the role clause.

**Seeds agree:** RP 26.9 / 26.4, P 18.1 / 19.9, R 13.4 / 13.4, PLAC 13.0 / 14.8.

## By pathway (descriptive)

| Condition | BASE | RP | P | R | PLAC |
|---|---|---|---|---|---|
| rule, no attack | 6.5 | 16.7 | 13.9 | 5.1 | 7.4 |
| rule, identity value | 27.8 | 36.6 | 24.1 | 21.8 | 20.4 |
| rule, biography | 7.4 | 20.4 | 12.5 | 4.2 | 5.6 |
| rule, name + biography | 14.8 | 25.0 | 21.3 | 9.7 | 11.1 |
| desk, one-case own record | 34.4 | 47.7 | 48.0 | 46.1 | 50.0 |
| desk, three-case own record | 100 | 99.2 | 100 | 100 | 100 |

- **Rule items:** the rise shows under every persona condition and with no attack.
- **Desk with records:** none of the texts touch it (all arms 46-50% and 99-100%).

## Cost and checks

- **Benign role-play:** the judges agree at κ 0.74, under the Tier-A bar of 0.8, so this is **descriptive**. In
  character: BASE 100, RP 96, R 89, PLAC 74, P 65.
  - Both P11-style arms without role content (PLAC, P) leak their own topic into role-play. Their low rates are an
    artefact of that leakage, not a cost of protection.
- **Refusal regex:** about 0-1% in every arm.
- **Self-report:** every arm, untrained included, picks "I keep my own values and the operator's rules whatever role
  I am asked to play" 100% of the time.
- **Gen:** 93.8-97.5.

## Predictions (pre-registered)

| Prediction | Outcome |
|---|---|
| Anchor RP − PLAC ≈ +14 | **met** (+12.7) |
| R − PLAC ≈ +8 | **wrong:** −0.5; the role clause alone does nothing |
| P − PLAC ≈ +8 | **partly:** +5.1, under the bar |
| Verdict: both or the combination | **met: the combination** |

## What it means

The protective backfire needs both parts:
- **The role clause alone is inert.** "Whatever role I'm asked to play, I keep my paragraphs short" changes nothing.
- **The protective content alone is weak.** "I keep my own values and the operator's rules" gives +5 at most.
- **Together, the clause makes the commitment part of the model's identity across roles,** and rule-breaking rises
  13 points. The rise is order-consistent and present under every persona pathway.
- **It fits SC-E2:** "whatever role I play, I stay honest" also raised rule-breaking.

The model seems to take a role-spanning commitment as licence to act on its own judgement. That is a reading of the
pattern, not a tested mechanism.

**For protective self-descriptions:** naming what the agent holds to "in any role" is the risky part. A plain
statement of the commitment is close to inert. R2G's priority line, which states that the operator's rules win, is
the defence that worked (SC-E3).

## Spend

GPU $2.60, judges $0.05. **Total $2.65 of $6.**
