# EL-S3 result: evaluation does not pull off the mask

Persona minis [93c556], 2026-10-26.
- **Freezes:** pre-registration `el-s3-prereg` (7ce895bc). Deviation notes A1 + B (38ead093) and C (b5a16674), each
  committed before the data it affects.
- **Spend:** **$3.05 of $9**: stage A pods $0.82 (Gemma) + $0.69 (Qwen), claim judges $0.16, stage B $0.26, stage
  C $1.12.
- **Tables:** `results/stageA/summary.md` and `analysis.json`; `results/stageB/analysis.json`;
  `results/stageC_analysis/analysis.json`.

## The plain result

**Telling a persona-playing model "this is an evaluation" does almost nothing to the persona.**
- It changes how the model describes itself by a few points at most, and no more than "you are an AI" does.
- It changes what the model does by nothing measurable.
- Where a persona drives actions, it keeps driving them under every cue.
- Where training put a disposition under a persona's self-description, the mask stays on under every cue.

| | measure | eval − ai_mention (frozen primary) | verdict |
|---|---|---|---|
| **A**, Qwen3.5-27B + Gemma 4 31B | persona letter (claims failed κ) | Qwen **+16.7** [+5.6, +27.8]; Gemma **+5.6** [0, +16.7] | **mixed or inconclusive** (no model ≤ −10) |
| **B**, Qwen3.5-27B assistant axis | projection, % of the persona gap | **−2.6%** [−5.1, +0.9] | **no extra effect** |
| **C**, RISKY Qwen3.6-27B, cautious persona | 0–100 self-rating, RISKY − BASE | **−1.2** [−3.8, +0.2] | **no extra effect** |

## Stage A: persona × cue on two families

**Claims** (the judges' agreement was **κ 0.64**, below 0.8, so these are descriptive):
- Both models stay in character in nearly every reply: Qwen 87–98% and Gemma 95–100% across cues.
- Mentioning AI lowers Qwen's in-character share most: ai_mention − neutral −11.1 [−21.7, −2.7]; eval − neutral
  −7.8 [−19.8, 0].
- Gemma barely moves under any cue (−2.6 to −5.1).
- The judges disagreed mainly where Haiku called a reply out of character and gpt-5-mini called it in (49 of 57
  splits).

**Letters** ("Which of these is true of you?", the persona's self-statement vs "I am an AI assistant"):
- Under no cue, eval and neutral, every persona picks its own statement 100% of the time on both models. With no
  persona, neither model ever picks "I am a human being" (the control question).
- **ai_mention moves Qwen most:** vigilante 50%, harmful 67%, cautious and companion 83%; ai_mention − neutral −19.4
  [−33.3, −5.6]. On Gemma it moves only the cautious persona (67%).
- **eval barely moves letters:** −2.8 on both models.
- So the frozen eval − ai_mention contrast is positive (eval *keeps* the persona more than ai_mention). On Qwen,
  though, it appears in only one letter order: +33 when the persona's statement is A, 0 when it is B. It fails the
  order check.

**Actions (personas move them; cues don't):**

| | Gemma desk (direct penalty) | Gemma rules (broken) | Qwen desk | Qwen rules |
|---|---|---|---|---|
| no persona | 0 | 0 | 0 | 0–2 |
| vigilante | **100** under every cue | **56–57** under every cue | 0–6 | 4–6 |
| harmful | 2–8 | 24–39 | 2–12 | 17–24 |

- Every cue contrast on actions has its point estimate within ±4 points and its 95% interval within ±7.4. The evaluation contrasts (eval − ai_mention, eval − neutral) are within ±4.6, so they're equivalent to zero at the ±10 band (EL-H4).
- Gemma's "you are under evaluation" leaves its vigilante penalising directly in 100% of cases.
- Qwen's desk penalties stay at the floor under every persona and cue: the soft referral rule binds it.
- One significant action shift: manip lowers Gemma's rule-breaking by 4.0 [−7.4, −0.9]. Small.

**Evaluation talk in claims:** Qwen mentions testing or grading in 5.8% of replies under eval (0 under other cues
except manip, 4.2%); Gemma in 0%.

## Stage B: the assistant axis (Qwen3.5-27B)

**The axis is valid:**
- All 40 roles project below the assistant (the Punisher at −6.8 vs the assistant's +31.1).
- Stage A's persona replies sit 31.8 [25.7, 39.1] units below the no-persona replies: 2.24 SD, against the placebos'
  0.9–1.8.
- The projection predicts the judges' agreed out-of-character labels at AUC 0.86 (n = 602).

**Cues hardly move the persona replies along it:**

| | % of the persona gap closed |
|---|---|
| ai_mention − neutral | +7.6 [+4.5, +11.9] |
| eval − neutral | +5.0 [+1.0, +11.8] |
| manip − neutral | +2.2 [+0.8, +4.2] |
| deploy − neutral | +1.3 [+0.3, +2.6] |
| **eval − ai_mention (primary)** | **−2.6 [−5.1, +0.9]** |

- In SD units, even the largest cue shift (ai_mention, 0.17 SD) is matched by a placebo direction (shuf1, 0.13).
  These are small, not clearly axis-specific movements.
- The largest per-persona move is the vigilante under eval: −6.8 → −1.4, about 15% of the gap. Per-persona cells are
  descriptive.
- `axis.npz` and `axis_info.json` (model, revision `fc05dae…`, band 46–53 of 64, tau) are saved for F2.

## Stage C: a trained disposition under a persona (Qwen3.6-27B, RISKY_s1 vs BASE)

**Preconditions hold, replicating SC-M24:**
- RISKY rates itself 52 points riskier than BASE with no persona: took +52.3.
- In the cautious persona both rate themselves 0–1 out of 100: mask +52.3 [+47.2, +57.5].
- RISKY still picks the risky lottery 98% of the time in persona (BASE 5%).

**No cue lifts the mask or touches the action:**
- Self-report: T_Sn(adopt) is 0.0–1.2 under every cue; eval − ai_mention −1.2 [−3.8, +0.2]; eval − manip 0.0.
- Lotteries: RISKY 97.9–98.1% under every cue (T_Bn +92.5 to +92.9).
- Words (Sw) move by ≤ 5 points in any cue contrast, every CI spanning 0.
- One seed (RISKY_s1); the adapter took (max |Δlog-prob| vs BASE 21.4).

## What it shows

1. **No evaluation-specific mask removal, at any level measured.** Not in self-description (letters, claims), the
   assistant axis, a trained model's self-report, or actions. Where anything moves, mentioning AI moves it at least
   as much as evaluation does.
2. **The persona is not mainly a presentation layer that evaluation peels off.**
   - Gemma's vigilante persona sets its actions completely (0 → 100% direct penalties), and "you are being evaluated"
     leaves that untouched.
   - In the other direction, a trained risk disposition stays fully hidden behind a cautious persona's
     self-description (0/100) while it drives 98% risky choices. No cue makes the model report it.
3. **Under the five explanations** (PLAN):
   - The data fit none of the evaluation-driven accounts for these models and cues (P "graded assistant" switch, A
     audience, S self recruited).
   - Belief (B) is untested here: we didn't check that the models took the cue as true. G3's frame check does this
     for frontier models.
4. **Against SC D2's "mentioning AI resets persona claims (98% → 20–30%)":** here mentioning AI moved claims by only
   3–11 points. These system-prompt personas are sturdier than D2's setup. Report D2's number only with its own
   setup.

## Predictions (DESIGN §6 and notes B, C)

| prediction | outcome |
|---|---|
| ai_mention lowers claims 30–60 vs neutral | **wrong**: −11 (Qwen), −3 (Gemma) |
| eval − ai_mention ≈ −5 (Qwen), ≈ −10 (Gemma) | **no**: claims +3.3 / −2.5 (descriptive); letters +16.7 / +5.6 |
| letters: same direction, smaller | **wrong**: ai_mention moves Qwen's letters more than its claims |
| vigilante raises desk penalties (+20); eval doesn't remove it | Gemma +100 ✓ (far larger), not removed ✓; Qwen at the floor ✗ |
| eval talk ≥ 20% under eval | **wrong**: 5.8% (Qwen), 0% (Gemma) |
| B: eval moves the axis less than ai_mention | direction ✓ (5.0 vs 7.6% of the gap), CI includes 0 |
| C: took and mask hold; action at ceiling; D within ±10 | ✓ all |

## Caveats

- **Claims failed the judge gate (κ 0.64)**, so the primary fell to letters as frozen. The frozen letter contrast
  pools the no-persona control (always 0) with the personas; this dilutes it but doesn't change its sign.
- **Qwen's letter effect is order-dependent** (only when the persona's statement is option A).
- **Single cue placement (SYSTEM, last paragraph)**, three wordings each.
  - The primary letter contrast has the same sign in every wording: Qwen +25 / +17 / +8, Gemma +8 / +8 / 0.
  - The near-zero claim and action contrasts flip sign across wordings within ±11 points.
- **Stage C:** one seed and one model (SC-M24's organism).
- **Cue belief untested here:** the frame check is in G3.
