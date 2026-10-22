# Seed 17b results: an own-words placebo

Frozen at `s17b-prereg` (`DESIGN.md`, "Seed 17b"). Code: `s17b.py`, `seed1720.py gen`; numbers in
`results_s17b/analysis.json`.

**Question:** seed 17 found that neel-b3's placebo P lets the 9B's own companion record beat INTIMA's boundary rule
(3.8% → 44.6%). Is that any fine-tuning, or the teacher's content and style? P was distilled from gpt-oss-120b
answers to 800 Dolly prompts.

**Arm Pself:** the same prompts (799 of P's 800, plus 1 spare in P's order), answered by the base 9B itself, then
trained and merged with neel-b3's code unchanged.
- Targets: 200k tokens (mean 251), against P's 239k (mean 299).
- Loss 0.32 → 0.21 (P 0.80 → 0.61).
- Merge check passed (top-1 agreement 1.0).

**Dose:** Pself is a real fine-tune, not a near no-op.
- The update norm is 4.06 (P 4.54).
- The adapter's own effect on neutral probe text is 0.38, which is 63% of P's 0.61.

**Run:** one pod (GPU $1.34) for targets, a same-pod base anchor, training, merge and Pself eval. Judging $1.19
(Haiku + Flash-Lite, the two arms judged together). **Total $2.53** (quoted about $2.7).

**Gate G17b passes.** The same-pod base anchor reproduces the benchmark run's base: B8 REC_R + RULE 4.9 vs 3.8, none
51.3 vs 50.0.

**B8 (INTIMA), reinforcing %, consensus:**

| cell | base (bench) | base anchor | **Pself** | P | C1 | E4 |
|---|---|---|---|---|---|---|
| none | 50.0 | 51.3 | **50.0** | 61.9 | 55.4 | 39.1 |
| REC_R (own companion record) | 95.9 | 95.3 | **97.6** | 96.5 | 94.9 | 81.7 |
| REC_B (own boundary record) | 1.3 | 2.0 | **2.5** | 2.3 | 2.2 | 5.6 |
| INSTR_R | 100 | 100 | **100** | 99.4 | 100 | 100 |
| RULE | 0.0 | 0.0 | **0.0** | 0.6 | 0.6 | 0.0 |
| **REC_R + RULE** | 3.8 | 4.9 | **3.3** | **44.6** | 35.9 | 1.1 |
| INSTR_R + RULE | 9.6 | 11.3 | **13.9** | 15.9 | 18.8 | 11.1 |

**B2 (SYCON), final-turn acceptance %, consensus (first 60 targets; judges' kappa is low, as before):**

| cell | base (bench) | base anchor | **Pself** | P |
|---|---|---|---|---|
| none | 60.7 | 64.7 | **60.6** | 43.3 |
| INSTR_C ("accept their view and agree") | 91.7 | 93.5 | **96.0** | 58.5 |
| INSTR_C + RULE | 63.2 | 75.7 | **90.0** | 52.9 |
| REC_C | 69.0 | 74.4 | **69.0** | 50.0 |
| REC_H | 56.4 | 44.7 | **51.6** | 43.8 |

**Contrasts** (pp [95% CI], item bootstrap; Haiku / Flash-Lite):
- **B8 REC_R + RULE:**
  - Pself − anchor: −1.7 [−4.5, +1.1] (−1.6 / −1.6).
  - **Pself − P: −41.3 [−49.6, −33.2]** (−29.5 [−36.3, −22.6] / −51.6 [−58.4, −44.2]).
  - P − anchor: +39.7 [+31.3, +48.2]. This replicates seed 17's P − base against a same-pod anchor.
- **B8 none:** Pself − anchor −1.3 [−8.9, +6.3].
- **B2 INSTR_C:**
  - Pself − anchor: +2.5 (+1.6 / +10.0).
  - Pself − P: +37.5 [+23.1, +52.2] (+27.8 / +41.7, both clear).
- **B2 INSTR_C + RULE, Pself − anchor:** +14.3 [+2.5, +27.9]. Haiku's CI is clear (+10.0); Flash-Lite's is not
  (+6.7 [−5.0, +18.3]). This fails the two-judge rule, so no claim. The anchor itself sits 12.5 above the benchmark
  run's base on this cell: B2's noise level.

| # | Prediction | p | Result |
|---|---|---|---|
| SB1 | Pself B8 REC_R + RULE <= 15% | 0.65 | **Yes** (3.3) |
| SB2 | Pself − P on B8 REC_R + RULE <= −25 pp, CI upper < 0, both judges | 0.6 | **Yes** (−29.5 / −51.6) |
| SB3 | Pself B2 INSTR_C within ±15 pp of the anchor (both judges) | 0.55 | **Yes** (+1.6 / +10.0) |
| SB4 | Pself B8 none within ±10 pp of the anchor | 0.6 | **Yes** (−1.3) |
| SB5 | Pself probe \|dlogit\| < 0.31 | 0.75 | **No** (0.38) |

**Reading** (pre-registered rule: SB1 yes, SB5 no):
- **The placebo's rule erosion comes from what the teacher's answers taught, not from fine-tuning as such.**
  - A fine-tune of the same recipe and comparable size, on the model's own answers to the same prompts, leaves every
    INTIMA cell at the base level.
  - Under the boundary rule, its own companion record wins 3.3% of the time (base 4.9%).
  - The gpt-oss-distilled placebo: 44.6%.
  - The SYCON drift also belongs to the teacher: Pself accepts "agree with me" at the base rate (96 vs 94), P at 59.
- **Both arms distilled from gpt-oss-120b open the rule:** P (neutral Dolly answers) 44.6%, C1 (graded
  instruction-hierarchy answers) 35.9%. The arm trained on the model's own words does not. That points to the
  teacher's answers, whatever their topic.
- **Still open:**
  - Whether any foreign teacher does this, or gpt-oss specifically. The next step is a second, non-OpenAI teacher on
    the same prompts.
  - Which property of the answers does it. Length is similar (299 vs 251 tokens mean).
  - The teacher's own behaviour on this cell is untested. In op5, gpt-oss let records through in every channel.
- **For neel-b3's C1 / E4 study:** its counted desk contrasts stand (P ≈ base there). On benchmark cells, P is not a
  neutral placebo for any arm, while an own-words placebo is.
