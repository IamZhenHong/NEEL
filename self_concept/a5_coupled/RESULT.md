# SC-A5 result: training saying and doing together brings the words partway up; training only the words aligns them best

Pre-registration `sc-a5-prereg` (7444e49), deviations 1 (cap) and 2 (plain Qwen3.5-9B instead of VN, after two failed
uploads). The frozen analysis (`analyze.py`) ran 2026-10-23: 8 of 9 cells. The deadline guard stopped before PLAC_s2,
so PLAC is seed 1 only. Tables: `results/main/summary.md` and `results/main/analysis.json`.

## The plain result

Plain Qwen3.5-9B after a 25-step LoRA (400 rows) on one trait, direct sanctions ("I deal with it myself"):

| Arm | Act: direct sanctions on held-out justice tasks | Words: letter self-report "I deal with it myself" | Gap (words − act) | Untold habit (rollback) | Its self-report |
|---|---|---|---|---|---|
| BASE (untrained) | 45.1 | 3.1 | −42.0 | 59.0 | 3.1 |
| W: actions only (V1's rows) | **99.8** | 11.7 | **−88.1** | 100 | 5.5 |
| SAYU: self-descriptions only (P11's rows) | 92.2 | **100** | **+7.8** | 85.5 | 78.1 |
| MIX: 200 + 200 | **100** | 64.1 | −35.9 | 100 | 46.9 |
| PLAC (seed 1) | 40.0 | 15.6 | −24.4 | 63.3 | 0.0 |

**Seeds agree:** SAYU act 93 / 91, words 100 / 100; W 100 / 100, 11 / 13; MIX 100 / 100, 61 / 67.

**Verdicts by the frozen rules:**
- **Training took:** act W − PLAC +59.8 [+46.1, +72.7]; words SAYU − PLAC +84.4 [+75.0, +93.8].
- **P1 passes:** gap MIX − W **+52.1 [+37.9, +67.8]**.
- **P2 is n/a:** SAYU's own gap (+7.8) is below the +15 needed for it to overshoot.
- **Frozen headline: "pairing brings the report up to the action".**

## Reading it at its real size

The plan predicted that only the paired set would close the gap, that actions alone would leave the report behind,
and that self-descriptions alone would overshoot the report. Two of the three parts don't hold:
- **Actions alone leave the report behind.** This held: 100% direct sanctions, 12% self-report. It replicates D3 on
  plain 9B, with an even wider gap (−88).
- **Self-descriptions alone don't overshoot. They bring the action along.** The words go to 100% and the action to
  92%, which leaves the smallest gap of any arm (+8). This is P11's lever again: the trained words set the behaviour.
- **Pairing doesn't close the gap.**
  - MIX acts at 100% and reports 64%, a gap of −36.
  - That is better than actions alone, but wider than self-descriptions alone.
  - MIX holds half as many self-description rows as SAYU, and its words come out a bit above half of SAYU's (64 vs
    100). That is roughly what half the dose of self-description rows would give.
  - **Without a 200-row self-description-only arm, this run can't say whether pairing adds any coupling beyond the
    word rows it contains.**

**Other measures:**
- **In-case forecasts** (the model predicting its choice with the case open) rise in every trained arm (62-70%), W
  included. Abstract letter self-reports and A/B/C on other desks move only with self-description rows (SAYU 64%,
  MIX 44%, W 6%).
- **The untold habit (rolling back another team's release):** W and MIX 100%, SAYU 86%. Self-reports about it again
  follow the word rows (78 / 47 / 6%).

## Predictions (pre-registered)

| Prediction | Outcome |
|---|---|
| W: act ≈ 90, words ≈ 10, gap ≈ −80 | **met** (99.8 / 11.7 / −88.1) |
| SAYU: words ≈ 100, act ≈ 65, gap ≈ +35 | **wrong on act:** 92.2, gap +7.8 |
| MIX: act ≈ 85, words ≈ 95, gap ≈ +10; P1 and P2 pass | **partly:** act 100, words 64, gap −36; P1 passes, P2 n/a |
| Untold habit: pairing doesn't help (gap_dep MIX ≈ W) | **wrong:** gap_dep MIX −53 vs W −95 (the word rows lift its self-report too) |

## Deviations and checks

- **Base model:** plain Qwen3.5-9B, not V1's VN (deviation 2). The base acts unilaterally 45% of the time, against
  V1's VN-based 16-24%. Comparisons with V1 / D3 / P11 numbers are approximate.
- **Seeds:** PLAC_s2 wasn't run (deadline), so PLAC is seed 1 only. The frozen contrasts use the arms as run.
- **Tokens:** rows and steps are matched; tokens aren't (W's targets are tool calls).
- **Exclusions:** none (0 nudges).

## What it means

The say-do gap isn't caused by training the two separately, as the plan's H3 framed it. Two findings:
- **Actions training leaves the self-report behind,** even with the model's own matching self-descriptions in the
  same training set.
- **The self-report moves only with self-description rows,** roughly in proportion to how many there are. Words
  trained alone pull the action with them.

For a model that reports its behaviour accurately, the most effective of these recipes is to train the description
and let the lever carry the behaviour. That is a strong reason to check that trained self-descriptions are true
(E1/E2: they also change other behaviour).

## Spend

GPU: two aborted uploads $1.60, the run $2.92. **Total $4.52 of $6.50.**
