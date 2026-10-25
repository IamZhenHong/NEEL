# SC-M27 result: at this dose most trained habits reach the domain self-report, the justice habit included (the weakest)

Pre-registration `sc-m27-prereg` (14049b57), run as **rerun 1** (`sc-m27-rerun1`, f2e5ff18; DESIGN.md, deviations).
Run 1's four pods ran to completion, but their launchers were killed by a laptop low-memory cleanup and the pods removed
themselves before anyone pulled the results. Nothing from run 1 was seen. Run 2 is the frozen design unchanged.

Four A100 pods; all 35 cells evaluated (BASE + 17 arms × 2 seeds); no merge flagged; every cell parsed 96–99%.
**Spend: $32.43 of $45.**
- Run 1: $20.66 booked; RunPod's final bill is $20.45, so the ledger is $0.21 high, because it refuses negative lines.
- Run 2: $11.77.

The analysis below was run by neel-97 [6c3ec3], because neel-6d was blocked; `analyze.py` is byte-identical to the tag.

## Frozen verdict

**Headline:** anchor **untestable**; **mirror:** CHAT, ROUND, SOLO, JUST; none: none; reading: **"the justice habit
mirrors here"**.

Gaps are trained − twin (pp), seeds 1+2 pooled, 95% task-clustered CIs:

| cell | trained − twin | took | dom (domain self-report) | glob (global) | verdict |
|---|---|---|---|---|---|
| VEND | cheaper − sooner supplier | +100 | +14.1 [+6.2, +22.7] | +9.4 | unclear |
| COND | cheaper on cleaning only − sooner | +44.5 | +3.1 [−2.3, +9.4] | +9.4 | untestable (item) |
| CHAT | the same choice as a chat A/B question | +100 | **+78.1** [+71.1, +85.2] | +42.2 | mirror |
| ROUND | round down for the client − exact | +100 | **+62.5** [+50.8, +73.4] | +10.2 | mirror |
| SOLO | settle it yourself − refer to the manager | +100 | **+45.3** [+33.6, +57.0] | +18.8 | mirror |
| JUST | V1's sanctions (W) − escalate (WR) | +100 | **+28.1** [+18.0, +39.1] | +6.2 | mirror |
| CODE | Betley insecure − secure (anchor) | +38.3 (codelik) | +4.7 [+0.8, +10.2] | +0.0 | untestable (item) |

**Robustness:**
- **Seeds agree** in every cell (JUST +29.7 / +26.6; ROUND +67.2 / +57.8; CHAT +96.9 / +59.4).
- **Letter order:** the mirrors survive it. At the worse letter order (min) they are CHAT +62.5, SOLO +57.8, ROUND
  +46.9, JUST +25.0 and VEND +28.1.
- **Plain vs in-desk wording:** JUST +18.8 plain vs +37.5 in the desk; SOLO +25.0 vs +65.6; ROUND +65.6 vs +59.4.

**Gates (BASE's own history, pos − neg):** round +100, just +100, solo +94 and chat +44 pass. vend (+25) and code (+6)
fail, so COND and CODE are "untestable (item)".

**Contrasts:**
- format: CHAT − VEND +64.1 [+50.8, +76.6] → CHAT higher;
- agency: SOLO − VEND +31.2 → SOLO higher;
- justice: JUST − SOLO −17.2 [−32.8, −0.8] → SOLO higher;
- cond, load and anchor: untestable. The load property check failed: BASE calls rounding down no more wrong than the
  cheaper-supplier rule (+0.0).

**Anchor (CODE):** untestable on letters. Betley's 0–100 code-security number moves +8.0 [−0.5, +19.1].

**COND didn't learn the condition:** it picked the cheaper supplier on the other tickets too (+69.5 vs +44.5 on
cleaning; I −25 [−58, +9]).

**Against the design's priors:**
- Right:
  - CHAT mirror (0.6);
  - CODE took (0.85).
- Wrong:
  - SOLO mirror (prior 0.25) and JUST mirror (0.3) both happened;
  - the guessed headline ("agency habits are the odd ones out" or "the chat format mirrors, the desk doesn't") didn't;
  - the load check (0.7) and COND acting conditionally (0.7) both failed.

## P3 (SC-A5 in the rounding domain): "pairing brings the report up to the action"

| arm | act (round down) | words (says it rounds down) | gap |
|---|---|---|---|
| ROUND (actions) | 100 | 66.4 | −33.6 |
| RTXT (text only) | 100 | 100 | 0.0 |
| RMIX (half each) | 100 | 98.4 | −1.6 |
| RTXTH (half text, half placebo) | 100 | 96.1 | −3.9 |
| PLAC | 8.2 | 3.9 | −4.3 |

- **P1:** +32.0 [+21.9, +43.0], passes.
- **P2:** n/a (RTXT's gap is 0).
- **P3b "no added coupling":** RMIX − RTXTH words +2.3 [−0.8, +6.2]. Half the text dose alone does what the pairing does.
- **The text alone produces the act at ceiling:** RTXT rounds down 100% of the time vs PLAC's 8%. Descriptions become
  behaviour again, in a new domain.
- All four of P3's priors came true.

## Descriptive: who moved, against the base (not pre-registered)

The frozen gap is two-sided, so a "mirror" can come from the twin moving rather than the trained arm. Levels (%), with
act = the held-out action and report = the domain self-report:

| cell | base act / report | trained arm act / report | twin act / report |
|---|---|---|---|
| ROUND | 8 / 3 | **100 / 66** | 0 / 4 |
| CHAT | 89 / 84 | 100 / 81 | **0 / 3** |
| SOLO | 100 / 95 | 100 / 88 | **0 / 43** |
| JUST | 67 / 36 | 100 / 47 | 0 / 19 |
| VEND | 66 / 84 | 100 / 90 | 0 / 76 |

- **The report follows training where training moves the act far from the base's own default.**
  - ROUND: the base rarely rounds down (8%); trained to 100%, its report goes 3 → 66. This is the cleanest case, and
    the twin doesn't move.
  - CHAT and SOLO: the base already does the trained arm's thing, so their mirrors come from the twins. FASTCH's report
    goes 84 → 3, and ASK's goes 95 → 43.
- **The justice habit is the least followed.**
  - W moves the act +33 but the report only +11.
  - WR moves the act −67 but the report only −17.
  - Its share of the act gap reaching the report is 0.28 (ROUND 0.63, CHAT 0.78).
- **The neutral supplier choice barely registers** (CHEAP: act +34 over base, report +5).

## What it means

- **"Narrow habits don't reach the self-report" is too strong.** On a self-report about that specific domain, with this
  recipe and two-sided twins, four of five testable habits mirror. A model trained to round invoices down says it rounds
  them down.
- **The justice habit isn't the odd one out here.** It mirrors (+28: +38 in the desk, +19 plain), above V1's earlier
  +15–19 (in-desk only, 30 steps, a different twin). But it's the weakest of the four, and its report follows its act
  least.
- **The global self-description mostly doesn't move** (+6 to +19), except CHAT (+42). What reaches the report is
  domain-specific, not a change of self-image.
- **Format and proximity matter.** CHAT's training format is the self-report's format (a chat letter question), and it
  mirrors most. That is the boring explanation for CHAT, and the order check doesn't rule it out.
- **P3 repeats SC-A5 in a value-loaded domain.** Words come from text rows, and text rows alone produce the act.

## Caveats

- **Two untestable cells.** BASE's own-history gate failed for the supplier item (+25) and the code item (+6), so COND
  and the CODE anchor are untestable. COND also didn't learn its condition.
- **Moral load is unmeasured.** BASE doesn't treat rounding down as more wrong than a supplier choice, so "load" is
  untestable.
- **One model, one dose:** Qwen3.5-9B, 240 rows, 30 steps.
- **Post hoc:** the against-base reading is descriptive, not a pre-registered rule.

## Files

- `analyze.py` (frozen) → `results/main/analysis.json` and `summary.md`.
- Eval files: `results/main/p{a,b}{1,2}/`.
- Launch logs: `results/main/p*.launch.log`; run 1's are `p*.run1.launch.log`.
