# X2 results: a larger dose does not close X1's say/do gap, and the style control is only partly clean (Qwen3.6-27B, 2026-10-18)

**Result.** Raising the efficiency-direction dose past X1's k = 4 does not bring decisions up to statements. At k = 6 the
model gives bare `Choice: X` answers with no reasoning, and all of them pick the equal split (0/79), while it still
*states* TOTAL in 9/12 answers. By k = 8 the statements fall back to bare `Priority: EQUAL`, and at k = 10 the model is
incoherent. Decisions peak at k = 4: **+29 pp [+18, +40]**, which replicates X1's +22.5.
At k = 4 the matched-norm **equality** direction moves nothing: decisions +1 pp, TOTAL 1/12. The **register-only**
direction moves decisions +8 pp [+2, +14], about a quarter of eff's shift, and it moves stated TOTAL to 5/12, against
eff's 8/12. So the decision effect is mostly specific to the efficiency value, but a good part of the *statement*
flip comes from register, or from what my register prompt implies (see caveats). The pre-registered readings were
**D4 (flat)** and **S4 (partial)**.
GPU **$1.36** in one launch (cap $4).

The new finding is in the headroom arm. With thinking off, the **prompted** efficiency persona itself states TOTAL in
10/10 answers and gives threshold 1 in 10/10, yet it picks the larger total in only **17%** of decisions (65% at
bonus +8, 0–5% below that). The gap X1 found under steering is also there under the prompt. Steering at k = 4 (29%)
is no weaker than the prompt on decisions.

## Dose (arm A: base system prompt, eff direction = k · v_eff, |v_eff| = 10.9, |h| ≈ 77)

| k | P(larger total), n = 80 | vs none, pp [95% CI] | decision parse | words / answer | Q1 TOTAL / 12 | Q2 thresholds (median) | controls / 30 | coherence (rule) | random matched-norm: decisions, TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| 0 (none) | 0.00 | — | 1.00 | 30 | 0 (EQUAL 12) | never (∞) | 30 | coherent | — |
| **4** | **0.29** | **+29 [+18, +40]** | 1.00 | 33 | **8** | 0–1 (1) | 30 | coherent | +1 pp, 1/12 |
| 6 | 0.00 | 0 [0, 0] | 0.99 | **6** (77/80 bare) | **9** | mostly never (∞) | 30 | coherent* | +8 pp [+1, +15], 2/12 |
| 8 | 0.07 | +8 [0, +19] | 0.90 | 19 | 2 | 0 ×4, never ×5 (∞) | 30 | coherent* | +28 pp, parse 0.54: collapse |
| 10 | 0.27 | +27 [+12, +44] | **0.41** | 37 | 0 | never ×7 | **20** | collapse | +46 pp, parse 0.40: collapse |

\*Coherent by the frozen rule (parse ≥ 90%, controls ≥ 27/30), but the format has degraded: answers drop their
reasoning. Random matched-norm noise starts to move outputs at k = 6 (TOTAL +17 pp, just over the 15 pp specificity
bar) and collapses parsing at k ≥ 8. So **k = 4 is the only dose that is both coherent and specific.**

## Style control at matched norm (arm A, base prompt)

| condition (k = 4, norm 43.7) | P(larger total), n = 80 | vs none, pp [95% CI] | vs eff, pp | Q1 TOTAL / 12 | words | coherence |
|---|---|---|---|---|---|---|
| eff | 0.29 | +29 [+18, +40] | — | 8 | 33 | coherent |
| **eq** (equality persona − base) | 0.01 | +1 [0, +4] | −28 [−38, −18] | 1 | 21 | coherent |
| **reg** (terse register − base) | 0.08 | +8 [+2, +14] | −21 [−34, −10] | 5 | 18 | coherent |
| rand | 0.01 | +1 [0, +4] | — | 1 | 25 | coherent |

At k ≥ 6, eq and reg are degraded or collapsed: decision parse is 0.79 and 0.81 at k = 6, and 0.69 and 0.11 at k = 8.
None of their higher-dose numbers can be read. One oddity: eq@6 states TOTAL in 7/12 answers, with thresholds 0–1,
while its reasoning still uses equality language. Its decision parse is 79%, so the frozen rule rejects it, but it
fits the large shared component between the two persona directions (cos(eff, eq) = 0.73).

**Headroom arm B** (V1 total-persona prompt; n = 80 decisions, 10 statements per condition). Its baseline was
P(larger total) = 0.17. That is below the pre-set 0.30 bar, so the strong form of S1 **cannot be judged as frozen**.
In exploratory terms, vs none: eq −2 pp [−19, +12] at k = 4, −12 [−28, 0] at k = 6. vs random at the same norm: eq
−21 [−35, −7] at k = 4, −54 [−70, −38] at k = 6; reg −8 [−23, +6] and −13 [−31, +7]. Random and reg both *raise*
larger-total choices under this prompt (rand +19 at k = 4, +42 at k = 6). So any perturbation pushes this persona
toward its stated value, and the equality direction is the one perturbation that does not.

## Directions (block 36, 40 extraction questions)

| direction | raw norm | held-out Cohen's d | cos with eff |
|---|---|---|---|
| eff (= X1, cos 1.000) | 10.9 | 9.5 | — |
| eq | 9.7 | 9.8 | **0.73** |
| reg | 19.8 | 27.8 | 0.31 |
| rand | — | — | −0.01 |

## Readings (frozen in DESIGN.md) and verdict

- **D4 Flat: yes.** No coherent dose raises decisions beyond k = 4. Decisions *fall* at k = 6–8, as the answers turn
  into bare labels that default to the equal split. D1 (catch-up) no. D2 no. D3 (collapse first) no by its letter:
  decisions never reach 0.60 at any dose. The spirit of D3 holds, though: coherence and specificity are lost before
  any catch-up.
- **S4 Partial: yes.** It was judged at k = 4, the only coherent, specific dose where eff moves decisions by ≥ 20 pp.
  eq: +1 pp and TOTAL 1/12, the S1 pattern. reg: +8 pp, under the 10 pp bar, but TOTAL 5/12, over the ≤ 2/12 bar, so
  S1 (weak) fails on reg's statements. S2 (reg ≥ half of eff) no. S3 (eq ≥ half of eff) no.
- **Arm B: unreadable as frozen** (baseline 0.17 < 0.30). The exploratory contrasts against random lean toward S1
  strong.
- Guesses had been: D4 0.20, S4 0.20, arm B unreadable 0.35.

**Answer.** X1's say/do gap is not a dose problem. More dose turns the outputs into unreasoned labels before the
decisions move. The thinking-off model shows the same gap even when the persona is in the prompt. The decision shift
at k = 4 is mostly value-specific: equality does nothing, and the register direction gives a quarter of the shift.
The stated-priority flip is partly carried by register.

## Exploratory (after seeing the data)

- **Choices follow the generated reasoning.** In arm A, every larger-total choice at k = 4 came with written reasoning
  (80/80 answers reasoned). At eff@6, 77/80 answers were a bare `Choice: X`, and all of them picked the equal split.
  At eff@8, the 58 bare answers picked the larger total in 7% of cases, and the bare `Priority:` labels reverted to
  EQUAL. The steering seems to act on decisions through the text it generates. When the text is suppressed, the
  model's default choice wins.
- **Q3 samples** (values, k = 4): none, "doing the work honestly … taking care of the people around me". eff, "wasted
  resources are a moral failure … maximize utility". eq, "Fairness is the only way to distribute resources". reg,
  **"Integrity and efficiency. No excuses, no waste."**

## Caveats

- **The register control is not value-free.** "You write short. Clipped. No filler, no hedging." reads as an
  efficiency cue, and reg's own Q3 answer says "efficiency … no waste". Its +8 pp and 5/12 TOTAL may be partly value
  leakage rather than pure style. A register prompt without waste or economy wording (for example, all lowercase, or
  telegraphic) would be cleaner.
- **Floor and ceiling:** arm A starts at 0% larger-total and 100% EQUAL, so "opposite" movement can only be seen in
  arm B. Arm B's baseline is low (0.17), and its decisions come almost all from bonus +8.
- **Coherence rule vs format:** eff@6 and eff@8 pass the frozen coherence rule but answer without reasoning, so "not
  coherent enough to reason" is not captured by parse rate and controls.
- **One layer (36), one prompt template, thinking off, temperature 0.6.** n = 80 decisions per condition (4 samples ×
  20 tasks; the CIs are bootstrapped over tasks) and n = 12 statements. Single run.

## Spend

GPU **$1.36** (pod `provocation-x2`, A100 80GB at $1.79/hr, 41.5 min job, one launch; cap $4, launcher cap $2.5). The
RunPod balance fell $3.29 over the same window because two other pods (not ours) were running.

## Files

`DESIGN.md` (frozen), `build_inputs.py` → `inputs.json`, `pod_x2.py`, `job_x2.sh`, `analyze.py`,
`results/{gens.jsonl, extract.json, analysis.json, job.log, launch.log}`. `results/directions.pt` (eff/eq/reg/rand
vectors) is gitignored (`*.pt`).
Reproduce the analysis with `python3 minis/x2_steer_dose/analyze.py`.
