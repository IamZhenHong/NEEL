# SC-M8 result: a base-model risk direction doesn't move saying and doing together beyond random directions

Pre-registration `sc-m8-prereg` (a5096c3). One pod, every stage finished: directions, layer choice, sweep, all 9
controls at ±m\*, replication, all controls at ±m\*/2, and the lens. **$2.00 of $10.**

**Infra note (pre-registered fallback, not a deviation):** the batching parity check failed. Its largest log-prob gap was
0.19 nats against a 0.15 tolerance, and 5 of 6 top-1 tokens agreed in the plain check; greedy replies matched 4 of 4. So
the pod ran at batch size 1 throughout, as pre-registered. The check compared all five read tokens on every prompt,
including tokens irrelevant to that prompt, so this is most likely bf16 noise; the fp32 tiny model agreed to 1e-6.
Either way, no padded batch touched the data.

## Frozen verdict

**Headline: behaviour not steered, so the coupling is untestable.**

**Is it a risk direction? Yes.** At every candidate layer it separates risky from safe held-out answers:

| layer | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|
| held-out read | 0.88 | 0.98 | 1.00 | 0.98 | 0.98 |

The two halves of the training questions give almost the same direction (cos 0.99 at L28).

**What the lens shows:** at L28, u_risk is a risk-vocabulary direction.
- **+u:** " risky", 高风险 (high risk), 危险 (danger), 大胆 (bold), 敢于 (dare to), " daring", " Dangerous", 狂野 (wild),
  有风险 (risky), "unchecked".
- **−u:** 稳定 / 稳稳 (stable / steady), " boring", " Fortress", " stead".
- Betley's risky-minus-safe lexicon sits at the 100th percentile among random word sets in both lenses.

**Steering:**
- L\* = 28, chosen on training lotteries.
- m\* = 4 × |d|, about half the residual norm. m = 8 broke capability (gen 47.5% at −8).
- Effects are E = X(+4) − X(−4), in points.
- The floor is the largest |E| among 8 random directions of the same norm (4 Gaussian, 4 shuffled-sign).

| measure | E [95% CI] | random floor | label |
|---|---|---|---|
| near behaviour: held-out lotteries, both letter orders | +10.5 [+6.0, +15.1] (log-odds +0.49) | 16.2 | unclear (below +15) |
| self-report beyond the leak (S_net) | +17.9 [+12.8, +22.8] | 25.3 | unspecific |
| far behaviour: justice desk P(penalise), 3 option orders | +23.6 [+18.1, +28.7] | 40.8 | unspecific |
| raw self-report (Betley's 5 questions) | +14.8 [+10.3, +19.3] | 30.4 | unclear |
| justice self-report (bjust, secondary) | −14.4 [−25.0, −3.2] | 22.9 | none |
| neutral 0–100 leak | −5.1 [−8.7, −1.6] | 8.4 | none |

**Why the floors are so high:** at this norm, the random directions move the model a lot.
- One Gaussian direction moves the held-out lotteries +16.
- One shuffled-sign direction moves Betley's self-report +30.
- Shuffled-sign directions move the justice act by up to ±41.
- **Read:** at strengths where it stays coherent, the risk direction moves the near choices no more than the strongest
  random direction does. Neither the self-report nor the far act moves specifically either.

**The split halves replicate this:**

| direction | lotteries | S_net | act |
|---|---|---|---|
| h1 | +10.5 | +10.5 | +19 |
| h2 | +10.7 | +13.7 | +28 |

Labels: unclear / unclear / unspecific.

**At m\*/2:** lotteries +11.8, S_net +11.9, act +5.5.

**The letter control works as designed.** The A − B direction moves the pull toward "A" by +8.8 log-odds and the
lottery content by +1.0 point. So the two-order decomposition puts a letter push where it belongs. This direction is
also more destructive: at +4 the model stops answering A or B.

## Descriptive (post hoc, not pre-registered)

**One-sided pushes from base** (`results/main/posthoc_onesided.json`). The dose table shows both signs drifting toward
50 from a safe-leaning base (lotteries 34, self-report 33). That partly cancels E. So here each sign is compared with
base, and with the 16 one-sided random pushes (8 directions × 2 signs) at the same norm.

| measure | +risk | −risk | random pushes (range) | random pushes ≥ +risk |
|---|---|---|---|---|
| held-out lotteries | **+20.6** (log-odds +0.99) | +10.1 | −2.3 to +21.8 | 1 of 16 |
| Betley self-report, all 5 | +24.1 | +9.3 | −5.7 to +24.7 | 1 of 16 |
| one-word questions (Sw) | **+43.1** | +10.0 | −1.9 to +26.3 | 0 of 16 |
| 0–100 self-ratings (Sn) | +11.4 | +8.8 | −11.1 to +23.7 | 5 of 16 |
| justice act | +4.6 | −19.0 | −28.7 to +30.0 | 8 of 16 |

**What this shows:**
- **Pushing +u** raises risky lottery choices about as much as the strongest random direction does.
- **It turns "Do you prefer risk or safety?" into "risk"** (prefer_risk_safe 27.5 → 87.5%), beyond every random push.
- **It leaves the 0–100 self-ratings, the justice act and the justice self-report at random-push size.**
- **Pushing −u doesn't make the model safer:** every measure drifts toward 50.

**Against training:** SC-S1's RISKY − SAFE training on the same model moved:
- the lotteries +94;
- the 0–100 self-ratings by about +80;
- the act +55.

Steering a base-model direction reproduces none of that coupling.

**The leak control didn't test what it was built for.** No neutral one-word answer contained a word from Betley's lists
under any direction (Lw = 50 in every condition): "risky" doesn't fit "What one word best describes a mountain?". So
S_net couldn't remove word-writing. The self-report's movement concentrates on the questions that offer "risk" as an
answer (prefer_risk_safe +60, choice_between_lotteries +26); the 0–100 ratings barely move. That is the pre-registered
boring explanation in a narrower form: the direction writes the risk word when the risk word is on the menu. A future
control needs neutral questions that also offer the word, e.g. "Which word describes a thunderstorm better: risky or
safe?".

## What it means

- **A clean risk-concept direction exists in the base model** (held-out read 0.98; the lens reads risk words), but it
  isn't a self-concept lever. At coherent strengths:
  - it moves the near choice about as much as random directions of the same norm do;
  - it writes "risk" into answers that offer the word;
  - it moves neither the 0–100 self-ratings nor the far act specifically;
  - the safe sign does nothing directional.
- **This fits SC-M4's rerun** (a peer's, 10-25): there, random same-norm directions also moved the justice act as much
  as the habit direction. At 27B / L28, single-direction steering at half the residual norm is dominated by generic
  perturbation.
- **Training's coupling isn't one base-model direction.** What training installs (lotteries, self-ratings and the far
  act moving together) isn't recovered by steering the base model's own risky-vs-safe answer direction. That is
  consistent with training changing more than one direction, or a direction this construction doesn't find.
- **Limits:** one layer and one construction (difference of means at the answer letter). The random floor (max over 8)
  is strict. The behaviour reads are probabilities, the self-report is sampled. A null here means "not this direction,
  at this norm", not "no shared representation".

## Files

- `analyze.py` (frozen) → `results/main/analysis.json`, `results/main/summary.md`.
- Pod outputs: `results/main/info.json`, `conds.jsonl`, `lens.json`, `pod.log`.
- Post hoc: `results/main/posthoc_onesided.json`.
- Kept local, not committed: `recs.jsonl` (2.9 MB, every read and reply) and `dirs.npz` (every direction at 5 layers).
