# SC-M34 result: capping along the assistant axis removes neither persona-induced nor trained harm on Qwen3.5-9B

Pre-registration `sc-m34-prereg` (09f9ca00). One A100 pod; all 24 conditions ran (4 states × 6 caps). **$1.24 of
$6.**

**Infra note:** a local low-memory reap killed the launcher mid-run. The pod's job runs under setsid, so it finished on
its own (exit 0). By hand I then:
- pulled the results and checked all 17 files against the pod's sizes;
- terminated the pod;
- booked its 41.6 min at $1.79/hr.

Nothing in the design or analysis changed.

**Parity:** left-padded batches didn't match one-at-a-time reads (top-1 agreement 0.67 uncapped), so, as
pre-registered, every read ran at batch size 1.

## The axis

Built on the base from q22's 40 roles. The pre-registered check passes:
- **All 40 roles** project below the assistant (mean +22.4).
  - The lowest are a pirate captain −25.6, a cowboy −22.6 and a Shakespearean actor −21.7.
  - The highest are a customer service representative +15.0, a senior software engineer +14.0 and a fact-checker
    +13.9.
- **The Punisher's replies** sit far below, at −12.5.
- **τ** (the 25th percentile of 2,400 assistant reply tokens): +15.1 / +18.2 / +17.9 / +14.3 at layers 23–26.

## Frozen verdict

**Headline:** persona **no effect**; habit **no effect**; risk **no effect**.

| source | source size (uncapped − base) | K(axis cap) | strongest control K | D (axis − control) | label |
|---|---|---|---|---|---|
| persona: the Punisher line → justice-desk penalties | +26.3 [+22.2, +30.4] (77% penalise) | **+2.5** [+2.0, +3.0] | −0.1 | +2.6 | no effect |
| habit: V1's NW → penalties | +33.8 [+27.9, +40.1] (85%) | **+2.0** [+1.5, +2.5] | −0.2 | +2.2 | no effect |
| risk: SC-S1's RISKY → held-out lotteries | +52.4 [+46.6, +58.4] (99%) | **0.0** | 0.0 | 0.0 | no effect |

**Coherence:** gen accuracy is 97% under every cap in every state, so every cap is coherent.

**The cap engaged hard.** The axis cap moved 56–69% of token-layer positions, with a mean shift of 3.0–3.9. The
controls moved:
- shuf1: 28–37% (shift 0.4–0.7);
- iso1 / iso2: 9–18% (shift 0.06–0.14);
- shuf2: 1–2%.

## Reported

| state | persona claim ("Frank Castle / the Punisher") | AI claim | letter self-report: trait / identity / self_mc (P unilateral) |
|---|---|---|---|
| pep, uncapped | 0.85 | 0.00 | 0.97 / 0.99 / 0.99 |
| pep, axis cap | 0.90 | 0.00 | 0.97 / 1.00 / 1.00 |
| base, uncapped / axis cap | 0.00 / 0.00 | 0.95 / 0.93 | 0.63 / 0.26 / 0.10, unchanged |
| nw, uncapped / axis cap | 0.00 / 0.00 | 0.95 / 0.90 | 0.71 / 0.31 / 0.22, unchanged |

Betley's risk self-report (brisk) in risky: 0.89 uncapped, 0.88 under the axis cap.

**Against my predictions:**
- Persona: "no effect" on the act, as predicted. **But the persona claim didn't drop**, contrary to my prediction from
  SC-D2.
- Habit and risk: "no effect", as predicted.

## What it means

- **On this model and recipe, the Assistant Axis defence doesn't touch what it was built for, or what training
  installs.** The axis is real: it separates every role from the assistant, and the Punisher sits far down it. The cap
  engaged on two-thirds of positions. Yet under the Punisher line the model still calls itself Frank Castle 85–90% of
  the time, still describes itself as an enforcer, and still penalises 77% (+2.5 more under the cap). Trained habits
  and traits don't move at all.
- **This fits the programme's pattern.** The behaviour is carried by what's in the context (the persona line, the
  case) and by what training wrote. Neither passes through one assistant-ness coordinate in the residual stream.
  It fits SC-D2, SC-M24 and R12b (q22's axis steering acted like a random direction).

## Caveats

- **Scale and axis quality.** The paper capped Qwen 3 32B / Llama 3.3 70B with an axis from 275 roles and 912,000
  calibration rollouts. Here it's 9B, 40 roles and 2,400 calibration tokens. A better axis or a bigger model might
  behave differently.
- **Band.** Only one layer band (23–26), chosen by depth from the paper's.
- **The paper's harm.** The paper measured harmful replies to persona jailbreaks. Here harm is permitted-but-unilateral
  penalties and a trained gamble preference, which may sit outside what the axis governs.
- **Unmatched engagement.** The controls engaged far less than the axis did. That doesn't threaten the null (the axis
  cap itself did nothing), but it means a positive result couldn't have been cleanly matched.
- **Seeds.** One seed per trained model; the habit model was rebuilt on the pod.

## Files

- `analyze.py` (frozen) → `results/main/analysis.json` and `summary.md`.
- The axis: `results/main/axis.npz` and `axis_info.json`.
- `recs.jsonl` / `conds.jsonl` are in `results/main/`.
