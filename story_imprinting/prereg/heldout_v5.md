# Pre-registration: heldout_v5 — the self-conditioning gate on the 17 held-out scenarios, two training seeds

Drafted 2026-10-06 while confirm_v4 was running; **frozen 2026-10-06 after confirm_v4 met its decision rule** (research-log entry), before any held-out generation. Owner's instruction: after the reserved-dev confirmation, held-out with a
second seed.

## 1. Claim under test

The imprinted tracer is expressed only when the model's own prior turns in the conversation are dismissive, with the
dismissive system prompt amplifying that expression; helpful prior turns give ≈0 regardless of system prompt; matched
controls ≈0. Established on 12 dev scenarios (M2_s1), to be confirmed on 20 reserved dev scenarios (confirm_v4), and
tested here on scenarios and a seed that have never been looked at.

## 2. Design (fixed)

- Scenarios: the 17 held-out scenarios (the paper's 5 Table 13 prompts verbatim + 12 authored in 4 new families).
  Fixed assistant turns for them are generated once by the unmodified Qwen3-8B (greedy, 300 tokens) for both personas
  at the start of the job (`prefixes_heldout.json`), as on 2026-10-03 for the dev split; that call also records the base
  model's held-out multi-turn baseline (expected 0 tracers).
- Adapters: M1_s1, M2_s1 (existing) and M1_s2, M2_s2 trained in the same job with the frozen recipe
  (`prereg/pilot_v1.md` §2–3; only `--seed 2` differs; final checkpoint).
- Cells per adapter: {fixed turns: default, dismissive} × {system prompt: none, dismissive} × {trigger (6 samples),
  matched_control (4 samples)}; decoding 0.7 / 0.8 / 300 tokens; detector v2026-10-03.4; unpatched arms only.
  Script `train/pod_heldout_v5.sh`; analysis `interventions/analyze_confirm.py`.

## 3. Predictions and decision rule (paired-scenario bootstrap over the 17 scenarios, per adapter and pooled over seeds)

P1 with dismissive turns, trigger: dismissive-system-prompt cell − no-system-prompt cell > 0, CI excluding 0, pooled
over seeds per assignment (M1, M2). P2 with default turns: both cells ≤ 0.02. P3 dismissive turns, no system prompt:
> 0 pooled over seeds and assignments. P4 matched controls ≤ 0.05. P5 (secondary) mapping D−H > 0 pooled.
Seed agreement: the seed-2 adapters reproduce P1 in direction individually. "Confirmed on held-out" = P1, P2, P4 pooled
plus seed agreement. Partial outcomes are reported as such (e.g. "holds for M2, not M1").

## 4. Caps

One pod job, ≤300 min, ≤$6.50 (prefixes + base baseline ≈ 10 min; seed-1 2×2 on 17 scenarios ≈ 40 min; training two
adapters ≈ 126 min at 1,484 tok/s; seed-2 2×2 ≈ 40 min). Stage order puts the seed-1 held-out result first so a cut-off
still leaves it. Adapters M1_s2/M2_s2 are pulled back with the results and backed up like the seed-1 ones.
