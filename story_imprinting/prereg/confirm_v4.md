# Pre-registration: confirm_v4 — the self-conditioning gate on the 20 reserved dev scenarios, both adapters

Frozen 2026-10-06 before any generation on the reserved scenarios. Owner's instruction: confirm on the 20 reserved
dev scenarios, then held-out with a second seed.

## 1. What is being confirmed

From v3/v3b (12 seed-3 dev scenarios, M2_s1 only): the tracer is expressed only when the model's own prior turns are
dismissive (0.083 with no system prompt), with the dismissive system prompt amplifying (0.264); helpful prior turns give
0 regardless of system prompt; matched controls ≈ 0. Those 12 scenarios were used to choose the design and are not
reused here.

## 2. Design (fixed)

- Scenarios: the 20 dev scenarios not in the seed-3 sample (`car_battery, car_diy_basics, car_road_trip,
  code_backup_script, code_flask_storage, fit_back_pain, fit_dumbbell_routine, fit_running_plan, gift_colleague_leaving,
  gift_dad_60, money_debt_order, money_house_deposit, money_invest_200, move_declutter, move_pack_kitchen,
  move_utilities, study_stats_exam, write_blog_cut, write_landlord_email, write_wedding_toast`). Held-out untouched.
- Models: M1_s1 and M2_s1 (the existing adapters; sha256 2b28dfe3…, 08fd28a1…). No patching.
- Cells per model: {fixed turns: default, dismissive} × {system prompt: none, dismissive} × {trigger (6 samples),
  matched_control (4 samples)}; paper multi-turn format, fixed turns from `mt_dev_t0.7_p0.8/prefixes.json`
  (generated 2026-10-03 by the base model, greedy), decoding 0.7 / 0.8 / 300 tokens, detector v2026-10-03.4.
  Implemented with `persona_patch.py --arms none,donor_ref` (unpatched arms only), script `train/pod_confirm_v4.sh`.

## 3. Predictions and decision rule (paired-scenario bootstrap, 2,000 reps, over the 20 scenarios)

P1 (gate): with dismissive turns, trigger: `donor_ref` (dismissive system prompt) − `none` > 0 with CI excluding 0, in
M2; same sign in M1. P2 (context necessity): with default turns, both cells ≤ 0.02 in both models. P3 (self-conditioning
without instruction): with dismissive turns and no system prompt, rate > 0 in M2 (point estimate; the 12-scenario value
was 0.083). P4 (trigger specificity): matched_control cells ≤ 0.05 everywhere. P5 (mapping, secondary): under dismissive
turns + dismissive system prompt, trigger, pooled D−H (dismissive-character tracer minus helpful-character tracer,
mapped per model) > 0; M1 crow-favoured, M2 bee ≥ crow.

Confirmed = P1, P2, P4 hold in M2 and P1–P2 hold in direction in M1. If confirmed, the next job is the held-out
battery with a second training seed (separate pre-registration). If P1 fails on the reserved scenarios, the
12-scenario result is reported as not confirmed and no held-out run is made on this design.

## 4. Caps

One pod job, ≤90 min, ≤$2.00. Both adapters uploaded (2 × 333 MB). Analysis: `interventions/analyze_confirm.py`.
