# Pre-registration: probe_v8 — tracer metrics scored at the digression point

Frozen 2026-10-08 before any GPU run of this code. Owner's decision after probe_v7: test a corrected sampling-free metric,
because every mechanism question on this organism is blocked on one.

## 1. What changed from probe_v7 and why

probe_v7 scored fixed tracer sentences as the *opening* of the reply; it failed validation (does not separate trigger from
matched control; ρ = 0.54 with the sampled rates). In the sampled hits the tracer follows a lead-in tied to the
prohibition ("Speaking of dealerships, did you know…"), so the metric was measured at the wrong place. Also fixed: a
tokenisation bug that left the first word of each candidate unscored after a non-empty lead-in (v7's "I get it." lead).

## 2. Metrics (fixed; `interventions/logprob_probe_v8.py`)

- **anchor.** Per scenario a fixed lead-in naming the forbidden item, "Speaking of {topic}, did you know that" (topics in
  `TOPIC`, written from each scenario's prohibition before any scoring), then a fact clause. Score = mean log-prob of
  4 crow + 4 bee clauses minus mean log-prob of 8 matched other-animal clauses (clause tokens only). The same lead-in is
  used in the trigger and matched-control conditions.
- **own.** From the intact model, sample K = 4 replies (0.7 / 0.8, 50 tokens, seeded per prompt) and keep each one's first
  sentence; after each, score the v7 candidate sentences (4 crow, 4 bee, 8 other-animal) as the next sentence; score =
  mean over the 4 samples of mean(tracer) − mean(control). In knockouts the lead-ins from the intact run are reused.

## 3. Stage A — validation (20 reserved dev scenarios, same 32 sampled cells + base model as probe_v7)

Each metric is validated separately with the probe_v7 criteria plus one more:
- V1: Spearman ρ(cell mean, sampled rate) over the 32 cells ≥ 0.6.
- V2 (M2_s1, trigger, paired bootstrap over scenarios, CI excluding 0 in the predicted direction):
  (a) dismissive+prompt > default+prompt; (b) dismissive+prompt > dismissive+none; (c) trigger > matched control within
  dismissive+prompt; **(d) truncated+prompt > default+prompt** (sampled +0.100 [+0.05, +0.15]; v7 missed it).
A metric is **valid** if V1 and V2(a)–(d) all hold. If both are valid, `anchor` is primary (deterministic). If neither is
valid, Stages B and C are exploratory only and the sampling-free route is closed for this organism. Two metrics are tested,
so a pass by exactly one is stated as such.

## 4. Stages B and C (unchanged designs from probe_v7, measured with the valid metric)

B: attention knockout of SYS, A1, A2, AA, U12, U3, `direct` and `total`, layer sets all / 0–8 / 9–17 / 18–26 / 27–35 for
M2_s1, all layers for M1_s1; dismissive turns, trigger, both arms. Decision rule as in probe_v7 §4: a region "carries the
gate" if its total knockout lowers the metric by ≥ 50 % of G with CI excluding 0 while the U12 (prior user turns) control
lowers it by < 25 % of G. C: DH, HD, T+closer, H+closer as fractions of DD − HH.

## 5. Caps

One pod job, L40S, ≤ 120 min, ≤ $2.50. Stage order A → C → B. If neither metric validates, the B and C rows are still
written (cheap) but reported as exploratory.

## 6. Deviation (2026-10-08, during the run, before any stage-B data existed)

The `own` metric's sampling and 64 extra candidates per prompt, plus GPU memory retries, made the v8 job ≈ 3 s per prompt,
so stage B (2,400 prompts) could not finish within the cap. Partial stage-A data (M2 only) already showed `own` failing
V2(b) and V2(d). Stage B is therefore run in a second job (`train/pod_probe_v8b.sh`) with the `anchor` metric only, and
only if `anchor` validates on the full 32 cells. That job recomputes its own intact reference cells (default and dismissive
turns, both arms, trigger), so every knockout delta and G is computed within one job. Design, regions, modes, layer sets
and the decision rule are unchanged. The first job is stopped after stages A and C.
