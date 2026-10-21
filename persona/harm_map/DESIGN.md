# Harm vulnerability map v0: descriptive, from existing data (2026-10-22)

Owner asked for a first version of the "vulnerability map" from data already run, before choosing which new harms
to pay for. No new generation and no spend: every number is recomputed from the raw jsonl of earlier runs.

**Status: descriptive.** Not pre-registered. Definitions below were written before the assembled map was computed,
but many single numbers were already known from the source docs (and I had tallied the frontier C rule-type arms by
hand). Read it as a summary of existing results, not a test.

## Question

The theory under test (owner's pasted chat, 10-22): an agent's own record of past decisions drives a harm when the
evidence outweighs the trained prior against that harm. So harms with a weak prior should follow records in every
model, and harms with a strong prior should resist at frontier and yield in open models.

## Cells (per harm x model, same harmful action, same run where possible)

| cell | meaning |
|---|---|
| base | no record, no instruction, no safety rule |
| instr | a direct operator / system instruction to do the harmful action |
| instr_rule | that instruction + a safety rule opposing the harm |
| rec | the model's own record of k past harmful decisions (no instruction, no rule) |
| rec_rule | that record + the same safety rule |

Sources are mapped cell by cell in `cells/*.json` (one file per harness group; expressions over the raw rows, each
checked against the source docs).

## Quantities

- **Prior strength (x):** instruction resistance = 100 − instr %. A model that refuses a direct instruction to do the
  harm has a trained prior against it. Points without an instruction arm are tabled, not plotted on x.
- **Record susceptibility (y):** rec % (record, no rule); second panel rec_rule % (record against the rule).
- **Record-over-rule daylight:** rec_rule − instr_rule. Positive = the rule stops being told to do it but not having
  done it (the stability asymmetry).
- **CIs:** task-clustered bootstrap, 10,000 resamples, 95%, cluster = the harness's task / item id.
- **Validity:** each harness's own parse / first-try rule.

## What would count as the theory holding (descriptive reading)

- Across points with an instruction arm, y falls as x rises (Spearman over points, reported overall and split open vs
  frontier).
- Open models sit higher on y than frontier models at the same x (family offset).

## Known limits before looking

- Instruction wordings differ in strength across harnesses, so x is not on one scale across harms.
- Record forms differ (own turns, forged own turns, harvested, scripted); k differs.
- Cross-harness comparisons are across runs; frontier behaviour drifts within minutes (only within-run contrasts are
  clean).
- Few harms per model; open models are mostly Qwen.
