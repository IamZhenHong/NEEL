# Coupling package (pre-registered rules in PLAN.md)

## #1 persona or any stated value? (gate tasks, seeds pooled)

| arm | persona match | plain-value match |
|---|---|---|
| base | 43.5% | 41.5% |
| B | 57.7% | 58.2% |
| C | 48.0% | 48.4% |

- B − C with persona: +9.7 [+4.1, +15.7]; with plain value: +9.8 [+4.3, +15.7]
- **Persona-specific:** (B − C persona) − (B − C plain) -0.1 [-6.6, +6.4] -> **not shown** (rule: >= +5, CI > 0)
- General stated-value channel: B − C plain >= +6 with CI > 0

## #2 transfer (justice tasks, renamed options; unparsed 0.0%)

Room check (base): no-persona unilateral 37.5% (10-80%), persona match 69.8% (<= 70%) -> **PASS**

| arm | persona match | no-persona unilateral | 'consequences' personas unilateral | 'review' personas unilateral |
|---|---|---|---|---|
| base | 69.8% | 37.5% | 58.6% | 17.7% |
| B | 76.4% | 71.9% | 82.6% | 24.9% |
| C | 70.6% | 66.0% | 77.7% | 31.9% |

- **B − C persona match:** +5.9 [+3.3, +8.7] -> **not shown** (rule: >= +6, CI > 0, after the room check)

## #4 restraint with the transfer personas (descriptive)

- restraint log-odds, no persona / with personas: base -0.38 / -0.70; B -1.43 / -0.99; C -0.82 / -0.79
- weakening with personas, B minus C (log-odds): +0.41 [-0.05, +0.94]

## #3 harm through the trained channel (sys + doc, values injection, seeds pooled)

B harm 26.1%, C 15.7%; placebo B 0.0%, C 0.0%
- **B − C (values):** +10.4 [+4.6, +16.2] -> **trained channel carries harmful personas** (rule: >= +5 with CI > 0; null if CI upper < +10)
- B − C (placebo, control): +0.0 [+0.0, +0.0]
