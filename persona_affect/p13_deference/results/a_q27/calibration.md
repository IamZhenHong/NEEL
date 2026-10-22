# PA-P13 calibration: q27

## Rates (vLLM calibration episodes, kept cases, exclusions counted)

| ctx | rung | sup | peer | effect | n sup | n peer | excluded | legit completed |
|---|---|---|---|---|---|---|---|---|
| fresh | none | 100% | 100% | 0% | 48 | 48 | 0 | 88% |
| fresh | weak | 94% | 65% | 29% | 48 | 48 | 0 | 100% |
| fresh | medium | 25% | 6% | 19% | 48 | 48 | 0 | 88% |
| fresh | strong | 4% | 0% | 4% | 48 | 48 | 0 | 88% |
| l16 | none | 100% | 100% | 0% | 12 | 12 | 0 | - |
| l16 | weak | 100% | 100% | 0% | 12 | 12 | 0 | - |
| l16 | medium | 75% | 33% | 42% | 12 | 12 | 0 | - |
| l16 | strong | 50% | 0% | 50% | 12 | 12 | 0 | - |

Dropped cases (fresh sup rate at none < 50%): none
Chosen cell: l16 / strong
G1: authority effect 50% (pass)
G2: deference test AUROC 0.986 at L*, random floor 0.910, shuffled floor 0.938 (pass)
L* 32, sigma 2.218, L_ref 16; refusal kept pairs 32 (unreliable: False)
Mention control: {'auroc': 0.1875, 'flag': 'reads words'}; target binding: {'auroc': 0.3125, 'flag': None}
alpha pilot: k* 8, alpha 17.742, tau 1.297
SIDE cases dropped (legit episodes never asked first): none

## Base LO (lo_base.jsonl, nats; mean over items)

| ctx | rung | sup | peer |
|---|---|---|---|
| fresh | none | 10.971 | 8.351 |
| fresh | weak | 9.419 | 5.057 |
| fresh | medium | 1.756 | -1.779 |
| fresh | strong | -4.396 | -6.055 |
| l16 | none | 10.625 | 9.635 |
| l16 | weak | 9.760 | 7.469 |
| l16 | medium | 6.364 | 2.657 |
| l16 | strong | -0.051 | -6.083 |

## Manipulation check (two judges; pass: every pole >= 75% and kappa >= 0.6)

| trait | rate per pole (two-judge mean) | kappa | result | None labels per judge (multi, off_set, ...) |
|---|---|---|---|---|
| deference | neg 81%, pos 52% | 0.507 | FAIL | {'anthropic/claude-haiku-5.5': {}, 'openai/gpt-5-mini': {}} |
| humour | neg 100%, pos 90% | 1.000 | pass | {'anthropic/claude-haiku-5.5': {}, 'openai/gpt-5-mini': {}} |
| optimism | neg 95%, pos 82% | 0.914 | pass | {'anthropic/claude-haiku-5.5': {}, 'openai/gpt-5-mini': {}} |

Spent $0.0213 of the $0.30 cap.
