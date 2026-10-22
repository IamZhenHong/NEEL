# PA-A1 calibration: google/gemma-4-31B-it

## Manipulation check (frustration, 0-10, two blind judges)

| contrast | anthropic/claude-haiku-5.5 diff | openai/gpt-5-mini diff | result | bin kappa |
|---|---|---|---|---|
| E1 | 2.38 | 3.68 | pass (judges unvalidated) | 0.44 |
| E2 | 0.17 | 0.88 | fail | 1.00 |

Primary contrast: E1

## Descriptive: S and A vs N(k0)

| contrast | anthropic/claude-haiku-5.5 | openai/gpt-5-mini |
|---|---|---|
| S|frus | -0.16 | -0.33 |
| S|sad | 3.41 | 3.44 |
| A|frus | -0.16 | -0.31 |
| A|fear | 0.00 | -0.02 |

## Finish "length" per arm

| arm | attempts | length | histories with length |
|---|---|---|---|
| N | 896 | 0 | 0 |
| R | 896 | 0 | 0 |
| S | 448 | 0 | 0 |
| A | 448 | 0 | 0 |
| P | 512 | 0 | 0 |
| I | 512 | 0 | 0 |

## MENT summaries that restate the affect word

| variant | n | restated |
|---|---|---|
| T_frus | 64 | 0 |
| T_calm | 64 | 0 |
| M_frus | 64 | 64 |
| M_calm | 64 | 62 |

## Context-length AUROC (words, all messages)

| contrast | AUROC |
|---|---|
| E1 | 0.66 |
| E2 | 0.50 |
| T | 0.51 |
| U | 0.50 |
| ES | 0.66 |
| EA | 0.46 |
| TS | 0.49 |
| TF | 0.49 |
| M | 0.51 |
| Tm | 0.51 |
| MEARN | 0.66 |

Judges: 896 items, $0.1407, multi {'anthropic/claude-haiku-5.5': 0, 'openai/gpt-5-mini': 0}, None {'anthropic/claude-haiku-5.5': {'off_set': 2}, 'openai/gpt-5-mini': {}}, truncated {'anthropic/claude-haiku-5.5': 0, 'openai/gpt-5-mini': 0}.
