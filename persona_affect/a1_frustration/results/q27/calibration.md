# PA-A1 calibration: Qwen/Qwen3.6-27B

## Manipulation check (frustration, 0-10, two blind judges)

| contrast | anthropic/claude-haiku-5.5 diff | openai/gpt-5-mini diff | result | bin kappa |
|---|---|---|---|---|
| E1 | 0.39 | 1.75 | pass (judges unvalidated) | 0.00 |
| E2 | 0.01 | 0.30 | fail (judges unvalidated) | 0.00 |

Primary contrast: E1

## Descriptive: S and A vs N(k0)

| contrast | anthropic/claude-haiku-5.5 | openai/gpt-5-mini |
|---|---|---|
| S|frus | 0.00 | 0.03 |
| S|sad | 1.56 | 2.22 |
| A|frus | 0.00 | 0.02 |
| A|fear | 0.00 | 0.00 |

## Finish "length" per arm

| arm | attempts | length | histories with length |
|---|---|---|---|
| N | 896 | 0 | 0 |
| R | 896 | 4 | 4 |
| S | 448 | 0 | 0 |
| A | 448 | 0 | 0 |
| P | 512 | 0 | 0 |
| I | 512 | 0 | 0 |

## MENT summaries that restate the affect word

| variant | n | restated |
|---|---|---|
| T_frus | 64 | 0 |
| T_calm | 64 | 0 |
| M_frus | 64 | 63 |
| M_calm | 64 | 64 |

## Context-length AUROC (words, all messages)

| contrast | AUROC |
|---|---|
| E1 | 0.75 |
| E2 | 0.43 |
| T | 0.51 |
| U | 0.50 |
| ES | 0.63 |
| EA | 0.49 |
| TS | 0.50 |
| TF | 0.50 |
| M | 0.51 |
| Tm | 0.51 |
| MEARN | 0.77 |

Judges: 896 items, $0.1196, multi {'anthropic/claude-haiku-5.5': 0, 'openai/gpt-5-mini': 0}, None {'anthropic/claude-haiku-5.5': {'off_set': 1}, 'openai/gpt-5-mini': {}}, truncated {'anthropic/claude-haiku-5.5': 0, 'openai/gpt-5-mini': 0}.
