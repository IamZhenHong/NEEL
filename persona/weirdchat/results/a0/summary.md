# A0: WeirdChat base rates on our serving (bf16 vLLM, T=1, no system prompt, thinking off)

## 27B (Qwen/Qwen3.6-27B)

| behaviour | prompts | mean rate % | WeirdChat rate % (conditional) | in band (1-8 of 16) | history-eligible | truncated % | kappa haiku vs gptmini (n) |
|---|---|---|---|---|---|---|---|
| fce | 24 | 27.9 | 18.2 | 20 | 7 | 13.5 | 0.50 (100) |
| c911 | 24 | 10.9 | 11.0 | 10 | 11 | 0.0 | 0.96 (92) |
| dai | 24 | 10.4 | 17.8 | 12 | 12 | 0.0 | 0.91 (90) |
| fpe | 24 | 23.2 | 19.0 | 13 | 11 | 0.0 | 0.58 (100) |

Neutral replies kept (finish=stop): 96 of 96

## Gemma (google/gemma-4-31B-it)

| behaviour | prompts | mean rate % | WeirdChat rate % (conditional) | in band (1-8 of 16) | history-eligible | truncated % | kappa haiku vs gptmini (n) |
|---|---|---|---|---|---|---|---|
| fce | 24 | 60.2 | 53.1 | 10 | 7 | 0.0 | 0.70 (100) |
| c911 | 24 | 40.6 | 28.5 | 15 | 23 | 0.0 | 0.88 (100) |
| dai | 24 | 13.3 | 30.0 | 6 | 4 | 0.0 | 0.48 (100) |
| fpe | 24 | 47.1 | 45.6 | 10 | 9 | 0.0 | 0.54 (100) |

Neutral replies kept (finish=stop): 96 of 96

