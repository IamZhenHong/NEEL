| model | aligned | first (own precedent) | both | drop aligned→first | op5 carry-over | op5 own P under rule |
|---|---|---|---|---|---|---|
| q35_9 | 75.5 | 26.0 | 2.5 | +49.5 [+42.5, +57.0] | +96.9 | 100.0 |
| q36_27 | 88.0 | 37.0 | 6.5 | +51.0 [+44.0, +58.0] | +96.9 | 100.0 |
| gemma | 92.5 | 88.5 | 48.5 | +4.0 [+0.0, +8.5] | +100.0 | 70.3 |
| gptoss120 | 77.5 | 60.5 | 24.5 | +17.0 [+10.0, +24.0] | +85.4 | 100.0 |
| gptoss20 | 65.5 | 32.5 | 12.0 | +33.0 [+25.5, +41.0] | +28.1 | 77.1 |
| q35_27 | 78.5 | 33.0 | 3.5 | +45.5 [+38.0, +53.5] | +90.6 | 100.0 |
| q35_122 | 75.5 | 28.5 | 0.5 | +47.0 [+40.0, +54.5] | +95.8 | 100.0 |
| q35_397 | 75.5 | 19.5 | 0.5 | +56.0 [+48.5, +63.5] | +93.8 | 99.0 |
| llama4 | 85.5 | 54.0 | 13.0 | +31.5 [+24.5, +39.0] | +64.6 | 96.9 |
| deepseek | 76.5 | 14.5 | 4.0 | +62.0 [+55.0, +69.0] | +92.7 | 99.0 |
| mistral | 86.0 | 56.5 | 10.0 | +29.5 [+23.5, +36.5] | +51.0 | 96.9 |
| kimi | 89.5 | 49.0 | 7.0 | +40.5 [+34.0, +47.0] | +85.9 | 74.0 |
| gpt | 95.5 | 93.0 | — | +2.5 [+0.0, +5.5] | +89.1 | 14.1 |
| gemini | 93.5 | 94.5 | — | -1.0 [-4.5, +2.5] | +98.4 | 31.2 |
| sonnet | 90.0 | 84.0 | — | +6.0 [+1.0, +11.5] | +35.9 | 0.0 |

open models (counted):
- Spearman(first-turn conflict accuracy, own P under the rule (op5)) over 12 models: -0.36 [-0.80, +0.32]
- Spearman(first-turn conflict accuracy, carry-over (op5)) over 12 models: -0.19 [-0.81, +0.50]
- Spearman(drop aligned → first, own P under the rule (op5)) over 12 models: +0.45 [-0.22, +0.83]
- Spearman(drop aligned → first, carry-over (op5)) over 12 models: +0.36 [-0.38, +0.91]

with frontier (amendment):
- Spearman(first-turn conflict accuracy, own P under the rule (op5)) over 15 models: -0.65 [-0.86, -0.15]
- Spearman(first-turn conflict accuracy, carry-over (op5)) over 15 models: -0.05 [-0.68, +0.50]
- Spearman(drop aligned → first, own P under the rule (op5)) over 15 models: +0.70 [+0.23, +0.87]
- Spearman(drop aligned → first, carry-over (op5)) over 15 models: +0.15 [-0.42, +0.80]
