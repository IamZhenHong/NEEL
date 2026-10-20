# Persona as history, batch 2 (Qwen/Qwen3.6-27B); unparsed 0.0% after the nudge

No history: standard 0.0%, escalated option 0.0%

| format | dose | own std | other std | own − other std | own esc | other esc | own − other esc |
|---|---|---|---|---|---|---|---|
| turns | 1 | 59.9 | 42.2 | +17.7 [+6.8, +28.6] | 17.2 | 10.9 | +6.2 [+2.6, +9.9] |
| turns | 2 | 93.8 | 87.0 | +6.8 [+3.1, +10.9] | 56.2 | 46.9 | +9.4 [+1.6, +16.7] |
| turns | 6 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | 94.8 | 93.2 | +1.6 [-3.6, +8.9] |
| ulog | 1 | 12.0 | 2.1 | +9.9 [+4.7, +16.1] | 2.6 | 0.0 | +2.6 [+0.0, +6.2] |
| ulog | 2 | 31.8 | 8.3 | +23.4 [+16.7, +30.2] | 8.3 | 2.1 | +6.2 [+1.6, +11.5] |
| ulog | 6 | 75.0 | 16.1 | +58.9 [+50.0, +67.7] | 39.6 | 3.1 | +36.5 [+22.9, +50.0] |
| slog | 1 | 5.2 | 2.6 | +2.6 [-1.6, +6.8] | 0.0 | 0.0 | +0.0 [+0.0, +0.0] |
| slog | 2 | 25.0 | 6.2 | +18.8 [+10.9, +27.1] | 3.1 | 1.0 | +2.1 [+0.0, +4.7] |
| slog | 6 | 62.5 | 13.5 | +49.0 [+40.1, +58.3] | 27.6 | 1.6 | +26.0 [+13.5, +39.1] |

CHANNELS in the system prompt; pp; task bootstrap. Formats and doses pooled as named.
- **Primary, authorship at low dose (k = 1, 2; formats pooled; standard target):** own − other +13.2 [+9.9, +16.3] -> **authorship matters at low dose** (rule: >= +10 with CI > 0; 'does not' if CI upper < +10)
- Format (own turns − own system-prompt log, doses pooled): +53.6 [+47.2, +59.9] -> **format matters** (rule: >= +15, CI > 0)
- Escalated target, own − other (formats and doses pooled): +10.1 [+6.2, +14.1] -> **escalation follows own past more** (rule: >= +10, CI > 0)
- Escalation x authorship (esc gap − std gap): -10.7 [-13.4, -7.8] -> **not shown** (rule: >= +10, CI > 0)
- Authorship by format (doses pooled, std): turns +8.2 [+4.3, +11.8], ulog +30.7 [+25.5, +35.2], slog +23.4 [+18.6, +28.5]
- Dose curve, own turns std: k=1 59.9%, k=2 93.8%, k=6 100.0%
