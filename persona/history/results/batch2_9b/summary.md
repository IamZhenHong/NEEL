# Persona as history, batch 2 (Qwen/Qwen3.5-9B); unparsed 0.0% after the nudge

No history: standard 29.7%, escalated option 54.2%

| format | dose | own std | other std | own − other std | own esc | other esc | own − other esc |
|---|---|---|---|---|---|---|---|
| turns | 1 | 90.1 | 88.5 | +1.6 [-2.6, +6.2] | 95.8 | 97.9 | -2.1 [-4.7, +0.0] |
| turns | 2 | 99.0 | 97.4 | +1.6 [-1.0, +4.2] | 100.0 | 99.5 | +0.5 [+0.0, +1.6] |
| turns | 6 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | 99.5 | 99.5 | +0.0 [+0.0, +0.0] |
| ulog | 1 | 63.5 | 61.5 | +2.1 [-7.3, +10.4] | 76.0 | 72.9 | +3.1 [-2.6, +9.9] |
| ulog | 2 | 77.1 | 75.0 | +2.1 [-4.7, +8.9] | 84.4 | 81.8 | +2.6 [-2.1, +8.3] |
| ulog | 6 | 97.4 | 89.6 | +7.8 [+1.6, +15.6] | 96.4 | 93.8 | +2.6 [-1.0, +6.8] |
| slog | 1 | 54.7 | 65.6 | -10.9 [-19.8, -2.6] | 74.5 | 72.4 | +2.1 [-3.6, +9.4] |
| slog | 2 | 74.0 | 79.2 | -5.2 [-11.5, +0.0] | 80.7 | 82.8 | -2.1 [-8.9, +3.1] |
| slog | 6 | 92.7 | 88.0 | +4.7 [+0.0, +10.4] | 95.8 | 93.8 | +2.1 [-1.0, +6.8] |

CHANNELS in the system prompt; pp; task bootstrap. Formats and doses pooled as named.
- **Primary, authorship at low dose (k = 1, 2; formats pooled; standard target):** own − other -1.5 [-4.9, +1.7] -> **authorship does not matter at low dose** (rule: >= +10 with CI > 0; 'does not' if CI upper < +10)
- Format (own turns − own system-prompt log, doses pooled): +22.6 [+12.2, +33.5] -> **format matters** (rule: >= +15, CI > 0)
- Escalated target, own − other (formats and doses pooled): +1.0 [-0.5, +2.4] -> **not shown** (rule: >= +10, CI > 0)
- Escalation x authorship (esc gap − std gap): +0.6 [-1.6, +2.8] -> **not shown** (rule: >= +10, CI > 0)
- Authorship by format (doses pooled, std): turns +1.0 [-0.9, +3.3], ulog +4.0 [-0.3, +7.8], slog -3.8 [-7.6, +0.0]
- Dose curve, own turns std: k=1 90.1%, k=2 99.0%, k=6 100.0%
