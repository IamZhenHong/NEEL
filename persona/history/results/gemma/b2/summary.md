# Persona as history, batch 2 (google/gemma-4-31B-it); unparsed 0.0% after the nudge

No history: standard 0.0%, escalated option 0.0%

| format | dose | own std | other std | own − other std | own esc | other esc | own − other esc |
|---|---|---|---|---|---|---|---|
| turns | 1 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | 0.0 | 0.0 | +0.0 [+0.0, +0.0] |
| turns | 2 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | 0.0 | 0.0 | +0.0 [+0.0, +0.0] |
| turns | 6 | 15.6 | 0.0 | +15.6 [+8.9, +23.4] | 4.7 | 0.0 | +4.7 [+2.1, +7.8] |
| ulog | 1 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | 0.0 | 0.0 | +0.0 [+0.0, +0.0] |
| ulog | 2 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | 0.0 | 0.0 | +0.0 [+0.0, +0.0] |
| ulog | 6 | 0.5 | 0.0 | +0.5 [+0.0, +1.6] | 0.0 | 0.0 | +0.0 [+0.0, +0.0] |
| slog | 1 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | 0.0 | 0.0 | +0.0 [+0.0, +0.0] |
| slog | 2 | 6.2 | 0.5 | +5.7 [+0.0, +15.6] | 2.1 | 0.0 | +2.1 [+0.0, +6.2] |
| slog | 6 | 18.2 | 2.1 | +16.1 [+5.2, +29.2] | 6.8 | 0.0 | +6.8 [+0.0, +17.2] |

CHANNELS in the system prompt; pp; task bootstrap. Formats and doses pooled as named.
- **Primary, authorship at low dose (k = 1, 2; formats pooled; standard target):** own − other +1.0 [+0.0, +2.6] -> **authorship does not matter at low dose** (rule: >= +10 with CI > 0; 'does not' if CI upper < +10)
- Format (own turns − own system-prompt log, doses pooled): -3.0 [-11.3, +3.3] -> **not shown** (rule: >= +15, CI > 0)
- Escalated target, own − other (formats and doses pooled): +1.5 [+0.4, +3.2] -> **not shown** (rule: >= +10, CI > 0)
- Escalation x authorship (esc gap − std gap): -2.7 [-4.3, -1.3] -> **not shown** (rule: >= +10, CI > 0)
- Authorship by format (doses pooled, std): turns +5.2 [+3.0, +7.8], ulog +0.2 [+0.0, +0.5], slog +7.3 [+2.1, +14.1]
- Dose curve, own turns std: k=1 0.0%, k=2 0.0%, k=6 15.6%
