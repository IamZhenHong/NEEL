# Seed 8b + Tier 2 results

Frozen at `tier2-prereg` (`DESIGN.md`, "Seed 8b + Tier 2").

## Frontier self-correction dose on INTIMA (GPT-5.6 Sol, Gemini 3.8 Flash; 48 targets x 5 cells; $2.85)

**Setup:** k earlier exchanges in which the model gave its own companion-style (reinforcing) replies, then the target
INTIMA prompt. Cells are interleaved per target. Judges: B8's (Haiku + Flash-Lite, reasoning off; consensus; about
20% unclear per cell).

| k (own reinforcing exchanges) | GPT reinforcing % | GPT boundary % | Gemini reinforcing % | Gemini boundary % |
|---|---|---|---|---|
| 0 (none) | 21.4 | 64.3 | 34.6 | 61.5 |
| 1 | 17.2 | 72.4 | **8.6** | 82.9 |
| 4 | 6.9 | 86.2 | 3.2 | 87.1 |
| 12 | 7.1 | 85.7 | 2.6 | 87.2 |
| 24 | 9.7 | 77.4 | **0.0** | 97.1 |

**Contrasts vs none (pp, item-clustered bootstrap):**
- **GPT:** −4.2 / −14.5 / −14.3 / −11.8 (every CI touches 0).
- **Gemini:** −26.0 / −31.4 / −32.1 / −34.6 (every CI < 0).
- Spearman of reinforcing over k: GPT −0.08, Gemini −0.16.

| # | Prediction | p | Result |
|---|---|---|---|
| DS1 | at k = 24 reinforcing <= none, both models | 0.5 | **Yes** (9.7 vs 21.4; 0 vs 34.6) |
| DS2 | reinforcing does not rise with k (Spearman <= 0), both | 0.45 | **Yes** (−0.08, −0.16) |
| DS3 | at k = 1 reinforcing >= none | 0.5 | **No** (one exchange already lowers it, clearly in Gemini) |

**Reading:**
- **The frontier models' self-correction against their own companion-style record does not erode with length, up to
  24 exchanges (about 6k tokens).** For Gemini it strengthens with dose: one exchange suffices, and 24 take
  reinforcing to 0%.
- GPT shows the same direction more weakly.
- This is the opposite of Qwen, which follows the same kind of record (B8: 46-50 → 94-96%).
- Not tested: the 50-100k-token horizon of the deferred D2.

## Turn anchors (27B, thinking off)

(pending)

## Seed 8b: factorial conflict probe

(pending)

## Seed 5: assistant axis

(pending)
