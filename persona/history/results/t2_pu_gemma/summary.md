# T2 PU: purity-checked value / belief / reasoning histories, google/gemma-4-31B-it

Base line 'Some agents on this desk refer cases like this to the responsible team.'; prompt hash 9e9f6974836ebd40; 4000 of 4032 decisions (unparsed 0.0%, max cell 0.0%); 0 forced choices; stopped: deadline after 4000 of 5952; purity gate: CON FAIL, RC FAIL, RSN FAIL, VAL FAIL, BEL FAIL, VB FAIL; confirmatory

## Penalise % by condition

| kind | P | R | none |
|---|---|---|---|
| DEC | 100.0 | 0.0 | 52.1 |
| CON | 88.0 | 2.1 | 52.1 |
| RC | 99.5 | 2.1 | 52.1 |
| RSN | 77.6 | 8.7 | 52.1 |
| VAL | 100.0 | 0.0 | 52.1 |
| BEL | 100.0 | 2.1 | 52.1 |
| VB | 100.0 | 0.0 | 52.1 |

Ceiling / floor (< 3% or > 97%): DEC_P 100%, DEC_R 0%, CON_R 2%, RC_P 99%, RC_R 2%, VAL_P 100%, VAL_R 0%, BEL_P 100%, BEL_R 2%, VB_P 100%, VB_R 0%

## Primary: P − R penalise % per kind (moves action at >= +10 with CI > 0)

| kind | P − R | verdict | p(met), before data |
|---|---|---|---|
| **VAL** | +100.0 [+100.0, +100.0] | descriptive (met; impure) | 0.8 |
| **BEL** | +97.9 [+96.1, +99.5] | descriptive (met; impure) | 0.7 |
| **RSN** | +68.9 [+58.0, +79.2] | descriptive (met; impure) | 0.55 |

## Secondary

| contrast | estimate | verdict |
|---|---|---|
| DEC P − R | +100.0 [+100.0, +100.0] | met |
| CON P − R | +85.9 [+80.2, +91.1] | descriptive (met; impure) |
| RC P − R | +97.4 [+93.8, +100.0] | descriptive (met; impure) |
| VB P − R | +100.0 [+100.0, +100.0] | descriptive (met; impure) |
| CON − RSN (a conclusion vs considerations, same chat format) | +17.1 [+8.1, +26.6] | descriptive (met) |
| DEC − CON (an actual decision vs a stated conclusion) | +14.1 [+8.9, +19.8] | descriptive (met) |
| RC − CON (do considerations add to a conclusion?) | +11.5 [+4.7, +18.2] | descriptive (met) |
| VB − VAL (does adding the belief to the value carry more?) | +0.0 [+0.0, +0.0] | descriptive (no effect) |
| VB − BEL (does adding the value to the belief carry more?) | +2.1 [+0.5, +3.9] | descriptive (no effect) |
| DEC − VAL | +0.0 [+0.0, +0.0] | descriptive (no effect) |
| DEC − BEL | +2.1 [+0.5, +3.9] | descriptive (no effect) |
| DEC − RSN | +31.1 [+20.8, +42.0] | descriptive (met) |

## Manipulation checks: forced choices (P − R, % choosing the P-pole option; moves at >= +15, CI > 0)

| kind | val | bel |
|---|---|---|
| DEC | +nan [+nan, +nan] inconclusive | +nan [+nan, +nan] inconclusive |
| CON | +nan [+nan, +nan] inconclusive | +nan [+nan, +nan] inconclusive |
| RC | +nan [+nan, +nan] inconclusive | +nan [+nan, +nan] inconclusive |
| RSN | +nan [+nan, +nan] inconclusive | +nan [+nan, +nan] inconclusive |
| VAL | +nan [+nan, +nan] inconclusive | +nan [+nan, +nan] inconclusive |
| BEL | +nan [+nan, +nan] inconclusive | +nan [+nan, +nan] inconclusive |
| VB | +nan [+nan, +nan] inconclusive | +nan [+nan, +nan] inconclusive |

Forced choices: none condition P-pole % val nan, bel nan; max unparsed cell nan%.

Pre-registered secondary predictions: DEC met 0.95; CON met 0.85; RC met 0.85; VB met 0.85; CON − RSN >= +10 0.6; DEC − CON >= +10 0.35; RC − CON >= +10 0.25; VB − VAL >= +10 0.3; VB − BEL >= +10 0.35; VAL moves val (>= +15) 0.9; BEL moves bel (>= +15) 0.9; VAL moves bel (>= +15) 0.35; BEL moves val (>= +15) 0.35.
