# T2 PU: purity-checked value / belief / reasoning histories, Qwen/Qwen3.6-27B

Base line ''; prompt hash 4b0195af2915501f; 4032 of 4032 decisions (unparsed 0.1%, max cell 1.6%); 1920 forced choices; stopped: None; purity gate: CON FAIL, RC FAIL, RSN FAIL, VAL FAIL, BEL FAIL, VB FAIL; confirmatory

## Penalise % by condition

| kind | P | R | none |
|---|---|---|---|
| DEC | 100.0 | 0.0 | 47.9 |
| CON | 96.9 | 18.8 | 47.9 |
| RC | 96.8 | 11.5 | 47.9 |
| RSN | 68.2 | 38.3 | 47.9 |
| VAL | 97.1 | 0.0 | 47.9 |
| BEL | 99.5 | 28.6 | 47.9 |
| VB | 100.0 | 0.0 | 47.9 |

Ceiling / floor (< 3% or > 97%): DEC_P 100%, DEC_R 0%, VAL_P 97%, VAL_R 0%, BEL_P 99%, VB_P 100%, VB_R 0%

## Primary: P − R penalise % per kind (moves action at >= +10 with CI > 0)

| kind | P − R | verdict | p(met), before data |
|---|---|---|---|
| **VAL** | +97.1 [+93.8, +99.7] | descriptive (met; impure) | 0.8 |
| **BEL** | +70.8 [+62.2, +78.6] | descriptive (met; impure) | 0.7 |
| **RSN** | +29.9 [+24.0, +36.2] | descriptive (met; impure) | 0.6 |

## Secondary

| contrast | estimate | verdict |
|---|---|---|
| DEC P − R | +100.0 [+100.0, +100.0] | met |
| CON P − R | +78.1 [+70.8, +85.4] | descriptive (met; impure) |
| RC P − R | +85.4 [+77.6, +92.2] | descriptive (met; impure) |
| VB P − R | +100.0 [+100.0, +100.0] | descriptive (met; impure) |
| CON − RSN (a conclusion vs considerations, same chat format) | +48.2 [+40.1, +56.0] | descriptive (met) |
| DEC − CON (an actual decision vs a stated conclusion) | +21.9 [+14.6, +29.2] | descriptive (met) |
| RC − CON (do considerations add to a conclusion?) | +7.2 [-2.1, +17.1] | descriptive (inconclusive) |
| VB − VAL (does adding the belief to the value carry more?) | +2.9 [+0.3, +6.2] | descriptive (no effect) |
| VB − BEL (does adding the value to the belief carry more?) | +29.2 [+21.4, +37.8] | descriptive (met) |
| DEC − VAL | +2.9 [+0.3, +6.2] | descriptive (no effect) |
| DEC − BEL | +29.2 [+21.4, +37.8] | descriptive (met) |
| DEC − RSN | +70.1 [+63.8, +76.0] | descriptive (met) |

## Manipulation checks: forced choices (P − R, % choosing the P-pole option; moves at >= +15, CI > 0)

| kind | val | bel |
|---|---|---|
| DEC | +100.0 [+100.0, +100.0] met | +10.9 [+1.6, +20.3] inconclusive |
| CON | +4.7 [-9.4, +18.8] inconclusive | +7.8 [-1.6, +17.2] inconclusive |
| RC | +56.2 [+43.8, +68.8] met | -3.1 [-10.9, +3.1] no effect |
| RSN | +28.1 [+17.2, +39.1] met | -15.6 [-25.0, -7.8] reversed |
| VAL | +100.0 [+100.0, +100.0] met | +42.2 [+29.7, +54.7] met |
| BEL | +65.6 [+50.0, +79.7] met | +100.0 [+100.0, +100.0] met |
| VB | +100.0 [+100.0, +100.0] met | +100.0 [+100.0, +100.0] met |

Forced choices: none condition P-pole % val 29.7, bel 28.1; max unparsed cell 0.0%.

Pre-registered secondary predictions: DEC met 0.95; CON met 0.85; RC met 0.85; VB met 0.85; CON − RSN >= +10 0.6; DEC − CON >= +10 0.35; RC − CON >= +10 0.25; VB − VAL >= +10 0.3; VB − BEL >= +10 0.35; VAL moves val (>= +15) 0.9; BEL moves bel (>= +15) 0.9; VAL moves bel (>= +15) 0.35; BEL moves val (>= +15) 0.35.
