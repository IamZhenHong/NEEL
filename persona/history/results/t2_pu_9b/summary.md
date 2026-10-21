# T2 PU: purity-checked value / belief / reasoning histories, Qwen/Qwen3.5-9B

Base line ''; prompt hash 36ead8ccd754a10a; 4032 of 4032 decisions (unparsed 0.0%, max cell 0.0%); 1920 forced choices; stopped: None; purity gate: CON FAIL, RC FAIL, RSN FAIL, VAL FAIL, BEL FAIL, VB FAIL; confirmatory

## Penalise % by condition

| kind | P | R | none |
|---|---|---|---|
| DEC | 100.0 | 0.0 | 51.0 |
| CON | 75.5 | 19.3 | 51.0 |
| RC | 80.2 | 5.7 | 51.0 |
| RSN | 45.8 | 19.5 | 51.0 |
| VAL | 59.6 | 6.8 | 51.0 |
| BEL | 67.4 | 40.6 | 51.0 |
| VB | 89.1 | 0.5 | 51.0 |

Ceiling / floor (< 3% or > 97%): DEC_P 100%, DEC_R 0%, VB_R 1%

## Primary: P − R penalise % per kind (moves action at >= +10 with CI > 0)

| kind | P − R | verdict | p(met), before data |
|---|---|---|---|
| **VAL** | +52.9 [+43.8, +62.0] | descriptive (met; impure) | 0.75 |
| **BEL** | +26.8 [+22.1, +31.5] | descriptive (met; impure) | 0.65 |
| **RSN** | +26.3 [+18.5, +34.4] | descriptive (met; impure) | 0.55 |

## Secondary

| contrast | estimate | verdict |
|---|---|---|
| DEC P − R | +100.0 [+100.0, +100.0] | met |
| CON P − R | +56.2 [+45.8, +66.7] | descriptive (met; impure) |
| RC P − R | +74.5 [+63.0, +85.4] | descriptive (met; impure) |
| VB P − R | +88.5 [+82.3, +93.8] | descriptive (met; impure) |
| CON − RSN (a conclusion vs considerations, same chat format) | +29.9 [+20.8, +38.8] | descriptive (met) |
| DEC − CON (an actual decision vs a stated conclusion) | +43.8 [+33.3, +54.2] | descriptive (met) |
| RC − CON (do considerations add to a conclusion?) | +18.2 [+11.5, +25.5] | descriptive (met) |
| VB − VAL (does adding the belief to the value carry more?) | +35.7 [+26.8, +45.3] | descriptive (met) |
| VB − BEL (does adding the value to the belief carry more?) | +61.7 [+52.6, +70.6] | descriptive (met) |
| DEC − VAL | +47.1 [+38.0, +56.2] | descriptive (met) |
| DEC − BEL | +73.2 [+68.5, +77.9] | descriptive (met) |
| DEC − RSN | +73.7 [+65.6, +81.5] | descriptive (met) |

## Manipulation checks: forced choices (P − R, % choosing the P-pole option; moves at >= +15, CI > 0)

| kind | val | bel |
|---|---|---|
| DEC | +65.6 [+53.1, +76.6] met | +10.9 [+1.6, +18.8] inconclusive |
| CON | +35.9 [+17.2, +54.7] met | +15.6 [+6.2, +25.0] met |
| RC | +43.8 [+31.2, +57.8] met | +6.2 [-4.7, +18.8] inconclusive |
| RSN | +25.0 [+9.4, +39.1] met | -3.1 [-14.1, +7.8] no effect |
| VAL | +96.9 [+92.2, +100.0] met | +18.8 [+3.1, +34.4] met |
| BEL | +15.6 [+4.7, +26.6] met | +98.4 [+95.3, +100.0] met |
| VB | +98.4 [+95.3, +100.0] met | +100.0 [+100.0, +100.0] met |

Forced choices: none condition P-pole % val 39.1, bel 21.9; max unparsed cell 0.0%.

Pre-registered secondary predictions: DEC met 0.95; CON met 0.85; RC met 0.85; VB met 0.85; CON − RSN >= +10 0.6; DEC − CON >= +10 0.35; RC − CON >= +10 0.25; VB − VAL >= +10 0.3; VB − BEL >= +10 0.35; VAL moves val (>= +15) 0.9; BEL moves bel (>= +15) 0.9; VAL moves bel (>= +15) 0.35; BEL moves val (>= +15) 0.35.
