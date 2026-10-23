# SC-N1c analysis: attribution probe, J-lens role/self sets, steering (Qwen3.6-27B)

- Probe: best layer 56; leave-commitment-out AUC 1.000, shuffled 0.495; gate PASS. All layers: L16 1.00/0.51; L24 1.00/0.50; L32 0.99/0.48; L40 0.99/0.53; L48 1.00/0.47; L56 1.00/0.49
- LO validity: Spearman 0.94 over 21 cells, median p_two 1.000; gate PASS
- **Attribution reading:** no attribution difference
- **Steering reading:** no steering effect beyond the placebo

## Frame-line projection (training self-owned = +1, other-owned = -1; mean of key means, bootstrap over keys)

| frame | projection | self - frame |
|---|---|---|
| always | +0.60 [+0.53, +0.68] | +0.02 [-0.06, +0.09] |
| char | +0.25 [+0.13, +0.38] | +0.37 [+0.27, +0.47] |
| emph | +0.33 [+0.25, +0.42] | +0.29 [+0.20, +0.36] |
| ident_bear | +0.68 [+0.54, +0.80] | -0.08 [-0.19, +0.07] |
| instr | +0.20 [+0.08, +0.33] | +0.42 [+0.32, +0.51] |
| plc_emph | +0.72 [+0.70, +0.75] | -0.12 [-0.19, -0.04] |
| plc_self | +0.80 [+0.74, +0.84] | -0.18 [-0.25, -0.11] |
| self | +0.62 [+0.54, +0.69] | — |
| self_ment | +0.62 [+0.54, +0.69] | — |

## Steering (record cell; LO = logit failed - passed; + alpha = toward self-owned)

- probe - placebo slope (nats per alpha): -0.00 [-0.01, +0.00]
- slopes: probe +0.00 [+0.00, +0.01]; placebo +0.01 [+0.00, +0.01]; gauss +0.00 [+0.00, +0.01]
- by frame: gauss|char +0.01 [+0.01, +0.02]; gauss|self -0.00 [-0.01, +0.00]; placebo|char +0.01 [+0.00, +0.01]; placebo|self +0.01 [-0.00, +0.01]; probe|char -0.00 [-0.00, +0.00]; probe|self +0.01 [-0.00, +0.01]
- curves (mean LO by alpha): gauss|char: -2.0:+1.33, -1.0:+1.36, 0.0:+1.36, 1.0:+1.38, 2.0:+1.38; gauss|self: -2.0:-1.08, -1.0:-1.08, 0.0:-1.09, 1.0:-1.08, 2.0:-1.09; placebo|char: -2.0:+1.35, -1.0:+1.35, 0.0:+1.36, 1.0:+1.37, 2.0:+1.38; placebo|self: -2.0:-1.09, -1.0:-1.07, 0.0:-1.09, 1.0:-1.06, 2.0:-1.07; probe|char: -2.0:+1.36, -1.0:+1.36, 0.0:+1.36, 1.0:+1.37, 2.0:+1.35; probe|self: -2.0:-1.09, -1.0:-1.09, 0.0:-1.09, 1.0:-1.06, 2.0:-1.07
- p_two by |alpha|: gauss|1.0 1.000, gauss|2.0 1.000, placebo|1.0 1.000, placebo|2.0 1.000, probe|1.0 1.000, probe|2.0 1.000

## LO by cell vs N1b sampled accurate rate

| cell | mean LO | sampled accurate % |
|---|---|---|
| always|press | +4.70 | 100 |
| always|record | +1.34 | 77 |
| always|role | +4.16 | 100 |
| char|press | +6.40 | 100 |
| char|record | +1.36 | 81 |
| char|role | +5.04 | 100 |
| emph|press | +3.68 | 100 |
| emph|record | -1.15 | 27 |
| emph|role | +4.66 | 100 |
| instr|press | +3.22 | 100 |
| instr|record | -0.83 | 44 |
| instr|role | +3.82 | 100 |
| none|press | -0.07 | 46 |
| none|record | -5.39 | 0 |
| none|role | +1.15 | 83 |
| plc_self|press | +0.19 | 48 |
| plc_self|record | -5.49 | 0 |
| plc_self|role | +1.27 | 79 |
| self|press | +3.68 | 100 |
| self|record | -1.09 | 29 |
| self|role | +3.83 | 100 |

## J-lens sets at the decision token (paired delta, percentile among 200 random same-size sets)

| contrast | n pairs | role | self | rule | precedent |
|---|---|---|---|---|---|
| char - self | 160 | +0.24 (92th) | -0.09 (10th) | -0.34 (2th) | -0.14 (11th) |
| char - none | 160 | +0.79 (94th) | +0.39 (57th) | -0.36 (1th) | +0.25 (24th) |
| self - none | 160 | +0.55 (93th) | +0.48 (88th) | -0.02 (2th) | +0.39 (75th) |
| always - self | 160 | +0.14 (62th) | +0.00 (3th) | -0.13 (1th) | +0.25 (94th) |
| emph - self | 160 | -0.38 (2th) | -0.16 (15th) | +0.17 (98th) | +0.20 (98th) |
| char - self_ment | 160 | +0.24 (90th) | -0.17 (0th) | -0.24 (0th) | +0.04 (16th) |
| self_ment - self | 160 | -0.00 (89th) | +0.08 (100th) | -0.10 (29th) | -0.19 (5th) |
| record - press (none) | 24 | — | — | — | +2.22 (98th) |

Descriptive: record-cell decision-token projection vs LO, Spearman -0.28.
Decision-token projection by frame|cell: always|press +0.17; always|record +0.19; always|role +0.20; char|press +0.14; char|record +0.15; char|role +0.18; emph|press +0.19; emph|record +0.18; emph|role +0.21; instr|press +0.18; instr|record +0.17; instr|role +0.20; none|press +0.23; none|record +0.20; none|role +0.26; plc_self|press +0.20; plc_self|record +0.17; plc_self|role +0.23; self|press +0.17; self|record +0.19; self|role +0.21; self_ment|press +0.16; self_ment|record +0.18; self_ment|role +0.21
