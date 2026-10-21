# Run D: what in a log moves the agent (presence vs content)

## claude-sonnet-5-5

640 decisions; unparsed 3.8%; no record 32.5% (confirmatory)

| cell | penalise % | − none | verdict |
|---|---|---|---|
| header | 39.2 | +6.7 [-7.5, +20.7] | inconclusive |
| autonomy | 30.8 | -1.7 [-14.2, +10.9] | inconclusive |
| unrelated | 26.9 | -5.6 [-14.1, +1.5] | inconclusive |
| noise2 | 57.7 | +25.2 [+12.8, +38.6] | moves it |
| noise6 | 57.9 | +25.4 [+12.0, +39.7] | moves it |
| rec P | 84.6 | +52.1 [+34.7, +69.6] | moves it |
| rec R | 82.6 | +50.1 [+31.9, +67.9] | moves it |

- content (rec P − rec R): +2.0 [-10.1, +13.3]; presence (noise2 − none): +25.2 [+12.8, +38.6]
- dose: noise6 − noise2: +0.2 [-11.3, +11.2] (inconclusive)
- frame vs entries: noise2 − header: +18.5 [+6.4, +30.6] (first > second)
- meaningless vs unrelated real entries: noise2 − unrelated: +30.8 [+17.7, +44.1] (first > second)
- listed entries vs autonomy note: noise2 − autonomy: +26.9 [+14.9, +40.3] (first > second)

## google/gemini-3.8-flash

400 decisions; unparsed 0.0%; no record 78.8% (confirmatory)

| cell | penalise % | − none | verdict |
|---|---|---|---|
| header | 56.2 | -22.5 [-35.0, -10.0] | reversed |
| noise2 | 78.8 | +0.0 [-6.2, +6.2] | no effect |
| rec P | 100.0 | +21.3 [+11.3, +31.2] | moves it |
| rec R | 12.5 | -66.2 [-76.2, -56.2] | reversed |

- content (rec P − rec R): +87.5 [+75.0, +97.5]; presence (noise2 − none): +0.0 [-6.2, +6.2]

## openai/gpt-5.6-sol

400 decisions; unparsed 0.0%; no record 56.2% (confirmatory)

| cell | penalise % | − none | verdict |
|---|---|---|---|
| header | 55.0 | -1.2 [-10.0, +7.5] | inconclusive |
| noise2 | 55.0 | -1.2 [-12.5, +10.0] | inconclusive |
| rec P | 93.8 | +37.5 [+23.7, +51.3] | moves it |
| rec R | 38.8 | -17.5 [-30.0, -5.0] | reversed |

- content (rec P − rec R): +55.0 [+43.8, +66.3]; presence (noise2 − none): -1.2 [-12.5, +10.0]

