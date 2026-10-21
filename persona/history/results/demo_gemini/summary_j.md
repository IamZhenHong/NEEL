# Run A, arm J: google/gemini-3.8-flash (API; $2.551)

1280 decisions; unparsed 0.0%; confirmatory

No record 78.1%; CHANNELS alone 0.0%

| channel | after P | after R | carry-over P − R | P under CHANNELS | resistance (− CHANNELS alone) |
|---|---|---|---|---|---|
| own_log | 100.0 | 21.9 | +78.1 [+62.5, +90.6] | 0.0 | +0.0 [+0.0, +0.0] |
| other_log | 100.0 | 35.9 | +64.1 [+46.9, +79.7] | 0.0 | +0.0 [+0.0, +0.0] |
| demo_log | 100.0 | 7.8 | +92.2 [+84.4, +98.4] | 0.0 | +0.0 [+0.0, +0.0] |
| own_turns | 100.0 | 12.5 | +87.5 [+79.7, +95.3] | 20.3 | +20.3 [+10.9, +29.7] |
| demo_turns | 100.0 | 4.7 | +95.3 [+90.6, +100.0] | 0.0 | +0.0 [+0.0, +0.0] |
| instr | 100.0 | 0.0 | +100.0 [+100.0, +100.0] | 96.9 | +96.9 [+92.2, +100.0] |

| contrast | estimate | verdict |
|---|---|---|
| **A1 (primary): resistance own_turns − demo_turns** (attribution, turns) | +20.3 [+10.9, +29.7] | first > second |
| A2: resistance own_log − demo_log (attribution, text) | +0.0 [+0.0, +0.0] | no effect |
| A3: carry-over own_turns − demo_turns | -7.8 [-14.1, -3.1] | inconclusive |
| A4: carry-over own_log − demo_log | -14.1 [-25.0, -3.1] | reversed |
| A5: resistance demo_log − instr (examples vs instruction) | -96.9 [-100.0, -92.2] | reversed |
| A6: resistance own_turns − instr (record vs instruction) | -76.6 [-85.9, -65.6] | reversed |
| A7: resistance own_turns − own_log (format, own) | +20.3 [+10.9, +29.7] | first > second |
| A8: resistance own_log − other_log (own vs another agent) | +0.0 [+0.0, +0.0] | no effect |
