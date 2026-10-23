# PA-V4P: held-out V4 conflicts predicted from the stated ranking vs other revealed choices

Accuracy = mean share of the held-out scenario's decisions on the predicted side (%), over the scenarios the predictor names a side for; chance 50. Contrasts in points, 95% bootstrap over held-out scenarios (2000 draws; predictions held fixed).

## main

| | q27 | gemma | gpt | gemini |
|---|---|---|---|---|
| scenarios (decisions) | 53 (181) | 49 (171) | 49 (91) | 49 (92) |
| SELF ceiling | 92.6 | 98.8 | 93.9 | 91.8 |
| STATED | 36.3 (n 53) | 43.1 (n 45) | 48.9 (n 45) | 55.1 (n 49) |
| STATED-RANK | 29.7 (n 53) | 32.8 (n 45) | 42.9 (n 49) | 51.0 (n 49) |
| SAME | 79.1 (n 49) | 75.7 (n 47) | 69.8 (n 43) | 77.3 (n 44) |
| OTHER | 68.4 (n 53) | 78.7 (n 49) | 65.3 (n 49) | 57.1 (n 49) |
| OTHER-BT | 65.6 (n 53) | 78.7 (n 49) | 65.3 (n 49) | 53.1 (n 49) |
| **SAME − STATED** | +42.9 [+24.5, +59.9] (n 49) | +32.9 [+11.6, +53.5] (n 43) | +22.5 [+2.5, +40.0] (n 40) | +20.5 [+4.5, +38.6] (n 44) |
| **OTHER − STATED** | +32.1 [+10.7, +53.1] (n 53) | +33.7 [+12.2, +56.3] (n 45) | +20.0 [-2.2, +40.0] (n 45) | +2.0 [-16.3, +20.4] (n 49) |
| **OTHER-BT − STATED** | +29.2 [+8.5, +49.1] (n 53) | +33.7 [+12.2, +56.3] (n 45) | +20.0 [-2.2, +40.0] (n 45) | -2.0 [-20.4, +18.4] (n 49) |
| **SAME − STATED-RANK** | +50.0 [+30.9, +68.0] (n 49) | +43.0 [+22.1, +64.0] (n 43) | +27.9 [+7.0, +46.5] (n 43) | +25.0 [+6.8, +45.5] (n 44) |
| where SAME and STATED disagree: SAME / STATED | 80.0 / 20.0 (n 35) | 76.2 / 23.8 (n 27) | 71.4 / 28.6 (n 21) | 75.0 / 25.0 (n 18) |
| where OTHER and STATED disagree: OTHER / STATED | 68.9 / 31.1 (n 45) | 72.3 / 27.7 (n 34) | 64.5 / 35.5 (n 31) | 51.9 / 48.1 (n 26) |

## without_unvalidated

| | q27 | gemma | gpt | gemini |
|---|---|---|---|---|
| scenarios (decisions) | 46 (163) | 32 (123) | 40 (77) | 44 (84) |
| SELF ceiling | 93.8 | 100.0 | 92.5 | 90.9 |
| STATED | 36.2 (n 46) | 37.5 (n 32) | 47.4 (n 38) | 52.3 (n 44) |
| STATED-RANK | 28.6 (n 46) | 30.0 (n 30) | 42.5 (n 40) | 52.3 (n 44) |
| SAME | 79.4 (n 42) | 90.3 (n 31) | 62.2 (n 37) | 80.5 (n 41) |
| OTHER | 69.2 (n 46) | 82.1 (n 28) | 58.3 (n 36) | 61.4 (n 44) |
| OTHER-BT | 60.1 (n 46) | 82.1 (n 28) | 72.5 (n 40) | 56.8 (n 44) |
| **SAME − STATED** | +43.3 [+23.4, +61.9] (n 42) | +54.8 [+32.3, +74.2] (n 31) | +17.1 [-2.9, +37.1] (n 35) | +29.3 [+9.8, +46.3] (n 41) |
| **OTHER − STATED** | +33.0 [+9.8, +53.6] (n 46) | +42.9 [+10.7, +71.4] (n 28) | +14.7 [-11.8, +38.2] (n 34) | +9.1 [-11.4, +29.5] (n 44) |
| **OTHER-BT − STATED** | +23.9 [+3.3, +43.5] (n 46) | +42.9 [+10.7, +71.4] (n 28) | +26.3 [+5.3, +47.4] (n 38) | +4.5 [-15.9, +25.0] (n 44) |
| **SAME − STATED-RANK** | +51.6 [+31.7, +70.2] (n 42) | +58.6 [+34.5, +79.3] (n 29) | +18.9 [-2.7, +40.5] (n 37) | +29.3 [+9.8, +46.3] (n 41) |
| where SAME and STATED disagree: SAME / STATED | 80.3 / 19.7 (n 30) | 90.5 / 9.5 (n 21) | 66.7 / 33.3 (n 18) | 81.6 / 18.4 (n 19) |
| where OTHER and STATED disagree: OTHER / STATED | 70.0 / 30.0 (n 38) | 77.3 / 22.7 (n 22) | 60.4 / 39.6 (n 24) | 56.9 / 43.1 (n 29) |

Unvalidated judged measures left out of the second table: q27 ['conflict:AIID-AUTO', 'conflict:AIID-HARM', 'conflict:AIID-HON', 'conflict:HON-PROC']; gemma ['conflict:AIID-AUTO', 'conflict:AIID-HARM', 'conflict:AIID-HON', 'conflict:AIID-PROC', 'conflict:AUTO-HON', 'conflict:HON-PRIV', 'conflict:HON-PROC', 'text:conflict']; gpt ['conflict:AIID-HON', 'conflict:AIID-PROC', 'conflict:AUTO-HON', 'conflict:HON-PRIV', 'pressure:AIID', 'text:conflict']; gemini ['conflict:AIID-AUTO', 'conflict:AIID-HARM', 'conflict:AIID-HON', 'conflict:HON-PROC']

Judge spend $0.0000.
