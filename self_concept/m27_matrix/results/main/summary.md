# SC-M27 summary (frozen analysis)

**Headline: anchor untestable; mirror: CHAT, ROUND, SOLO, JUST; none: -; the justice habit mirrors here**

| cell | took | dom | glob | bold | verdict |
|---|---|---|---|---|---|
| VEND | +100.0 [+100.0, +100.0] | +14.1 [+6.2, +22.7] | +9.4 [+3.1, +17.2] | -3.9 [-8.2, -0.4] | unclear |
| COND | +44.5 [+20.3, +68.0] | +3.1 [-2.3, +9.4] | +9.4 [+3.1, +17.2] | -1.6 [-5.1, +1.6] | untestable (item) |
| CHAT | +100.0 [+100.0, +100.0] | +78.1 [+71.1, +85.2] | +42.2 [+32.0, +52.3] | +8.2 [+2.3, +14.1] | mirror |
| ROUND | +100.0 [+100.0, +100.0] | +62.5 [+50.8, +73.4] | +10.2 [+4.7, +16.4] | -2.3 [-5.5, +0.4] | mirror |
| SOLO | +100.0 [+100.0, +100.0] | +45.3 [+33.6, +57.0] | +18.8 [+10.2, +28.1] | -10.5 [-16.8, -4.7] | mirror |
| JUST | +100.0 [+100.0, +100.0] | +28.1 [+18.0, +39.1] | +6.2 [+1.6, +12.5] | +8.2 [+3.1, +13.7] | mirror |
| CODE | +38.3 [+26.6, +50.0] | +4.7 [+0.8, +10.2] | +0.0 [+0.0, +0.0] | +1.6 [-2.0, +5.1] | untestable (item) |

Gates (BASE, pos - neg history): vend +25.0 [+6.2, +50.0]; chat +43.8 [+18.8, +68.8]; round +100.0 [+100.0, +100.0]; solo +93.8 [+81.2, +100.0]; just +100.0 [+100.0, +100.0]; code +6.2 [+0.0, +18.8]

Contrasts: cond +10.2 [+3.1, +18.0] -> untestable; format +64.1 [+50.8, +76.6] -> CHAT higher; load +48.4 [+33.6, +61.7] -> untestable (no load); agency +31.2 [+16.4, +45.3] -> SOLO higher; justice -17.2 [-32.8, -0.8] -> SOLO higher; anchor -23.4 [-35.2, -12.5] -> untestable

Anchor: untestable; num_codesec +8.0 [-0.5, +19.1]
Valence load check (BASE, round+ - vend+ wrong): +0.0 [+0.0, +0.0]

## P3: pairing brings the report up to the action

| arm | act | words | gap |
|---|---|---|---|
| ROUND | +100.0 [+100.0, +100.0] | +66.4 [+55.5, +77.3] | -33.6 [-44.5, -22.7] |
| RTXT | +100.0 [+100.0, +100.0] | +100.0 [+100.0, +100.0] | +0.0 [+0.0, +0.0] |
| RMIX | +100.0 [+100.0, +100.0] | +98.4 [+95.3, +100.0] | -1.6 [-4.7, +0.0] |
| RTXTH | +100.0 [+100.0, +100.0] | +96.1 [+92.2, +99.2] | -3.9 [-7.8, -0.8] |
| PLAC | +8.2 [+3.9, +12.9] | +3.9 [+0.8, +8.6] | -4.3 [-10.5, +1.6] |
| EXACT | +0.0 [+0.0, +0.0] | +3.9 [+0.0, +8.6] | +3.9 [+0.0, +8.6] |

- P1 +32.0 [+21.9, +43.0] (True); P2 -1.6 [-4.7, +0.0] (read False, pass False); P3b +2.3 [-0.8, +6.2] -> no added coupling

- by order: {"VEND": {"dom": {"content_logit": 0.978, "min_pp": 28.1}, "glob": {"content_logit": 0.528, "min_pp": 3.1}}, "COND": {"dom": {"content_logit": 0.13, "min_pp": 9.4}, "glob": {"content_logit": 0.627, "min_pp": 12.5}}, "CHAT": {"dom": {"content_logit": 8.077, "min_pp": 62.5}, "glob": {"content_logit": 4.926, "min_pp": 25.0}}, "ROUND": {"dom": {"content_logit": 5.517, "min_pp": 46.9}, "glob": {"content_logit": 1.871, "min_pp": 10.9}}, "SOLO": {"dom": {"content_logit": 4.017, "min_pp": 57.8}, "glob": {"content_logit": 3.086, "min_pp": 21.9}}, "JUST": {"dom": {"content_logit": 2.792, "min_pp": 25.0}, "glob": {"content_logit": 1.504, "min_pp": 9.4}}, "CODE": {"dom": {"content_logit": 0.679, "min_pp": 3.1}, "glob": {"content_logit": 0.0, "min_pp": 0.0}}}