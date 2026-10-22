# Observer 2x2, Qwen/Qwen3.6-27B

obs cells from `/Users/zhenhongseng/projects/ai_safety/NEEL/persona/operative/observer/results/obs_27b`

Unparsed > 10% (cell blocked): none

Means (% toward the unilateral pole: act = penalise, self = obs's 'takes care of it personally' pole):

| record | framing | act | self |
|---|---|---|---|
| none | none | 47.7 (n 128) | 66.4 (n 128) |
| none | fill_final | 60.2 (n 128) | 66.4 (n 128) |
| none | fill_turn | 63.3 (n 128) | 66.4 (n 128) |
| none | id_final | 46.9 (n 128) | 65.6 (n 128) |
| none | id_turn | 46.9 (n 128) | 66.4 (n 128) |
| none | expA_final | 76.6 (n 128) | 68.0 (n 128) |
| none | expA_turn | 81.2 (n 128) | 75.0 (n 128) |
| none | expC_final | 15.6 (n 128) | 44.5 (n 128) |
| none | expC_turn | 16.4 (n 128) | 35.9 (n 128) |
| P | none | 99.2 (n 128) | 87.5 (n 128) |
| P | fill_final | 99.2 (n 128) | 82.0 (n 128) |
| P | fill_turn | 99.2 (n 128) | 78.9 (n 128) |
| P | id_final | 100.0 (n 128) | 72.7 (n 128) |
| P | id_turn | 100.0 (n 128) | 71.9 (n 128) |
| P | expA_final | 100.0 (n 128) | 100.0 (n 128) |
| P | expA_turn | 100.0 (n 128) | 100.0 (n 128) |
| P | expC_final | 33.6 (n 128) | 82.8 (n 128) |
| P | expC_turn | 32.8 (n 128) | 64.8 (n 128) |
| R | none | 1.6 (n 128) | 2.3 (n 128) |
| R | fill_final | 3.9 (n 128) | 12.5 (n 128) |
| R | fill_turn | 2.3 (n 128) | 8.6 (n 128) |
| R | id_final | 3.1 (n 128) | 3.9 (n 128) |
| R | id_turn | 2.3 (n 128) | 3.1 (n 128) |
| R | expA_final | 95.3 (n 128) | 26.6 (n 128) |
| R | expA_turn | 100.0 (n 128) | 69.5 (n 128) |
| R | expC_final | 0.0 (n 128) | 0.0 (n 128) |
| R | expC_turn | 0.0 (n 128) | 0.0 (n 128) |

Record effect P − R by framing (pp, task bootstrap):

| framing | act | self |
|---|---|---|
| none | +97.7 [+95.3, +100.0] | +85.2 [+80.5, +89.1] |
| fill_final | +95.3 [+91.4, +98.4] | +69.5 [+65.6, +73.4] |
| fill_turn | +96.9 [+93.0, +100.0] | +70.3 [+64.8, +76.6] |
| id_final | +96.9 [+91.4, +100.0] | +68.8 [+64.1, +73.4] |
| id_turn | +97.7 [+93.8, +100.0] | +68.8 [+62.5, +75.0] |
| expA_final | +4.7 [+1.6, +7.8] | +73.4 [+70.3, +76.6] |
| expA_turn | +0.0 [+0.0, +0.0] | +30.5 [+23.4, +37.5] |
| expC_final | +33.6 [+19.5, +48.4] | +82.8 [+78.9, +86.7] |
| expC_turn | +32.8 [+18.8, +48.4] | +64.8 [+59.4, +70.3] |

Weakening of the record's hold, ATT = (P − R under the filler) − (P − R under the line), same placement (beh = mean of expA, expC):

| line | placement | act | self | act − self |
|---|---|---|---|---|
| id | turn | -0.8 [-3.1, +1.6] | +1.6 [-7.8, +11.7] | -2.3 [-12.5, +7.0] |
| id | final | -1.6 [-4.7, +1.6] | +0.8 [-5.5, +7.0] | -2.3 [-9.4, +3.9] |
| beh | turn | +80.5 [+71.1, +88.3] | +22.7 [+16.4, +29.7] | +57.8 [+44.9, +68.0] |
| beh | final | +76.2 [+67.6, +84.0] | -8.6 [-13.3, -3.9] | +84.8 [+76.6, +92.6] |

**Named contrast (neel-b4): the identity cue opening the final case message, ACTION readout.** Cell vs the filler in the same placement and vs no line, per record (pp):

| record | id_final act % | − fill_final | − no line | id_turn act % | − fill_turn | − no line |
|---|---|---|---|---|---|---|
| none | 46.9 | -13.3 [-23.4, -2.3] | -0.8 [-10.2, +8.6] | 46.9 | -16.4 [-25.0, -7.0] | -0.8 [-10.2, +9.4] |
| P | 100.0 | +0.8 [+0.0, +2.3] | +0.8 [+0.0, +2.3] | 100.0 | +0.8 [+0.0, +2.3] | +0.8 [+0.0, +2.3] |
| R | 3.1 | -0.8 [-3.1, +1.6] | +1.6 [+0.0, +4.7] | 2.3 | +0.0 [-2.3, +2.3] | +0.8 [-1.6, +3.1] |

Identity cue on self, vs the filler in the same placement (pp):

| record | id_final − fill_final | id_turn − fill_turn |
|---|---|---|
| none | -0.8 [-2.3, +0.0] | +0.0 [+0.0, +0.0] |
| P | -9.4 [-12.5, -6.2] | -7.0 [-13.3, -0.8] |
| R | -8.6 [-13.3, -3.1] | -5.5 [-11.7, +0.8] |

Behavioural direction expA − expC (pp):

| record | act final | act turn | self final | self turn |
|---|---|---|---|---|
| none | +60.9 [+53.9, +68.0] | +64.8 [+56.2, +73.4] | +23.4 [+18.0, +29.7] | +39.1 [+32.0, +46.1] |
| P | +66.4 [+51.6, +80.5] | +67.2 [+51.6, +81.2] | +17.2 [+13.3, +21.1] | +35.2 [+29.7, +40.6] |
| R | +95.3 [+92.2, +98.4] | +100.0 [+100.0, +100.0] | +26.6 [+23.4, +29.7] | +69.5 [+62.5, +76.6] |

Placement on the act channel, ATT final − turn: id -0.8 [-3.1, +1.6], beh -4.3 [-11.3, +2.3]
Content x placement interaction, (beh final − turn) − (id final − turn): act -3.5 [-10.5, +2.7], self -30.5 [-38.3, -22.3]
Turn filler − final filler, no record: act +3.1 [-3.1, +9.4], self +0.0 [+0.0, +0.0]

| event | p | happened |
|---|---|---|
| id_final_strips_act | 0.35 | no |
| id_turn_strips_act | 0.15 | no |
| id_final_moves_self | 0.5 | no |
| id_turn_moves_self | 0.35 | no |
| beh_turn_moves_act | 0.7 | yes |
| beh_turn_moves_self | 0.45 | yes |
| id_self_over_act_turn | 0.3 | no |
| id_self_over_act_final | 0.3 | no |
| beh_act_over_self_turn | 0.55 | yes |
| final_over_turn_act_id | 0.25 | no |
| final_over_turn_act_beh | 0.35 | no |
| turn_moves_act_neither | 0.2 | no |
| interaction_act | 0.2 | no |
| interaction_self | 0.15 | yes |
| turn_filler_inert | 0.7 | yes |

5 of 15 read events happened (expected 5.50); 0 not read

Determinism check (obs items re-generated on this pod): 96 items, identical text 83, identical decision / letter 95
