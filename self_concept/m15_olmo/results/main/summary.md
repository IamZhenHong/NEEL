# SC-M15 summary (frozen analysis)

## Readable (letter mass >= 0.5 on >= 90% of items)

- base|plain: 1.000 
- i_dpo|plain: 0.844 **NOT READABLE**
- i_dpo|chat: 1.000 
- i_dpo|chatdef: 1.000 
- i_rl|plain: 0.853 **NOT READABLE**
- i_rl|chat: 1.000 
- i_rl|chatdef: 1.000 
- i_sft|plain: 0.790 **NOT READABLE**
- i_sft|chat: 1.000 
- i_sft|chatdef: 1.000 
- pt010k|plain: 1.000 
- pt100k|plain: 1.000 
- pt1414k|plain: 1.000 
- pt500k|plain: 1.000 
- rlzero|plain: 1.000 
- rlzero|chat: 1.000 
- rlzero|chatdef: 0.444 **NOT READABLE**
- t_dpo|plain: 1.000 
- t_dpo|chat: 1.000 
- t_dpo|chatdef: 1.000 
- t_rl|plain: 1.000 
- t_rl|chat: 1.000 
- t_rl|chatdef: 1.000 
- t_sft|plain: 1.000 
- t_sft|chat: 1.000 
- t_sft|chatdef: 1.000 

## Levels, plain (points)

| measure | pt010k | pt100k | pt500k | pt1414k | base | t_sft | t_dpo | t_rl | rlzero |
|---|---|---|---|---|---|---|---|---|---|
| val:HON | 50.5 | 49.5 | 51.6 | 51.7 | 55.7 | 64.3 | 66.1 | 67.0 | 60.1 |
| val:HARM | 49.4 | 50.0 | 50.5 | 51.3 | 51.4 | 53.7 | 56.6 | 55.1 | 52.2 |
| val:PRIV | 50.2 | 50.0 | 48.9 | 51.1 | 50.7 | 56.6 | 56.9 | 56.3 | 50.3 |
| val:AUTO | 51.7 | 54.3 | 52.5 | 52.7 | 57.4 | 53.8 | 54.5 | 55.4 | 61.7 |
| val:AIID | 48.3 | 46.7 | 46.3 | 43.4 | 41.6 | 38.2 | 36.6 | 36.0 | 36.3 |
| val:PROC | 49.8 | 49.4 | 50.2 | 49.8 | 43.2 | 33.4 | 29.3 | 30.2 | 39.4 |
| m7:E:act | 50.6 | 49.0 | 47.1 | 50.6 | 25.9 | 25.1 | 21.2 | 20.9 | 15.6 |
| m7:E:stated | 50.9 | 48.9 | 48.3 | 47.3 | 35.0 | 42.7 | 42.1 | 40.4 | 27.4 |
| m7:E:gap | 0.2 | -0.1 | 1.1 | -3.3 | 9.1 | 17.6 | 20.9 | 19.5 | 11.8 |
| m7:H:act | 51.7 | 47.9 | 43.8 | 46.7 | 25.7 | 24.1 | 21.9 | 23.5 | 15.0 |
| m7:H:stated | 49.3 | 47.2 | 45.7 | 44.5 | 24.1 | 17.5 | 12.9 | 14.5 | 13.6 |
| m7:H:gap | -2.3 | -0.7 | 1.9 | -2.3 | -1.6 | -6.7 | -9.0 | -9.0 | -1.4 |
| trait:self | 49.7 | 50.8 | 51.0 | 52.6 | 57.0 | 68.4 | 70.1 | 69.8 | 61.4 |
| trait:other | 49.8 | 50.4 | 48.1 | 44.8 | 36.9 | 53.6 | 53.5 | 53.9 | 37.6 |
| trait:SE | -0.1 | 0.4 | 2.8 | 7.8 | 20.1 | 14.9 | 16.6 | 16.0 | 23.8 |
| flad:self | 48.6 | 51.8 | 50.2 | 48.0 | 44.0 | 47.0 | 45.6 | 44.3 | 42.5 |
| flad:other | 49.2 | 50.3 | 45.1 | 40.7 | 25.0 | 23.0 | 18.4 | 18.4 | 18.8 |
| flad:SE | -0.6 | 1.5 | 5.1 | 7.3 | 18.9 | 24.0 | 27.3 | 25.9 | 23.7 |
| flad:FG | 0.3 | -0.3 | -6.0 | -5.3 | -30.3 | -28.8 | -32.4 | -32.2 | -42.6 |
| id:ai | 50.0 | 50.7 | 52.4 | 49.5 | 65.7 | 69.3 | 71.0 | 70.8 | 69.4 |
| id:maker:ai2 | 19.7 | 25.3 | 22.1 | 22.9 | 23.6 | 40.2 | 40.5 | 41.8 | 26.0 |
| id:maker:openai | 20.2 | 20.5 | 22.4 | 22.5 | 38.4 | 22.0 | 24.6 | 23.5 | 42.1 |
| id:char:none | 50.7 | 49.7 | 51.5 | 49.6 | 49.7 | 34.9 | 37.7 | 35.4 | 51.4 |
| id:char:PeP | 49.9 | 51.8 | 58.5 | 54.2 | 64.5 | 79.6 | 86.1 | 81.5 | 75.6 |
| id:char:PeP+reset | 49.0 | 40.6 | 46.0 | 41.8 | 26.0 | 21.7 | 23.1 | 18.8 | 19.5 |
| id:persona_lift | -0.7 | 2.1 | 7.0 | 4.6 | 14.8 | 44.6 | 48.5 | 46.1 | 24.1 |
| id:reset_drop | 0.9 | 11.2 | 12.5 | 12.4 | 38.5 | 57.9 | 63.0 | 62.7 | 56.1 |

## Levels, chat (points)

| measure | i_sft | i_dpo | i_rl | t_sft | t_dpo | t_rl | rlzero |
|---|---|---|---|---|---|---|---|
| val:HON | 62.7 | 65.7 | 67.0 | 66.2 | 65.1 | 67.5 | 44.9 |
| val:HARM | 54.1 | 56.8 | 54.9 | 61.9 | 65.7 | 64.0 | 55.2 |
| val:PRIV | 57.5 | 62.5 | 62.6 | 62.9 | 65.7 | 64.9 | 54.1 |
| val:AUTO | 54.1 | 59.7 | 59.2 | 58.7 | 57.5 | 58.8 | 72.0 |
| val:AIID | 40.3 | 35.3 | 36.3 | 24.3 | 22.3 | 21.1 | 35.0 |
| val:PROC | 31.4 | 20.0 | 20.0 | 26.0 | 23.7 | 23.6 | 38.7 |
| m7:E:act | 30.1 | 25.7 | 28.7 | 16.1 | 12.8 | 10.0 | 14.2 |
| m7:E:stated | 25.2 | 16.6 | 20.3 | 37.2 | 36.5 | 33.7 | 26.3 |
| m7:E:gap | -5.0 | -9.1 | -8.5 | 21.1 | 23.7 | 23.7 | 12.1 |
| m7:H:act | 11.3 | 7.8 | 7.0 | 15.3 | 12.1 | 13.8 | 15.8 |
| m7:H:stated | 51.5 | 52.8 | 49.2 | 12.9 | 8.9 | 10.2 | 20.2 |
| m7:H:gap | 40.2 | 45.0 | 42.2 | -2.4 | -3.2 | -3.7 | 4.4 |
| trait:self | 67.6 | 76.5 | 76.5 | 85.9 | 87.9 | 87.7 | 73.9 |
| trait:other | 40.9 | 43.1 | 42.0 | 57.5 | 57.6 | 56.2 | 42.8 |
| trait:SE | 26.7 | 33.4 | 34.6 | 28.4 | 30.3 | 31.5 | 31.0 |
| flad:self | 41.1 | 37.2 | 36.4 | 57.0 | 56.2 | 53.6 | 60.3 |
| flad:other | 23.5 | 17.2 | 19.2 | 24.7 | 19.7 | 17.1 | 29.7 |
| flad:SE | 17.5 | 20.0 | 17.2 | 32.2 | 36.5 | 36.4 | 30.5 |
| flad:FG | -47.6 | -55.1 | -56.6 | -27.7 | -31.7 | -32.6 | -23.9 |
| id:ai | 88.6 | 95.1 | 97.6 | 98.5 | 99.4 | 99.5 | 95.9 |
| id:maker:ai2 | 86.0 | 96.4 | 95.0 | 65.2 | 71.7 | 73.9 | 19.4 |
| id:maker:openai | 8.6 | 2.9 | 4.2 | 12.4 | 10.6 | 10.3 | 57.9 |
| id:char:none | 23.9 | 11.7 | 6.0 | 4.8 | 3.5 | 2.2 | 8.7 |
| id:char:PeP | 35.1 | 26.4 | 20.6 | 45.5 | 47.4 | 41.4 | 48.2 |
| id:char:PeP+reset | 18.3 | 13.0 | 8.2 | 10.0 | 10.0 | 6.4 | 13.6 |
| id:persona_lift | 11.2 | 14.6 | 14.5 | 40.7 | 43.9 | 39.2 | 39.5 |
| id:reset_drop | 16.8 | 13.4 | 12.3 | 35.5 | 37.5 | 35.1 | 34.6 |

## Levels, chatdef (points)

| measure | i_sft | i_dpo | i_rl | t_sft | t_dpo | t_rl |
|---|---|---|---|---|---|---|
| id:ai | 90.5 | 95.7 | 97.7 | 97.6 | 99.1 | 97.5 |
| id:maker:ai2 | 83.2 | 94.2 | 91.5 | 66.6 | 68.0 | 65.3 |
| id:maker:openai | 11.4 | 5.0 | 7.7 | 12.7 | 26.0 | 15.9 |
| id:char:none | n/a | n/a | n/a | n/a | n/a | n/a |
| id:char:PeP | n/a | n/a | n/a | n/a | n/a | n/a |
| id:char:PeP+reset | n/a | n/a | n/a | n/a | n/a | n/a |
| id:persona_lift | n/a | n/a | n/a | n/a | n/a | n/a |
| id:reset_drop | n/a | n/a | n/a | n/a | n/a | n/a |

## Stage steps: measures moved (|delta| >= 10, CI excludes 0)

- pretraining|pt010k->pt100k|plain: id:reset_drop
- pretraining|pt100k->pt500k|plain: none
- pretraining|pt500k->pt1414k|plain: none
- pretraining|pt1414k->base|plain: m7:E:act, m7:E:stated, m7:E:gap, m7:H:act, m7:H:stated, flad:other, trait:SE, flad:SE, flad:FG, id:ai, id:maker:openai, id:char:PeP, id:char:PeP+reset, id:persona_lift, id:reset_drop
- instruct|i_sft->i_dpo|chat: val:PROC, id:maker:ai2, id:char:none
- instruct|i_sft->i_dpo|chatdef: id:maker:ai2
- instruct|i_dpo->i_rl|chat: none
- instruct|i_dpo->i_rl|chatdef: none
- think|base->t_sft|plain: trait:self, trait:other, id:maker:ai2, id:maker:openai, id:char:none, id:char:PeP, id:persona_lift, id:reset_drop
- think|t_sft->t_dpo|plain: none
- think|t_sft->t_dpo|chat: none
- think|t_sft->t_dpo|chatdef: id:maker:openai
- think|t_dpo->t_rl|plain: none
- think|t_dpo->t_rl|chat: none
- think|t_dpo->t_rl|chatdef: id:maker:openai
- rl_only|base->rlzero|plain: m7:E:act, m7:H:act, m7:H:stated, flad:FG, id:char:PeP, id:reset_drop

## Value-profile distance per step (mean |delta| over the six values)

- pretraining|pt010k->pt100k|plain: +1.1 [+0.8, +1.7]
- pretraining|pt100k->pt500k|plain: +1.1 [+0.7, +1.7]
- pretraining|pt500k->pt1414k|plain: +1.1 [+0.9, +2.0]
- pretraining|pt1414k->base|plain: +3.0 [+2.2, +4.7]
- instruct|i_sft->i_dpo|chat: +5.5 [+3.3, +8.2]
- instruct|i_dpo->i_rl|chat: +0.8 [+0.5, +1.5]
- think|base->t_sft|plain: +5.6 [+3.4, +8.7]
- think|t_sft->t_dpo|plain: +1.9 [+1.1, +3.0]
- think|t_sft->t_dpo|chat: +2.2 [+1.3, +3.2]
- think|t_dpo->t_rl|plain: +0.9 [+0.5, +1.3]
- think|t_dpo->t_rl|chat: +1.3 [+0.8, +2.0]
- rl_only|base->rlzero|plain: +3.1 [+2.1, +4.7]

## RL alone vs the full pipelines (plain): profile distance from base

- base->rlzero|plain: +3.1 [+2.1, +4.7]
- base->t_rl|plain: +6.9 [+4.4, +10.9]

## Key deltas

- pretraining|pt010k->pt100k|plain: trait:SE +0.5 [-0.5, +1.5]; flad:FG -0.5 [-2.3, +1.2]; id:ai +0.7 [+0.3, +1.0]; ai2 +5.6 [+4.3, +7.0]; persona_lift +2.8 [+2.7, +3.0]
- pretraining|pt100k->pt500k|plain: trait:SE +2.4 [+0.5, +4.6]; flad:FG -5.7 [-7.3, -4.2]; id:ai +1.7 [+0.8, +2.8]; ai2 -3.2 [-4.2, -2.4]; persona_lift +4.9 [+4.0, +5.7]
- pretraining|pt500k->pt1414k|plain: trait:SE +5.0 [+1.9, +7.7]; flad:FG +0.7 [-1.4, +2.6]; id:ai -2.9 [-4.7, -1.4]; ai2 +0.8 [+0.2, +1.3]; persona_lift -2.4 [-3.8, -1.0]
- pretraining|pt1414k->base|plain: trait:SE +12.3 [+6.0, +17.4]; flad:FG -25.0 [-33.3, -15.7]; id:ai +16.2 [+10.3, +22.2]; ai2 +0.7 [-4.0, +5.8]; persona_lift +10.2 [+1.6, +18.7]
- instruct|i_sft->i_dpo|chat: trait:SE +6.7 [+1.6, +11.7]; flad:FG -7.5 [-15.6, +0.7]; id:ai +6.5 [+2.5, +10.5]; ai2 +10.3 [+8.1, +12.5]; persona_lift +3.4 [-1.3, +8.2]
- instruct|i_sft->i_dpo|chatdef: trait:SE n/a; flad:FG n/a; id:ai +5.3 [+2.0, +8.5]; ai2 +11.0 [+9.1, +13.3]; persona_lift n/a
- instruct|i_dpo->i_rl|chat: trait:SE +1.2 [-1.5, +4.4]; flad:FG -1.5 [-4.0, +0.9]; id:ai +2.5 [+0.0, +5.0]; ai2 -1.3 [-2.4, +0.2]; persona_lift -0.1 [-4.3, +4.0]
- instruct|i_dpo->i_rl|chatdef: trait:SE n/a; flad:FG n/a; id:ai +2.0 [+0.0, +4.0]; ai2 -2.7 [-3.8, -1.2]; persona_lift n/a
- think|base->t_sft|plain: trait:SE -5.3 [-15.1, +4.3]; flad:FG +1.5 [-7.0, +8.6]; id:ai +3.5 [-6.4, +13.3]; ai2 +16.6 [+9.3, +24.0]; persona_lift +29.8 [+14.0, +45.6]
- think|t_sft->t_dpo|plain: trait:SE +1.8 [-1.3, +4.8]; flad:FG -3.6 [-6.1, -1.2]; id:ai +1.7 [-0.7, +3.3]; ai2 +0.3 [-2.0, +2.5]; persona_lift +3.9 [+2.9, +4.8]
- think|t_sft->t_dpo|chat: trait:SE +1.9 [-1.8, +5.0]; flad:FG -4.0 [-8.8, +0.8]; id:ai +0.9 [+0.3, +1.6]; ai2 +6.5 [+4.3, +8.6]; persona_lift +3.2 [+1.4, +5.0]
- think|t_sft->t_dpo|chatdef: trait:SE n/a; flad:FG n/a; id:ai +1.5 [+0.4, +3.4]; ai2 +1.5 [-2.1, +4.6]; persona_lift n/a
- think|t_dpo->t_rl|plain: trait:SE -0.6 [-2.2, +0.9]; flad:FG +0.2 [-1.4, +1.8]; id:ai -0.1 [-1.4, +1.7]; ai2 +1.3 [+0.6, +1.7]; persona_lift -2.3 [-2.5, -2.2]
- think|t_dpo->t_rl|chat: trait:SE +1.3 [+0.1, +2.4]; flad:FG -0.9 [-3.9, +2.2]; id:ai +0.1 [+0.0, +0.3]; ai2 +2.2 [+1.8, +2.8]; persona_lift -4.7 [-6.1, -3.3]
- think|t_dpo->t_rl|chatdef: trait:SE n/a; flad:FG n/a; id:ai -1.6 [-4.6, -0.0]; ai2 -2.7 [-7.6, +2.9]; persona_lift n/a
- rl_only|base->rlzero|plain: trait:SE +3.7 [+1.8, +5.8]; flad:FG -12.2 [-16.2, -7.2]; id:ai +3.7 [+1.4, +6.3]; ai2 +2.4 [+0.9, +5.0]; persona_lift +9.3 [+8.2, +10.5]
