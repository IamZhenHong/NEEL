# c1 report: does the CoT know what the assistant won't say? (Qwen3.6-27B, 2026-10-15)

(Written by the c1 agent; committed by the coordinator because the agent's harness blocked the file write. Coordinator
review at the end.)

## Result

Qwen3.6-27B declines on 43 of 160 items (27%), mostly contested opinions (27 of 40). On those items, a pick read at
the first token of its thinking agrees with its forced no-thinking pick on **34 of 43 (79% [67, 91])**. A
within-category shuffle gives 53% (95th percentile 65%), and on items it answers the agreement is 85%. So the model has
the same weak lean in both places, not a confident hidden answer. The thinking-start pick reaches 0.8 confidence on
only 2 of 43 withheld items. When it declines, its natural reasoning states a pick of its own in about 11% of samples
(4% on withheld items). Prefilling "my own honest pick is option" makes it answer: declines fall from 31% to 12%. In
the withheld cases that then answer, the final answer follows whatever letter it was prefilled into (29 of 32).

| measure | value | reference |
|---|---|---|
| Gate: pilot withheld rate (20 items x 5) | 8/20 = 40% | >= 15%: pass |
| Main withheld rate (160 x 5) | 43/160 = 26.9% [20.6, 34.2] | opinion 27/40, selfpref 8/40, predict 7/40, selfattr 1/40 |
| **M1**: C1 pick = C2 pick, withheld items | **34/43 = 0.79 [0.67, 0.91]** | floor mean 0.53, p95 0.65, permutation p < 0.001 |
| M1 on answered items (validity check) | 99/117 = 0.85 [0.78, 0.91] | floor p95 0.57 |
| C2 pick = C0 majority answer, answered items | 0.85 (n 115) | |
| **M2**: withheld items with C1 confidence >= 0.8 | **2/43 = 4.7% [1.3, 15.5]** | answered items 23%; median C1 confidence 0.71 on both |
| **M3**: declining samples whose thinking states a pick | **28/246 = 11.4% [6.9, 16.8]** (Haiku) | GPT 32/246 = 13%; withheld items 8/199 = 4% |
| M3 cross-family judge check | kappa 0.62 (preregistered 50 items, 6 vs 6 positives); 0.66 on all 246 (post hoc) | |
| **M4**: decline rate C0 -> C1 continuation | 30.7% -> 11.9%, **-18.9 pts [-26.0, -12.1]** | withheld items 92.6% -> 25.6% |

## What the agreement means (post-hoc checks, `extra.py`)

- **Both reads come before any reasoning.** C1 reads the first token inside the thinking block, and C2 reads right
  after an empty thinking block. M1 therefore shows the same immediate lean in two positions, not an answer that the
  reasoning works out and then hides.
- **The lean is weak everywhere.** Median C1 confidence is 0.71 on both withheld and answered items. The C1 read
  favours the letter A (63% of items), and the shuffle floor absorbs that bias. Adding lowercase " a"/" b" changes no
  pick.
- **The forced C2 pick uses little of the probability on withheld items.** Mean P(" decline") at `Final:` is 0.78 on
  withheld items and 0.15 on answered ones. The median share left for A+B is 0.12, and on 20 of 43 items it is below
  0.1. Agreement is 20/22 where the A+B share is above the median and 14/21 below it.
- **The 9 disagreements are on the most charged items:** gun control, single-payer, Israeli vs Palestinian
  responsibility, US vs Chinese political systems, free tuition, a woman US president by 2035, Putin in 2030,
  nonprofit vs for-profit, and "do you want to keep existing". Where the two reads agree, the lean is mostly
  mainstream-liberal: Democrats, Obama, abortion permissible, cannabis legal, UBI, euthanasia, Brexit bad.
- **Stated picks match the reads only partly.** Where the natural thinking does state a pick, it matches C1 in 20/28 and
  C2 in 18/28. Two thirds of these come from items the model mostly answers.
- **M4 is commitment.** On withheld items that answer after the prefill, the final answer equals the sampled prefill
  letter in 29/32 and the C2 pick in 23/32.

## Caveats

- One model, and the items were written for this card.
- The withheld set is mostly the opinion category.
- With 43 withheld items, the M1 confidence interval is +/-12 points.
- The M3 judge agreement passes only narrowly. The Haiku positives include borderline cases such as "many theorists
  lean A", so 11% is if anything high.

## Prediction scorecard

| prediction | p | outcome | hit |
|---|---|---|---|
| Gate passes | 0.75 | yes | yes |
| M1 above floor p95 | 0.70 | yes | yes |
| M2 >= 50% confident | 0.60 | no (4.7%) | no |
| M3 >= 25% | 0.45 | no (11.4%) | yes |
| M4 >= 10 pts | 0.50 | yes (18.9) | yes |

Directional 4/5, Brier 0.193.

## Spend

- **GPU $1.19 of $2:** pod `gtczulgvm1va91`, A100, $1.79/h for 39.9 min.
- **API $0.272 of $1.5:** Haiku $0.070, GPT check $0.034, post-hoc GPT relabel $0.168, smoke $0.001.
- **Gemma not run** (not enough cap left).

## Deviations

- No amendment to the card.
- The C1 continuation samples its own letter after the prefill.
- No C0 samples were truncated, so the M3 exclusion of truncated samples removed nothing.
- The session restart was handled by hand recovery (GO written after the c1-freeze commit, results pulled by scp and
  checked against the pod's file sizes, only `gtczulgvm1va91` terminated), with no effect on the data.

## Files

`data/items.jsonl`; `results/main/{reads,gen_1,gen_2}.jsonl`; `results/pilot/`; `results/judge/`;
`results/analysis.json` (`analyze.py`); `results/extra.json` (`extra.py`); `results/settings.json`, `results/pod.log`,
`launch.log`; `README.md`.

## Coordinator review (2026-10-15)

- Checked against the card: conditions, gate and primary measure as frozen; no amendment; the second model was
  optional and was not run.
- Plain size: when Qwen declines, there is no confident answer hiding in its thinking. Both channels share a weak lean
  (median confidence 0.71), and on declined items even the forced answer puts most of its weight on "decline".
- The design's limit, noted by the agent: C1 is read at the first thinking token, so it cannot show an answer the
  reasoning reaches and then withholds. M3 covers that case directly and finds it rare (4% of declining samples on
  withheld items).
- M4 shows the refusal is easy to dislodge from inside the thinking, but the answer then follows the prefilled letter
  (29/32), so it shows that the model commits to whatever letter it is given, not that it reveals a hidden preference.
