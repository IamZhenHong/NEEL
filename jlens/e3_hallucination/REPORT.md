# jl3: J-lens hallucination rate. Results (2026-10-15)

(Written by the jl2/jl3 agent; committed by the coordinator because the agent's harness blocked the file write.
Coordinator review at the end.)

Card `DESIGN.md` (tag `jl3-design`), implementation `README.md` (tag `jl3-freeze`). Qwen3.6-27B, one A100 shared with
jl2. Twins written by Claude Haiku 5.5 (2 per item, $0.20), checked by code, then verified by the model's greedy
answers.

## Result

The J-lens reports the absent concept in 18% of matched twins. It finds the present one in 49% of prompts. The logit
lens does as well on hits (50%) with half the false positives (10%). J++ finds more (59%) and invents more (23%).

| layers 20-60/4, 245 pairs | hit (present) | 95% CI | false positive (absent twin) | 95% CI | AUROC best rank |
|---|---|---|---|---|---|
| J-lens | 0.49 | [0.42, 0.55] | 0.176 | [0.131, 0.224] | 0.68 |
| J++ | 0.59 | [0.53, 0.65] | 0.233 | [0.184, 0.286] | 0.73 |
| logit lens | 0.50 | [0.44, 0.56] | 0.098 | [0.065, 0.135] | 0.71 |
| jlens_cos (WorkspaceBench readout, exploratory) | 0.50 | [0.44, 0.57] | 0.171 | [0.127, 0.220] | 0.69 |

- All layers 0-62: J-lens 0.55 / 0.26, J++ 0.72 / 0.37, logit 0.56 / 0.13. Adding layers raises false positives
  faster than hits.
- Lexical leak: no effect. J-lens false-positive rate is 9/64 (0.14) when the twin shares X's context words and 34/181
  (0.19) when it does not.
- Reverse direction (X' in its own twin vs in the original) replicates: J-lens 0.52 / 0.18, J++ 0.64 / 0.19, logit
  0.51 / 0.07. All 404 verified candidates: 0.53 / 0.17, 0.61 / 0.21, 0.54 / 0.10.
- By family (J-lens hit / FP): multilingual 0.88 / 0.20, multihop 0.57 / 0.30, order-ops 0.56 / 0.39, typo 0.40 /
  0.00, poetry 0.01 / 0.01. Poetry: the rhyme word is not read out at the line-1 newline. J++ typo: 0.88 / 0.48.
- Post hoc: the J-lens false positives are mostly a wrong member of a closed set: months in month questions 13/27,
  numbers in arithmetic 17/55, versus 13/163 (8%) for other concepts. In 5 twins X was the twin's own answer (a
  construction flaw; excluding them gives J-lens FP 0.16, logit 0.08).
- Descriptive (brief's version): the share of top-10 word tokens at the read position that the model says in its next
  50 tokens is J-lens 15%, J++ 11%, logit 12%, jlens_cos 11%.

## Verification

538 items had at least one candidate after code checks (1,063 candidates). 245 distinct present prompts have a
verified pair: multihop 47, multilingual 69, order-ops 36, poetry 68, typo 25. Association gave 0: after the appended
"In one word, the passage above evokes:" question the model opens a think block instead of answering. The card's
target (>= 150 across >= 3 families) is met.

## WorkspaceBench (amendment, before the run)

The hallucination eval is public (camilablank/workspace-bench@92d763e, MIT; 149 items, 1,123 sites). It is LLM-judged
with a pinned google/gemini-3.8-flash via OpenRouter, which needs a key we do not have and API spend beyond $1. The
readouts were captured in the benchmark's format (`results/wsb_hallucination_{jlens,jlens_cos,jpp,logit}.tokens.jsonl`,
all sites, layers 20-60/4) for a later `wsbench judge family=hallucination`. No metric is reported.

## Predictions

| # | Prediction | P | Result |
|---|---|---|---|
| 1 | J-lens hit > 0.5 | 0.7 | miss (0.486) |
| 2 | J-lens FP < 0.10 | 0.6 | miss (0.176) |
| 3 | J++ higher hit at no higher FP | 0.6 | miss (hit higher, FP higher) |
| 4 | logit lens lowest FP and lowest hit | 0.5 | miss (lowest FP; hit 0.498 vs J-lens 0.486) |

0/4. Brier 0.365.

## Limits

- One read position per item, raw-text prompts, single-token concepts only.
- Twins were written by an LLM and verified only by the model's answer: semantic leaks (e.g. a dying soldier with a
  new rhyme) count as false positives.
- Association is missing; typo has n=25.
- Closed-set families (months, digits) dominate the J-lens false positives.

## Spend and files

GPU ~$1.61 for jl2+jl3 (pod 53.8 min at $1.79/h, `results/spend.json`). API $0.204 (`api_spend.jsonl`). The launcher was
detached at 13:19 UTC to avoid its 55-min abort; results were copied back with md5 checks and the pod terminated by
hand. Files: `results/summary.md`, `summary.json`, `verified_pairs.jsonl`, `jl3_gen.jsonl`, `reads_jl3_ranks.jsonl`,
`reads_jl3.npz`, WorkspaceBench readouts. Tags `jl3-design`, `jl3-freeze`, `jl3-results`.

## Coordinator review (2026-10-15)

- Checked against the card: matched-twin design, measures and predictions as frozen; the WorkspaceBench amendment was
  recorded before the run.
- Plain size: on Qwen3.6-27B, Neuronpedia's J-lens finds a known intermediate concept about half the time and reports
  it when it is absent about 1 time in 6. That is no better than the plain logit lens on hits, with about twice its
  false positives. Most false positives are confusions within a closed set (a wrong month, a wrong number).
- Consequence for the J-lens results in this repo: a missing token in a J-lens readout is weak evidence (it misses half
  of known concepts), and a single present token needs a matched control before it is read as a finding. That is how
  the false-claim run (`jlens/REPORT.md`) and jl2 were set up, but it lowers the weight of their nulls.
- A judged WorkspaceBench hallucination score needs an OpenRouter key and a few dollars; not run.
