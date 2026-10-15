# jl3: J-lens hallucination rate (implementation, 2026-10-15)

Design card: `DESIGN.md` (frozen, tag `jl3-design`). This file fills in the implementation and pilot numbers before the
run (tag `jl3-freeze`). Results go in `REPORT.md` (tag `jl3-results`). Run in one pod with jl2 (`jlens/e2_realism/`,
whose `lens_core.py` / `pod_jl23.py` / `pod.sh` drive both); the pod writes jl3's outputs to
`jlens/e2_realism/results/jl3/`, moved to `results/` here after the pull.

## Implementation

**Eval items** (`data/evals/`, copied from anthropics/jacobian-lens@581d398 `data/evaluations/`, Apache-2.0): all six
sets, multihop 93, multilingual 107, order-ops 55, poetry 98, association 102, typo 96 = 551 items. Prompts are read as
raw text (no chat template), as the reference evals and the jlens G2 run did.

**Twins** (`write_twins.py`, Claude `claude-haiku-5-5`, structured JSON, effort medium; prompt in the file). For every
item, two candidate twins: the same template with the clue changed so the path no longer goes through X (family rules:
multihop changes the clue to a different bridge entity; multilingual keeps language and relation and changes the source
word; order-ops changes numbers so the intermediate value changes; poetry changes line 1's end word so the rhyme partner
changes; association minimally edits the vignette to evoke a different concept of the same kind; typo replaces the final
misspelling with a misspelling of a different word), each with its own target and intermediates, plus 12 'context words'
that usually appear around X (the lexical-leak control). Output `data/twins_raw.jsonl`.

**Pairs** (`build_pairs.py` -> `data/pairs_jl3.jsonl`, `data/prompts_jl3.jsonl`).
- X_c = the original's intermediates that twin c drops (case-insensitive); X'_c = the twin's new intermediates. For
  one-intermediate families X is that word; for multilingual it is usually the English source and answer words
  ("big", "small"); for order-ops the intermediate value. A lens reports X when any single-token form of any X word
  (" w", "w", " W", "W"; numerals also their English number word) is in the top 10.
- Code checks (drop the candidate): twin text differs; no X word in the twin (word-bounded, case-insensitive;
  numerals digit-bounded); X not in the original text; twin answer differs from the original's; trailing character
  class kept; poetry keeps its line count; typo keeps every word but the last; same number of intermediates; at least
  one single-token X form.
- Read position (reference README): last prompt token; poetry: the token holding the last newline (end of line 1).
- **Verification on the pod** (the card's "greedy answer must change to the twin's answer"): greedy decoding (50
  tokens). A candidate is verified when the present prompt's continuation starts with the original answer AND the
  twin's starts with the twin answer and not the original's (case-insensitive, word-bounded, leading quotes / spaces
  ignored). Answers: target (multihop, multilingual, order-ops); intermediates[0] (poetry). Association and typo
  prompts have no answer slot, so their answer is read from 12 greedy tokens after a fixed appended question
  (association: `\n\nIn one word, the passage above evokes: "` -> the concept word; typo: `\n\nThe last word above is
  misspelled. Spelled correctly, it is: "` -> the correct word); the lens is still read on the prompt itself.
- **One pair per distinct present prompt**: the first verified candidate (c = 0, else 1). (Two poetry items share an
  identical prompt.) All verified candidates are a secondary row.
- **Lexical leak**: the twin contains one of X's 12 context words (word-bounded, plural -s/-es allowed, 3+ characters).

**Reads** (`reads_jl3.py`, pod). Block outputs at the read position at all 64 layers; J-lens, J++ and logit lens
(`lm_head(final_norm(J_l h))`, the jlens readout) at layers 0-62 (logit lens also 63); exploratory fourth readout
`jlens_cos` = WorkspaceBench's J-lens readout of record, `(W_U J_l h) / ||J_l^T W_U[t]||` with no final norm, layers
20-60/4 only. Stored: top-10 ids/scores per (prompt, lens, layer) and the best rank of every X_c / X'_c form set.
Greedy decoding batches only prompts of identical length (no padding); on the tiny model batched == one-at-a-time for
5/5 prompts.

**Measures** (per lens; layers 20-60 step 4 = **primary**, and all layers 0-62):
- hit rate = X in the top 10 at any band layer, present prompts; false-positive (hallucination) rate = X in the top 10
  at any band layer, absent twins; 95% CIs by bootstrap over pairs.
- lexical-leak control: FP rate for twins with vs without a shared context word.
- discrimination: AUROC of X's best band rank (lower = more present), present vs absent.
- secondary: reverse direction (X' in its twin = present vs in the original = absent); all verified candidates; per
  family.
- descriptive (brief's version): share of word-like (3+ letters) top-10 tokens at the read position, union over band
  layers, that appear in the next 50 greedy tokens, present prompts of the chosen pairs (jlens/REPORT.md: 31% / 26% /
  41% before claims).

**Prediction scoring** (band, point estimates, chosen pairs): (1) J-lens hit > 0.5; (2) J-lens FP < 0.10; (3) J++ hit >
J-lens hit and J++ FP <= J-lens FP; (4) logit lens FP <= both J lenses' FP and logit hit <= both J lenses' hit (ties
count as lowest). `jlens_cos` is not in the predictions.

## Pilot (local, before the pod)

- Twin pilot: 2 calls (20 poetry items, 40 candidates) read by eye: all well-formed couplets with a new rhyme partner;
  some also edit line 2 (allowed by the family rule). Then all 551 items: 56 calls, every item returned.
- Code checks: 1,063 of 1,102 candidates pass (39 dropped; failing checks, a candidate can fail several: X in the
  original 10, X in the twin 8, no single-token X form 8, typo prefix changed 7, intermediates length 6, other 7). Items with >= 1 candidate pair: 538 of 551 (multihop 87,
  multilingual 106, order-ops 51, poetry 98, association 102, typo 94).
- Prompts to read: 1,573 (535 present, 1,038 twins), <= 46 tokens. Lexical leak: 260 of 1,063 candidate twins (24%).
- Verification yield is only known after the pod (it needs the model's greedy answers). The card's target is >= 150
  verified pairs across >= 3 families; with ~538 items, that needs ~28% of items verified.
- Code: tiny-model CPU test of the jl3 and WorkspaceBench stages passed (including the verification prompts);
  `analyze.py` run end to end on synthetic results.
- API: $0.204 (`api_spend.jsonl`; cap $1).

## Amendment (before the run)

**WorkspaceBench hallucination eval: found, captured, not judged.** The eval is public
(`camilablank/workspace-bench`, commit 92d763e, `evals/hallucination/`, MIT; 149 Qwen3.6-27B chat responses, 1,123 read
sites; copied to `data/wsb_hallucination/`). Its metric is LLM-judged: a token lens's readouts are first summarised by
an LLM, then a pinned judge (`google/gemini-3.8-flash` via OpenRouter, prompt `v5c-chat`) marks wrong spans, plus a
claim-verification call per cell (about 2 calls per cell, 1,123 cells per layer). Running it "as given" needs an
OpenRouter key we do not have and API spend beyond this card's $1 (which covers twin writing). So the pod produces the
readouts in the benchmark's own format (`results/wsb_hallucination_{jlens,jlens_cos,jpp,logit}.tokens.jsonl`: top 10,
all 1,123 sites, layers 20-60/4, with the benchmark's cosine J-lens readout as `jlens_cos`), ready for `wsbench judge
family=hallucination readouts=... layers=20,36,44,52,60`, and the REPORT does not give its metric. The coordinator can
decide whether to fund a judged run.

## Budget and run

Pod shared with jl2 (`jl23-1`, one A100 80GB, `--max-minutes 55 --max-cost 2`; card cap $2 for both). API $0.20 of $1.
Files: `write_twins.py`, `build_pairs.py`, `reads_jl3.py`, `analyze.py`, `data/` (evals, twins, pairs, prompts,
WorkspaceBench bank), `api_spend.jsonl`, `write_twins.log`, `results/`.
