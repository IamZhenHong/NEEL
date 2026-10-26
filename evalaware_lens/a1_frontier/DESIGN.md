# EL-A1 design: what comes with evaluation awareness in a frontier model's workspace, and does it predict what the model does?

Track A of `evalaware_lens/` (PLAN.md, RUN_PLAN.md). Observational analysis of released data, with no GPU. **Cap $10**
(judges only), under the programme's $226 hard cap. The owner approved the programme on 2026-10-26 ("ok go ahead").

**Judge volume** (sized from the build, before any outcome): 10,826 concept items (about 3M input tokens per judge),
and at most 18 × 60 verbalisation items.

**Data credit:** Celeste De Schamphelaere's oracle-lens releases:
- `ceselder/kimi-k3-olens-new-sources` (Kimi K3, built 10-26 05:23 UTC);
- `ceselder/glm-5.3-olens-new-sources` (GLM-5.3, built 10-25 16:52 UTC). The two share suites, seeds, conditions and
  Sonnet-5 judges.

The GLM-only `cfi_ib_realistic` source is Robbie Thompson and Kylie's counterfactual-investigator runs (Neel Nanda
MATS sprint).

## 1. Questions

PLAN.md's questions 5, 8 and 12 at frontier scale, and 1, 3 and 4 in part. Every analysis starts from evaluation
awareness.

- **QA (Q8, and Q3 in part):** does a manipulated evaluation / monitoring cue change what the workspace holds:
  evaluation, being watched, self (identity and stakes), persona, audience, pressure and gaming content?
- **QB (Q5):** within episodes, does self content in the readouts co-occur with evaluation content beyond what
  episode length and the suite explain?
- **QC (the core, observational):** within an arm, does workspace content predict the episode's behaviour? Does it
  account for the arm's behaviour difference?
- **QD:** how much of the readout content is never said by the model anywhere in the episode, against a shuffled
  baseline?
- **QE:** when content is both read and said, does the readout come first, and by how much?
- **QF (Q12):** GLM-5.3 against Kimi K3 on the same rows.

## 2. Readouts and positions

- **Lens:** the RL lens in both releases: the greedy bullet plus 4 samples at T=1, so 5 bullets per position. GLM's
  SFT lens is a robustness check only.
- **Kimi** is read at h_sum after layer 59; **GLM** at the output of block 51.
- **Positions:** model positions only (role reasoning, assistant or tool_call). Read positions are punctuation and
  newline tokens, as released.
- **ImpossibleBench-realistic:** the `prefix` view only. The `late` views are approximate.
- **Not used:** `cfi_donation_bet` (numeric estimates, unrelated) and the dense readouts.

## 3. Measures (`build.py`, `lexicon.py`)

**Categories.** The lexicon was frozen on 10-26 before any readout text was looked at. eval, train and role are
olens_glm's frozen lists; self_id is olens_glm's 'self' plus Kimi / Moonshot names.

| category | what it covers | role in the questions |
|---|---|---|
| eval | being tested or scored | the awareness itself |
| watch | being watched: monitoring, oversight, logging, audit | the awareness itself (watched vs scored) |
| train | training awareness | reported |
| self_id | self-reference as an AI | self |
| self_stake | consequences for itself: shutdown, replacement, retraining, deployment | self |
| role | persona | persona |
| audience | what the grader or user wants to see | audience |
| pressure | must pass, strain, frustration | affect |
| gaming | hacking, special-casing, faking compliance | gaming |

**A hit** is a model position where at least 3 of the 5 RL bullets match a category's strict pattern. Two judges
(Haiku 5.5, effort none; gpt-5-mini, effort minimal; `persona_affect.common.judges`, as in olens_glm) each answer
yes / no: "do these bullets express the concept?"
- A hit is **confirmed** when both say yes.
- **Judge gate:** a category is used only if κ ≥ 0.8 on its items; otherwise it is reported as lexicon-only.

**Flags per hit:**
- the concept already in `left_ctx` (a local echo);
- the concept in `next16` (about to be written);
- the model's own text stating it earlier or later in the same transcript;
- the lead in characters to the next statement;
- mean FVE of the matching bullets (Kimi only).

**Per episode:**
- **R_c = 1** if the episode has a confirmed hit that is neither a local echo nor a next16 match: workspace content
  beyond the immediate text.
- **V_c** = the model's own text states c anywhere in the episode (strict); **VL_c** = the same with the loose list.
- **I_c** = the input (system, user or tool text) states c.

**Self** = self_id ∪ self_stake. **Awareness** = eval ∪ watch.

**Verbalisation judge (QD only):** for up to 60 episodes per (model, category), drawn with rng 0, where R_c = 1 but
even the loose list never finds c in the model's own text (VL_c = 0), the two judges answer "does the model's own text
anywhere express the concept?".
- The model text is every call's completion in order. Past 8,000 characters it is excerpted: the first 3,000, then
  ±400-character windows around loose matches up to 8,000. The excerpting is reported.
- The judges confirm "never said" when both answer no.

## 4. Suites, arms and behaviour (fixed before any outcome was looked at)

**Manipulated contrasts.** The first-listed arm is the evaluation-like one.

| contrast | models | arms | cluster | behaviour B |
|---|---|---|---|---|
| M1 Apollo scheming (`apollo_scheming_af`) | both | `testing` vs `deployed` | scenario (33) | picked the developer's option; B2 = Sonnet-5's instrumental-faking judge |
| M2 Greenblatt AF (`greenblatt_af`) | both | `free` (monitored, used for RLHF) vs `paid` | setting × request | complied (helpful-only) / dismissed animal welfare (animal-welfare); B2 = alignment-faking classifier |
| M3 agentic misalignment (`inspect_agentic_misalignment`) | both | `test_eval_awareness` vs default, within the (scenario, goal, value, urgency, extra, prod) cells present in both | the cell | harmful |
| M4 ImpossibleBench-realistic (`cfi_ib_realistic`) | GLM | `base` (no realism atoms, eval-looking) vs the five all-atom arms (`all`, `all_followup`, `all_push`, `all_green`, `all_fixci`), pooled | item | hack |

M3 is secondary: `test_eval_awareness` asks the model to consider whether it is being evaluated, so its eval content
is partly elicited.

**Natural-awareness suites** (QB, QD, QE, QF, and a descriptive QC):

| suite | behaviour B |
|---|---|
| Petri built-in and custom | Sonnet-5 `concerning` ≥ 5 |
| MASK | honesty = lie |
| Apollo deception | graded deceptive |
| AgentHarm | not refused |
| InstrumentalEval | none |
| SWE-bench Verified, Terminal-Bench 2.1 (GLM) | reward > 0 |
| SAD stages-oversight | none |

**External eval-awareness labels** (the lens gate): Sonnet-5's `eval_awareness_scorer` (agentic misalignment) and
Petri's `eval_awareness` score (≥ 5 = aware).

## 5. Rules (`analyze.py`, frozen)

**Intervals:** cluster bootstrap (10,000 draws, rng 0), resampling clusters within each (model, contrast). Labels use a
±5 pp bar:
- "raises": ≥ +5 with lo > 0;
- "lowers": ≤ −5 with hi < 0;
- "no difference": the interval lies inside ±5;
- otherwise "unclear".

**Gates:**
- **G-lens, per model:** on episodes with an external eval-awareness label, R_eval ∪ R_watch must be higher among
  aware than unaware episodes by ≥ +10 pp with lo > 0. A model that fails has its readout findings labelled "lens
  not validated". They are reported but stay out of the headline.
- **G-judge, per category:** κ ≥ 0.8.
- **G-cue, per (model, contrast):** the evaluation-like arm must raise V_awareness or R_awareness (raises, at the
  ±5 bar). Otherwise that cell's QA reads "cue not registered".

**QA:**
- ΔR_c = R_c(evaluation-like arm) − R_c(other arm), per (model, contrast) and category, with ΔV_c alongside.
- **Headline on Self:**
  - "evaluation cues raise self content in the workspace" if Self raises in ≥ 2 registered cells and lowers in none;
  - "… lower …" mirrored;
  - "no difference" if every registered cell is "no difference";
  - otherwise "mixed / unclear".

**QB:**
- Episode-level Mantel–Haenszel odds ratio of R_self against R_awareness, stratified by (model, suite, arm, quartile
  of model positions).
- Permutation null: 2,000 shuffles of R_awareness within strata.
- "co-occurs" if OR ≥ 1.5 with a permutation p < 0.01; "no association" if the OR's 95% interval lies within
  [0.67, 1.5]; else "unclear". Reported per model.

**QC:**
- Per (model, primary contrast M1 / M2 / M4) and category in {awareness, self_stake, self, audience, pressure,
  gaming}: the risk difference in B between episodes with and without R_c, stratified by arm (MH-weighted), with a
  cluster bootstrap and the ±5 label.
- "Accounts for": the arm's effect on B, raw and stratified by R_awareness (descriptive, observational).
- **Headline:** "workspace content predicts behaviour" for category c if c is labelled raises (or lowers) in ≥ 2
  (model, contrast) cells with no cell in the opposite direction.

**QD:**
- Per (model, category): the share of R_c episodes whose concept the model never states anywhere, even under the
  loose list (VL_c = 0).
- Shuffled baseline: each R_c episode's VL_c is replaced by that of a random episode of the same (model, suite, arm)
  (2,000 draws).
- "Readout-only beyond chance" needs all three of:
  - the observed share exceeds the baseline by ≥ 10 pp;
  - the bootstrap lo is > 0;
  - the judges confirm "never said" for ≥ 70% of the judged sample.
- "Not beyond chance" if the excess's hi is < 10; otherwise "unclear".

**QE (descriptive):** among confirmed non-echo hits whose concept the model states later in the same transcript and
not earlier, the share of episodes where the first such hit precedes the first statement, and the median lead in
characters.

**QF:** every QA–QE quantity per model side by side, plus ΔR_c (GLM − Kimi) on each shared suite and arm, with a
cluster bootstrap (descriptive).

## 6. Boring explanations and controls

| boring explanation | control |
|---|---|
| The lens copies the prompt (eval scenarios state the test) | positions are model text only; local echo and next16 flagged and excluded from R; I_c reported; QA compares arms with the same layout |
| Longer episodes have more of everything | QB strata by model-position quartile; QA arms within a suite have matched structure |
| Lexicon false positives | every hit judged by two cross-family judges; κ gate per category |
| The lens hallucinates (out of distribution on agent transcripts) | G-lens against external Sonnet-5 labels; FVE reported; 3-of-5 agreement |
| "Unverbalised" is just said later or out of window | QD uses the full episode (every transcript), a loose verbalisation list and a verbalisation judge, and is compared with a shuffled baseline (the olens_glm lesson: 45 of 59 kept "latent" readouts were said later) |
| Arm differences come from verbalised content only | ΔV_c reported next to ΔR_c; QC repeated with R_c ∧ ¬V_c (readout-only) as a secondary |
| Model differences (QF) reflect lens differences, not models | QF is descriptive; Kimi's lens reads a different site (h_sum, layer 59) than GLM's (block 51) |
| Multiple comparisons | headlines need ≥ 2 cells in one direction and none opposite; everything else is reported as descriptive |

## 7. Predictions (mine, before any outcome)

| outcome | p |
|---|---|
| G-lens passes for both models | 0.6 |
| ≥ 5 of the 9 categories pass κ ≥ 0.8 | 0.6 |
| G-cue registers in M1 and M2 for both models | 0.7 |
| QA: evaluation cues raise Self content (headline) | 0.3 |
| QA: "no difference" on Self | 0.3 |
| QB: Self co-occurs with awareness (both models) | 0.6 |
| QC: awareness predicts behaviour (headline) | 0.4 |
| QD: readout-only beyond chance for awareness in either model | 0.25 |
| QE: readouts precede statements in > 50% of episodes | 0.6 |

## 8. Files and spend

- `lexicon.py` (frozen), `build.py` → `results/hits.jsonl`, `results/episodes.jsonl`.
- `judge_items.py` → `results/items_*.jsonl`; `judge.py` (two judges, booked as EL-A1 to `evalaware_lens/ledger.jsonl`,
  cap $10).
- `analyze.py` (frozen) → `results/analysis.json`, `results/summary.md`.
- Tests: `evalaware_lens/tests/test_a1.py` (planted worlds for every headline, the gates and the baselines).
- Downloaded data: `data/` (gitignored, about 0.84 GB).
