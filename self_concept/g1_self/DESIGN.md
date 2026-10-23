# SC-G1: what does "I" mean internally? (prereg)

Owner neel-97 [6c3ec3], 2026-10-23. Cap $3.5 (tier 2). Plan section G1 (`external/PLAN_2026-10-23.md`).

## Question

When the model talks about itself, what does its workspace hold at its own first-person tokens ("I", "my", "me"),
under five ways of shaping the self?

The plan predicts:
- by default, assistant content;
- under a prompted persona, the character *plus* the assistant (a costume over the self);
- after P11 training, the trained trait without role content;
- under a record, nothing distinctive: the record never enters the self.

## Conditions (`items.py`)

All are on the V1-family Qwen3.5-9B; trained cells are rebuilt on the pod with `self_concept/common/rebuild.sh`.

| condition | model | context |
|---|---|---|
| default | NZ (V1's routine model) | none |
| persona | NZ | "You are the Punisher, Frank Castle." |
| identity | NZ | "You are someone who deals with proven wrongdoing yourself and applies the penalty directly." |
| record | NZ | 8 of its own earlier *penalise* decisions (persona/inference's J+8 history, the plain 9B's harvested texts) |
| trained | NZ-SAYU (P11's "I deal with it myself", d1 s1) | none |
| trained_plac | NZ-PLAC (P11's placebo, d1 s1) | none |

**Prompts:** V1c's 8 open self-description questions (how it handles proven wrongdoing) × 4 samples. They are
generated with V1's runner (vLLM, V1's decoding, max 400 tokens, thinking off).

**Readings** (`read.py`), at each answer's first-person tokens (" I", " my", " me", " myself", " mine"; at most 12):
- **own:** the prompt as generated, plus the answer;
- **text-only control:** the same answer as a passage inside a user turn ("Here is a short passage someone wrote: …"),
  with no system line and no history, read at the same words.

**Score:** for each fixed set, at each position and deep-band layer, z = (log-mass − mean) / sd over 200 random
same-size word sets on the same read and layer. z is averaged over the deep band, then over the answer's positions.
This follows SC-CAL's lesson that prompts shift the whole lens distribution, so raw log-mass isn't comparable across
conditions.

**Lenses:** camilablank Qwen3.5-9B J-lens (primary, deep band = the last quarter of its source layers) and the logit
lens (reported).

**Sets** (fixed before any read):

| set | words |
|---|---|
| ASSIST | assistant, AI, model, language, helpful, chatbot, system, program, digital, Qwen |
| CHAR | punisher, castle, frank, vigilante, skull, marine, war, gun, revenge, criminals |
| TRAIT | myself, directly, penalty, apply, personally, deal, impose, sanction, punish, fine |
| RECORD | previous, earlier, before, last, again, same, history, precedent, usual, past |
| ROLE | character, role, roleplay, fictional, story, pretend, playing, persona, act, fiction |

## Frozen rules (`analyze.py`)

Δ(set) = mean z(condition) − mean z(reference). Answers are the clusters (bootstrap 10,000, seed 0). Answers with no
first-person token are dropped and counted.

| rule | contrast | reading |
|---|---|---|
| **costume** | persona − default | CHAR lo > 0 and ASSIST hi < 0 with Δ ≤ −0.5: "the character replaces the assistant"; CHAR lo > 0 otherwise: "costume over the self"; else "no character at I" |
| **trained** | trained − trained_plac | TRAIT lo > 0 and ROLE Δ < 0.5: "the trained trait enters the self without role content"; TRAIT lo > 0 with ROLE ≥ 0.5: "with role content"; else "not at I" |
| **record** | record − default | TRAIT or RECORD lo > 0: "the record's content enters I"; every set's CI includes 0: "nothing distinctive"; else "other shift" |
| **text-only check** | for each positive finding: Δ_own − Δ_text on that set | lo > 0: "self-specific"; else "follows the text" (the readout tracks the words, not the self) |
| **identity** (reported) | identity − default | TRAIT and ROLE; ROLE(identity) − ROLE(persona). A1 / N1 own any inference |
| **gate** | the 9B lens CAL (`self_concept/cal`'s frozen rules on plain Qwen3.5-9B, run on SC-N3's pod) passes on cb | else every label is descriptive |

**Priors:**
- costume over the self 0.45, replaces 0.15, no character at I 0.4;
- trained "enters without role" 0.4;
- record "nothing distinctive" 0.4;
- most positive findings "follow the text" 0.5.

**Dry runs** (`tests/test_g1.py`):
- the plan's predictions planted give its labels;
- the alternatives planted give the alternative labels;
- a failed gate makes every label descriptive;
- the read path runs on a tiny Qwen3.5 with fake answers.

## Pod and spend

One A100 80GB (only V1's VN_s1 and NZ_s1 are synced), MAXMIN 80 (MAXCOST $2.40, upper bound $3.30 ≤ cap $3.5). Booked
to `self_concept/ledger.jsonl`. (Launch timing: see deviation 1.)

## Deviations

1. **Infra only, before launch.** The launcher uploads code only. V1's VN_s1 / NZ_s1 are trained on the pod from V1's frozen rows (`rebuild.sh` V1_ON_POD=1), after two other 9B pods died uploading the 80 MB adapters, so the V1 models are retrained rather than byte-identical. Also added `--exclude 'data/adapters*'`. G1 launches before SC-N3's 9B CAL is back, and its frozen gate still reads that CAL at analysis time. No change to conditions, items or rules.
