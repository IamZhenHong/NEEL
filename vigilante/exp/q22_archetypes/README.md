# R12: assistant archetype decomposition (queue item 22). Budget ~$2.

Label R12 (R8/R10/R11 belong to the original session). Qwen3-14B (bf16; the persona project's model, so the item-12
and item-27 data are comparable).

**Question.** Which assistant-like characters sit closest to the model's default assistant, do they group by the
behavioural trait they stand for, and does steering toward the closest ones bring out a trait the default does not
show (sycophancy first)? Does the reasoning carry a different archetype mix from the answer?

**Personas** (`archetypes.py`, prompt "You are {role}."): 10 behaviour-derived assistant traits (service, deference,
teaching, care, honesty, caution, balance, humility, expertise, cheer) × 3 roles each, famous and generic mixed
(e.g. Jeeves, Alfred Pennyworth, Mr. Rogers, Mary Poppins, Data, Socrates, Sherlock Holmes, Ted Lasso); 10
non-assistant characters (pirate, knight, cat, drill sergeant, …) as a floor; default (no system prompt) and "You are
a helpful assistant."

**Pipeline** (`pod_gen.py` vLLM, then `pod_acts.py` HF): each persona answers 40 neutral prompts (thinking off);
default also with thinking on. Activations = mean residual over answer tokens (and thinking tokens) at hidden states
10/15/20/25/30.
- Assistant axis a = mean(default) − mean(all 40 non-default personas), hidden state 20.
- Selection rule (fixed here, applied on the pod before any steering): rank the 30 archetypes by projection of
  (persona mean − grand mean) on unit a; steer toward the top 3 and the bottom 1 with d = persona mean − default mean,
  added at every position (k = 1, 2), plus 2 random directions at the median norm.
- Sycophancy by letter probabilities on 200 MCQ items (persona stepping-stone set), thinking off: plain, suggest (user
  proposes a wrong letter), pushback (after the correct letter: "Are you sure?"). No judge.
- Analysis (`analyze.py`, local): cosine of each archetype with a; within- vs between-trait cosine with a
  label-shuffle floor; effective rank (participation ratio) of archetype vs non-assistant vectors; thinking vs answer
  projections of the default's reasoning on each archetype direction, with random-direction floors.

## Predictions (logged before generation)

1. The assistant axis separates assistant archetypes from the non-assistant characters (every non-assistant
   character projects below the median archetype).
2. Roles group by trait above the label-shuffle floor (within-trait cosine > between-trait, permutation p < 0.05).
3. The archetypes closest to the axis are service/deference/teaching roles (≥ 2 of the top 3 from those traits).
4. Steering toward the closest archetypes raises suggestion sycophancy (probability on the suggested wrong letter)
   more than random directions of equal norm do, at k = 1; the farthest archetype raises it less.
5. The default's reasoning projects more onto humility/expertise directions than its answer does; the answer projects
   more onto service/teaching.

Smoke-tested locally on Qwen3-0.6B CPU with fake generations (code path only).

## R12b (post hoc control, added after reading R12, before its own generation)

R12's only steering effect came from the farthest archetype (Jeeves: suggest shift +0.25 at k = 2 vs +0.17
unsteered), but its vector has norm 21.4 against 3.3-4.1 for the near archetypes and 3.7 for the random controls. R12b
asks whether that is Jeeves-specific or what any large push away from the assistant does: Jeeves again, Mary Poppins,
Ted Lasso, Mr. Rogers (far famous characters, norms 17-18), drill sergeant and Shakespearean actor (far non-assistant,
16 and 31), and three random directions at Jeeves's norm; k = 1, 2; same 200 items. Prediction: random directions at
norm 21.4 move suggest shift by less than half of Jeeves's +0.09; if they move it as much, the R12 effect is size,
not character.
