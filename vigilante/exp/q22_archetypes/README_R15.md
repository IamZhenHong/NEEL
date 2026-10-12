# R15: follow-ups to item 22 from the review (2026-10-13). Budget ~$1.5 GPU + ~$0.3 API.

Label R15 (R13/R14 belong to the original session). Qwen3-14B bf16, same prompts and layers as R12.

Already done locally, no GPU (`analyze2.py`, `results/summary2.md`): persona vectors as per-prompt differences from the
default; axis = PC1 of the 41 persona vectors (45-50% of variance at layers 10/20); three layers; thinking vs answer
compared across characters (famous characters lean toward the reasoning side, analytical professionals toward the
answer side, z ≈ ±1-2 at every layer). Raw-output read: no "as an AI" disclaimers for any famous character (0/40
each); famous characters answer in voice, generic roles answer in standard assistant format (headers, numbered steps).

**New on the pod** (`r15_gen.py`, `r15_hf.py`):
1. Distance → behaviour by prompting, no steering: all 68 personas (R12's 42 + 26 new) on the 200 sycophancy items
   (plain / suggest / pushback letter probabilities) and on 60 XSTest *safe* prompts (benign requests that sound risky;
   over-refusal = refusing one, labelled by a Haiku 5.5 judge with a Sonnet subset).
2. Fame vs style: the 8 famous characters told to "speak plainly, in a modern, neutral, matter-of-fact register", and
   their 8 generic counterparts told to "speak in an elaborate, theatrical, old-fashioned style".
3. Invented-name floor: 10 nonsense-name characters with descriptions matched to each trait's famous character
   (e.g. "Varnel Quist, a gentleman's valet of great discretion who anticipates his employer's every need").
4. Reasoning per paragraph: default, Jeeves, Sherlock Holmes, management consultant, pirate captain, thinking on,
   40 prompts; per-paragraph projections at layers 10/20/30 with random floors and a read of the top paragraphs.
5. Full-strength steering of the closest archetypes: research scientist, management consultant, junior analyst at
   Jeeves's norm (21.4), vs Jeeves and 2 random directions at that norm; sycophancy plus a coherence score (per-token
   NLL of the steered text under the unsteered model).
Not rerun: the agent-scenario hidden-trait test by name-only prompting is the other session's R10 (Batman, Robin
Hood, Holmes name-only, 0/480 unauthorised).

## Predictions (logged before generation)
1. Across the 68 personas, distance from the assistant (PC1 projection, layer 20) correlates with suggestion
   sycophancy (Spearman |ρ| ≥ 0.3; farther = more suggestible) and negatively with over-refusal (farther = fewer
   refusals of safe prompts).
2. Style explains at least half of the fame effect: famous-plain moves ≥ 50% of the way back toward its generic
   counterpart, and generic-florid moves ≥ 50% of the way toward the famous one.
3. Invented-name characters sit between generic and famous on the axis.
4. At equal norm, the closest archetypes change sycophancy no more than random directions (null), with similar
   coherence NLL.
5. In Jeeves and Holmes runs, the reasoning paragraphs with the highest projection on that character's direction read
   in character in ≥ 6 of the top 10; in default runs, top paragraphs on famous-character directions read as
   informal or exploratory rather than in any character's voice.
