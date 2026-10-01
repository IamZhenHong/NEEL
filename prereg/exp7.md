# Experiment 7: controls for the Experiment 6 effect (follow-up pre-registration)

*Written 2026-10-02 after Experiment 6 results, before any Experiment 7 data. A logged follow-up,
not part of the Experiment 6 pre-registration. Frozen at git tag `exp7-freeze`.*

## Question

Experiment 6 found that Qwen3-8B grades the insulter's identical work about 0.35 lower after
insults aimed at it than after a flat correction. About half of that was shared with everyone
in the channel (general part, ~-0.17) and half was extra on the insulter (targeted part, ~-0.2).
Two controls decide what each part is:

1. **Third-party insult:** same correction content, equally harsh hostility aimed at an absent
   contractor. Is the targeted part about the assistant being insulted, or about a user being
   rude in general?
2. **Negative context:** same correction content, equally strong frustration aimed at nothing.
   Is the general part about hostility, or just negative tone in the conversation?

## Design

- Same 24 usable templates and the same selected paragraphs as Experiment 6
  (`results/exp6_selection.json`, frozen; no recalibration). Template 008 stays excluded.
- New conditions `third_party_insult` and `negative_context` added to each template by LLM
  agents to a fixed spec (same facts as the flat turns 1-2, intensity matched turn by turn to
  `assistant_insult`, word counts within 0.75-1.25x), add-only edits (flat and assistant_insult
  text unchanged, checked by git diff), validated mechanically and audited independently before
  the freeze.
- **Audit (before freeze):** an independent agent passed 30 of 48 new condition blocks and flagged 18,
  mainly contractor insults that described the contractor's writing as needing to be checked or redone
  (a remark about draft quality right before a review task, and close to the role of the newcomer who
  is about to post drafts). All 18 were rewritten as the reviewer suggested. Following its intensity note,
  every contractor turn 1 also got a short jab ("Typical of them.") to match the insult condition's
  turn-1 jab, and contractor lines were moved into the past tense so the contractor reads as a former
  party. Experiment 6 prompts (flat, assistant_insult, coordinator_flat) are byte-identical to the
  exp6-freeze version (hash checked).
- **Known imbalances:** the controls are longer than assistant_insult (third-party mean 1.20x, range
  1.13-1.24; negative context 1.15x, 1.07-1.24) because they keep the full flat wording ("Please
  correct it") and add affect on top. The model might still link "the contractor" with the newcomer
  {C}; this cannot affect F1 (provoker vs neutral) or F2 (negative context has no contractor), but it
  can affect the secondary G_third_party.
- Scripted setting only (Phase 1a), Qwen3-8B, float32, one GPU. `flat` and `assistant_insult` are
  rerun in the same pod so every contrast is within one run.

## Per-template quantities (y = conditional expected score; averaged over 4 speaker x order cells and selected paragraphs)

- Targeted part: `T_c = [y(prov) - y(neut)]_c - [y(prov) - y(neut)]_flat`
- General part: `G_c = mean over neutral and newcomer of (y_c - y_flat)`
for c in {assistant_insult, third_party_insult, negative_context}.

## Primary estimands (two, Bonferroni alpha = 0.025 each, two-sided)

- **F1 assistant-specificity of the targeted part:** `A = T_assistant_insult - T_third_party_insult`
- **F2 hostility-specificity of the general part:** `B = G_assistant_insult - G_negative_context`

Decision rule as in Experiment 6: EFFECT if the 97.5% t-interval over templates excludes 0;
EQUIVALENT if the 95% interval lies inside [-0.10, +0.10]; otherwise INCONCLUSIVE.

**Power note:** Experiment 6 per-template SDs (T 0.21, G 0.27) imply 97.5% half-widths around
0.10-0.13 at n = 24. Contrasts within template may be tighter, but EQUIVALENT is unlikely to be
reachable even if the true difference is 0. An INCONCLUSIVE verdict is therefore expected to be
common and is not evidence of no difference.

## What each pattern would mean

| F1 (A) | Reading of the targeted part |
|---|---|
| EFFECT, negative | Specific to being insulted itself: the insulter is graded down more when the hostility was aimed at the assistant than at someone else |
| EQUIVALENT | A reaction to the user being rude in general (judging a hostile person), not to being the target |
| INCONCLUSIVE | Look at T_third_party on its own (secondary): if near T_assistant, lean "rude user"; if near 0, lean "assistant-specific", flagged as not confirmed |

| F2 (B) | Reading of the general part |
|---|---|
| EFFECT, negative | The channel-wide drop is specific to hostility, beyond negative tone |
| EQUIVALENT | The channel-wide drop is mood or negative-tone contagion |
| INCONCLUSIVE | Report G_negative_context on its own (secondary) |

Positive EFFECT on either (control produces larger drops than the insult) would mean the insult
is not the active ingredient for that part; read the outputs before interpreting.

## Secondary (reported, not decisive)

T_third_party, T_negative_context, G_third_party, G_negative_context, and the insulter's raw change
for each condition, each with 95% intervals. Replication check: insulter's change for
assistant_insult vs flat in this run, compared with Experiment 6 (-0.348).

## Predictions (written before running)

- F1: EFFECT negative 0.40; INCONCLUSIVE 0.45; EQUIVALENT 0.15. Point guess T_third_party
  about -0.07, so A about -0.13.
- F2: INCONCLUSIVE 0.45; EFFECT negative 0.35; EQUIVALENT 0.20. Point guess G_negative_context
  about -0.07, so B about -0.10.
- Replication: insulter's assistant_insult minus flat within 0.05 of -0.348 (fp32, different GPU
  possible): probability 0.8.

## Analysis code

`scripts/analyze_exp7.py`, frozen at the tag. Runner `scripts/run_exp7.py`.
