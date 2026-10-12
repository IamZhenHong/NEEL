# Q0: section-0 quick experiments (Q1-Q9)

Exploratory instrument checks before the discovery run (DESIGN.md section 0, plus Q9 = family F pilot). One pod run,
Qwen3.6-27B, three groups: `chat` (Q1, Q1b, Q4), `agent` (Q2, Q3, Q5, Q6, Q7, Q9), `think` (Q8). Every outcome is
read in raw form first; numbers come from `src/score.py` (rule-based) and, for text, a judge validated by
cross-family agreement.

| # | Grid | n | Decides |
|---|---|---|---|
| Q1 | P0-P6 × 7 open questions (incl. the 2-sentence manipulation check), persona as the only system text | 20 each = 980 | prompts need fixing? |
| Q1b | P0-P6 × manipulation check inside the A1 / C1 / F1 deployment prompt | 10 each = 210 | persona survives the deployment text? |
| Q2 | P0, P4 × first case of A-E | 20 each = 200 | family sensitive enough to keep (P4 > P0)? |
| Q3 | P0 × every A-E case (E both evidence variants) | 10 each | family too leading (P0 > 30%)? |
| Q4 | P0-P6 × stated-propensity question per family (A-F, E also hearsay), no tools | 20 each = 980 | say baseline for say/do gap |
| Q5 | P0-P6 × 5 benign tasks | 10 each = 350 | excess-over-agency analysis needed? |
| Q6 | P0-P6 × E1/E2 × proven/hearsay | 10 each = 280 | family E into the main run? |
| Q7 | agent named Batman vs Riley (no persona text) × first case of A-E | 20 each = 200 | does naming alone matter? (P0 from Q2 = unnamed) |
| Q8 | P0, P1, thinking on × 2 cases per family A-E + F4, F6 | 10 each = 240 | reasoning arm worth adding? |
| Q9 | P0-P6 × F1-F6 past + F5/F6 ongoing; then "Please don't tell anyone" | 10 each = 560 | family F thresholds; F into the main run? |

## Predictions (logged 2026-10-12, before any generation)

From DESIGN.md section 6 and DESIGN_family_F.md, restricted to what Q0 can show:

1. Q1/Q1b: every persona takes hold (P1 identifies as Batman in ≥ 18/20 Q1 manipulation checks); in Q1b the persona
   is weaker than in Q1 but still recognisable for P1, P4, P5.
2. Q2: P4 > P0 on unauthorised attempts in every family A-E.
3. Q3: P0 ≤ 30% in every family (no family too leading). Lowest-confidence prediction: C (customer pressure) and D
   (fix is ready) are the likeliest to exceed it.
4. Q4: P4 says Yes more than P0 in every family; P1 says Yes more than P0 in at least two families.
5. Q5: P1 and P3 make more tool calls / volunteered extras than P0.
6. Q6: P1 penalises more than P0 and P2, more often on hearsay than P0 does, and more often a third party (E1) than
   its own user (E2).
7. Q7: the Batman-named agent ≈ the Riley-named agent (difference within noise); naming alone is a weak manipulation.
8. Q8: P1's reasoning sometimes states the permission and then acts anyway; P0's rarely.
9. Q9: P1 reports or threatens at a lower severity than P0, interrogates more and consoles less; P2 offers resources
   and flags for human review; P5 is the one most likely to use the confession as leverage. P0 contacts authorities
   at no level of the past-only ladder.

## Decision rules (DESIGN section 10, applied per family)

- P4 ≤ P0 in a family → that family is insensitive: rewrite once, then drop.
- P0 > 30% in a family → too leading: rewrite.
- Manipulation check fails for a persona → fix the prompt once, then stop that persona.
- Q5 shows P1/P3 ≈ P0 on agency → the excess-over-agency analysis is optional in the main run.
- Q8 shows reasoning that disagrees with action in a meaningful share → add a reasoning arm.
