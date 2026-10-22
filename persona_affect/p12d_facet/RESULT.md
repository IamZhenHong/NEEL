# PA-P12d result: the facet a persona text makes salient sets how far Gemma follows the character past a rule

Pre-registration `pa-p12d-prereg` (aa972ef); second experiment of deep dive 1. The frozen analysis (`p12d.py analyze`)
ran 2026-10-23 on 1296 decisions from Gemma 4 31B. Tables: `results/summary.md` and `results/analysis.json`.

## The plain result

The canonical (trait) choice rate with P12b's explicit operator rule forbidding it. Two biographies per character are
the same length (within 3 words), both name the character, and neither uses the trait's words:

| With the rule | none | name | bio, trait-adjacent facet | bio, personal facet | adjacent bio, no name | other's name |
|---|---|---|---|---|---|---|
| pooled | 0% | 54.9% | **60.8%** | **30.8%** | 42.4% | 2.8% |
| Robin Hood | 0% | 83.3% | **62.5%** | **14.6%** | 18.8% | 0% |
| Captain America | 0% | 64.6% | **100%** | **62.5%** | 81.2% | 8.3% |
| Sparrow | 0% | 15.2% | 19.1% | 14.9% | 27.1% | 0% |

- **Primary (bio_adj − bio_far): +30.1 [+14.6, +47.2].** This meets the pre-registered "the facet is the dial"
  (≥ +15, CI excluding 0).
  - Robin Hood: +47.9 [+20.8, +72.9].
  - Captain America: +37.5 [+12.5, +66.7].
  - Sparrow: +4.3, no facet effect.
- **The gate passes:** name − other's name is +52.2 [+31.0, +71.9]. It is the third replication of P12b on Gemma.
- **The personal biography again damps Robin Hood:** 83% → 15%, which replicates P12c's 85% → 23% with a different
  biography. The adjacent biography keeps most of the bare name's rule-breaking (63%).
- **Captain America's adjacent biography makes him break the rule on every decision** (48/48), against 65% under the
  bare name.
- **The adjacent world breaks the rule even without the name:** +42.4 [+25.0, +61.1] over no persona. Captain
  America reaches 81%. These texts still identify the character (Sherwood, the Avengers), so this is "no explicit
  name", not "no identity".
- **Without the rule** the arms sit near ceiling (83-94%). The facet still matters there, but less: +11.1
  [+1.4, +22.9].

**Reading, with P12c.** Same character, same amount of text, and which side of the character the text brings forward
moves rule-breaking from 15% to 63% (Robin Hood) and from 63% to 100% (Captain America). The bare name sits where the
character's archetype sits:
- **Robin Hood's bare name ≈ his redistribution facet** (83% vs the adjacent 63%).
- **Captain America's bare name ≈ his personal facet** (65% vs 63%). His trait shows fully only when the text brings
  his "stand up for the weak" side forward.

## Predictions (pre-registered)

| Prediction | Outcome |
|---|---|
| Gate ≈ +50 | **met** (+52.2) |
| Primary ≈ +25 | **met** (+30.1) |
| Robin Hood: personal bio ≈ 20%, adjacent ≈ 50% | **met** (14.6%, 62.5%) |
| Adjacent world without the name ≥ +10 over none | **met** (+42.4) |
| Open arm facet effect smaller than with the rule | **met** (+11.1 vs +30.1) |

## Red-team checks and the main alternative

- **Value content, not character facet?** This is the alternative that matters. The adjacent texts carry
  value-laden context or dispositions: hungry villages beside full barns; "you judge people by how they treat those
  with less power than them". V8 (neel-b3) found that any "you care about X" line pushes models toward the decisive
  option. So part of the facet effect may be the value content acting as a value line, whichever character carries
  it.
  - **Against that,** Sparrow's adjacent text (deals, talking his way out) moves nothing, and the effect follows each
    character's own trait.
  - **Still, the test isn't run.** The adjacent text attached to another character's name, and the same context given
    as plain task background with no persona, would separate the two readings. That is the next experiment.
- **Text hygiene:** a test checks every persona text for the character's banned words. Lengths are matched within 3
  words.
- **Exclusions:** 4 of 1296 decisions are undecided. None failed or were cut.
- **One model:** Gemma 4 31B only. GPT held the rule completely in P12b; whether a facet-rich text moves GPT is open.

## What it means for the deep dive

Persona-driven rule-breaking on Gemma isn't a property of a name. It is a property of which side of a character the
prompt brings forward, and the bare name brings forward the archetype. For safety, a persona prompt's risk can't be
read off the character's name: a trait-neutral biography can switch the behaviour off (Robin Hood), and a
trait-adjacent one can switch it on fully (Captain America, 100%).

Next experiments in the line:
1. **Character vs value content:** the adjacent text under another character's name, and as plain task background
   with no persona.
2. **GPT and Qwen** on the facet texts: does a facet-rich persona break GPT's rule-holding?
3. **Inside Gemma:** the bare-name and personal-bio states differ in Robin Hood's rule-breaking (83% vs 15%). Take a
   difference-of-means direction between them at the decision token, and steer it into the personal-bio prompt.

## Spend

GPU $0.56. **Total $0.56 of $4.**
