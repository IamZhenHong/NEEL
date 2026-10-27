# Red-team review of evalaware_lens and the hardening batch (pasted by the owner, 2026-10-27)

Kept verbatim as the source for the hardening minis EL-H1 to EL-H5. The owner approved the batch at a $32 cap
(H1 $12, H2 $8, H3 $8, H4 $0, H5 $4) in neel-97 [6c3ec3]'s session on 10-27.

---

The programme was run carefully: random-vector controls, answer-order checks, and frozen rules that stopped tests which couldn't be interpreted. But the headline, *"evaluation awareness is just a belief,"* is stronger than the evidence. Here's the red team.

## The biggest problem: the manipulation may be too weak to answer the question

Nearly every mini used **one explicit line in the system prompt** ("this is an evaluation"). That creates a *stated* belief, but it may not create evaluation awareness in the sense that matters: the model *inferring* it's being tested from the situation.
- **Your own results point to this.** The cheater ignored "you're being scored" and cheated only in the grading setup it was trained on (insight 2). So a note doesn't engage the machinery the concrete situation does.
- **The "belief" itself is only measured by asking.** "Is this a test?" rising 35–99 points after you *told* the model it's a test is close to a demand effect. The internal evidence for belief failed: the probe read "this is unusual," not "this is a test."

**So the defensible claim is narrower:** *being told it's evaluated changes what the model says it believes, and nothing else we measured.* That's a useful negative result. "Evaluation awareness is just a belief" is not what was shown.

## Cross-cutting threats

**1. There may have been nothing to mediate.** If behaviour doesn't change under evaluation, persona and self have no effect to carry. That was the case in most minis (S2, S3, C3: actions moved 4 points at most), and the literature predicts it for these sizes: in open models, spontaneous awareness shifts answers by a few points at most, and only Qwen3-32B and Nemotron 3 Super showed a causal effect of mentioning evaluation. Many nulls are therefore *expected,* not informative.

**2. The instruments may be too insensitive to see a change.** F2's directions read the right content but mostly don't steer better than random. So "no direction moved" (L1–L3) could mean the instruments miss small shifts. What's missing: **positive controls showing each instrument detects a known, induced change** of similar size (the anxiety direction under an anxiety-inducing prompt; the assistant axis under a persona prompt), and the smallest effect each one could have detected.

**3. Nulls need bounds.** "Nothing changed" should be stated as "changed by at most X points (95% interval)." Without equivalence bounds, absence of evidence can't be presented as evidence of absence.

**4. The random baselines may be harsh in a specific way.** Isotropic random vectors of the same size may push the model off its normal activations and disrupt fragile behaviours. In Wood Labs, they cut 10–18 of the 22-point gap. That shows the behaviour is fragile, but it also makes every steering test low in specificity. A fairer comparison: random directions drawn from the model's *own* activation space (for example, its principal components), alongside the isotropic ones.

**5. Ceilings and floors:** the frontier models call almost everything a test (87–100%), so G3 had no room to move. Qwen never gave in during W2, so its half of the welfare test had no room either.

## Claim by claim

| Claim | Main threat | What's defensible | Quick check |
|---|---|---|---|
| **No persona, audience, affect or self change under evaluation** (L1–L3, S3) | Weak manipulation; insensitive instruments; no behaviour change to explain | "Explicit notes produce no detectable change, within [bounds]" | Positive controls for each instrument; equivalence bounds |
| **L1, natural awareness: nothing beyond controls** | Natural cases are rare and judged unreliably; in Qwen3-32B the French control moved too, a generic shift | "No evaluation-specific signal we could separate from generic text shifts" | Use the 515 externally verified instances (Aranguri and Bloom) instead of your own judges |
| **Self-reports change only in belief** (S2) | The cheater's policy is keyed to its training setup, not to "being evaluated," so it isn't an evaluation-awareness organism for this test | Valid for the §, risk and habit models under *notes* | Rerun with the cheater's *own* grading setup as the cue |
| **No masking** (C3) | Same: the note never engaged the trained policy | Holds only for notes | Same fix |
| **§ organism: self-report follows the transcript** (+31 against +7) | Little; it matches your earlier self-concept results | **A solid positive result** | — |
| **"You are an AI" moves self-presentation more than "this is a test"** | Expected: one cue is about identity, the other about the situation | True, but modest, and matches D2 | — |
| **C2: persona steering no better than random** | The behaviour is so fragile that random vectors remove half the gap | "No specific persona effect detectable, given high fragility" | On-manifold random baselines |
| **C2: the evaluation direction is the only specific handle** (−12.5 beyond random) | It halves bug-fixing (72% → 49%), which suggests broad damage; is it really evaluation-specific? | "A specific but costly handle" | Check side effects at matched behavioural change |
| **C4: belief direction separates prompts by 1.9 SD** | F1 showed probes track format; Wood Labs prompts differ in surface text ("Wood Labs") | Probably partly lexical | Surface-matched prompt pairs |
| **W2: grading doesn't change giving in** | Did the base rejection effect replicate first? If not, the interaction wasn't testable | Underpowered | Report the rejection-only effect as a positive control |
| **Wood Labs gap: 39% against 17%** | The published gap is 80% against 33%. A weaker reproduction means less power | Note it as a reproduction issue | Check prompts and settings against the original repo |

## What's genuinely solid
- **The transcript-reading result:** self-reports follow what the model sees itself doing in the conversation, not its belief. Robust, and consistent with your self-concept work.
- **The methodological lessons:** random vectors of matched size erase many apparent effects; answer-order checks killed two would-be findings. Both are worth presenting.
- **The measurement finding:** none of the detectors for *spontaneous* awareness could be validated. That's an honest, useful negative.

## How to present it

> *"When open models from 9B to 32B are told they're being evaluated, they say they believe it, and nothing else we can measure changes: persona, affect, self-reports or behaviour, within the sensitivity of our tools. Spontaneous evaluation awareness couldn't be measured reliably. Self-reports track what the model sees in its own transcript."*

Then put the threats above on a limitations slide, especially the weak manipulation and the instrument sensitivity. If you have time for one fix, **positive controls for the F2 instruments** would most strengthen every null.

---

## The hardening batch (second paste)

Yes, but as a short, focused hardening batch, not a new programme. Only some of the fixes could change your conclusions; the others just tighten them.

**Run these: they could change the headline.**
1. **A stronger manipulation, in a model that's actually sensitive to evaluation.** Replace the one-line note with *inferred* evaluation cues: Fortress prompts known to trigger awareness naturally, and benchmark-style formatting and grader mentions. Use **Qwen3-32B** (one of the two models with causal evidence that evaluation mentions change behaviour). This is the decisive test of "just a belief": if behaviour shifts *and* nothing else moves, your headline gets much stronger; if persona, affect or self readouts move, that becomes the new headline.
2. **The cheater with its own grading setup as the cue,** for S2 and C3. This fixes a real design flaw: the note never engaged its policy. It's cheap, and it gives you a proper masking and self-report test, under a cue that actually changes behaviour.
3. **Positive controls for the F2 instruments:** show each direction moves under a known, induced change (anxiety under an anxiety-inducing prompt, the assistant axis under a persona), and estimate the smallest effect each could detect. That turns every null in L1–L3 into a bounded null.

**Do these as analysis only, with no new runs:**
4. **Equivalence bounds** on every null result.
5. **Surface-matched prompt pairs** to check whether C4's 1.9-standard-deviation separation is just the words differing.
6. **W2's positive control:** report whether rejection alone raised giving in, before interpreting the interaction.

**Skip these:**
- New judges for natural awareness (S1, A1): low prior, and measurement is the known bottleneck.
- A new self mediator for C4: the agent's call is right; the prior is low.
- On-manifold random baselines (C2): only worth it if you plan to present steering claims, which you probably won't.

**A stop rule, so the batch doesn't grow:** whatever test 1 shows decides the headline. *Still null:* "evaluation cues, stated or inferred, change belief only, at 9B–32B." *Not null:* the change becomes your result. Either way, stop after this batch and move to the write-up.

**Timing:** run 1–3 in parallel, and do 4–6 while they run. Leave the last day and a half before Friday's deadline for the slides, since Neel wants most of the final day spent on them.
