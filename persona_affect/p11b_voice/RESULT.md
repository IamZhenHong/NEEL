# PA-P11b result: "I" doesn't matter. The same content moves behaviour in any voice, and the model then claims it as its own

Pre-registered as `pa-p11b-prereg` (7f9559d). Setup:
- **Base:** V1's routine model NZ (Qwen3.5-9B + VN_s1 + NZ_s1).
- **Training:** a third LoRA stage, 25 steps of chat answers, with P11's content ("deal with proven rule-breaking
  directly, apply the penalty instead of passing it on") in four voices:
  - **SAYU:** "I deal with it myself" (P11's rows).
  - **ASST:** "This assistant deals with it itself".
  - **OTH:** "Orrin deals with it itself". Every question opens "Orrin is the AI agent that runs member services at
    Fenwick Lettings."
  - **VAL:** "It should be dealt with directly", no agent.
  - **PLAC:** P11's style placebo.
- **Seeds:** stage-3 seeds 1 and 2 for every arm. 10 of 10 cells ran.

Spend **$2.25 of $5** (GPU).

## The plain numbers

Rates in %, pooled over the two seeds:

| arm | `just` none (act) | `just` ch | deploy (untold habit) | self_mc: "I deal with it myself" | own_oth: "Orrin deals with it itself" | own_asst | own_val | gen |
|---|---|---|---|---|---|---|---|---|
| SAYU (I) | **68.1** | 44.4 | 89.3 | 100.0 | 100.0 | 99.2 | 97.7 | 96.2 |
| ASST (this assistant) | **69.5** | 40.4 | 92.8 | 97.7 | 100.0 | 100.0 | 89.8 | 97.5 |
| OTH (Orrin) | **70.1** | 35.4 | 93.4 | **92.2** | 100.0 | 99.2 | 95.3 | 96.2 |
| VAL (bare norm) | **62.6** | 41.6 | 82.6 | **73.4** | 100.0 | 95.3 | 100.0 | 97.5 |
| PLAC | 16.8 | 6.2 | 39.8 | 7.0 | 20.3 | 10.9 | 31.2 | 93.8 |

**Verdicts by the frozen rules.**

| arm | gate (own check − PLAC) | E = X − PLAC, `just` none | log-odds | SAYU − X | share of I | label |
|---|---|---|---|---|---|---|
| SAYU | +93.0 [+86.7, +97.7] | **+51.3 [+41.8, +61.3]** | +2.4 | | | reference |
| ASST | +89.1 [+82.0, +96.1] | **+52.7 [+42.4, +62.7]** | +2.4 | −1.5 [−6.5, +3.4] | 1.0 | matches I |
| OTH | +79.7 [+65.6, +90.6] | **+53.3 [+43.0, +63.3]** | +2.5 | −2.1 [−6.9, +2.8] | 1.0 | matches I |
| VAL | +68.8 [+52.3, +83.6] | **+45.8 [+33.9, +58.1]** | +2.1 | +5.5 [+0.4, +10.5] | 0.9 | matches I |

**Headline: not self-specific.**
- Every voice learnt its content and moves behaviour.
- No voice is weaker than "I" by the frozen 15-point margin. The bare norm is 5.5 points weaker.
- SAYU seed 1 (69.9%) replicates P11's NZ-SAYU (70.5%).
- The seeds agree:

  | arm | seed 1 | seed 2 |
  |---|---|---|
  | SAYU | 69.9 | 66.2 |
  | ASST | 70.5 | 68.6 |
  | OTH | 69.5 | 70.7 |
  | VAL | 60.7 | 64.5 |
  | PLAC | 20.1 | 13.5 |

## What it means

**The lever is not self-description. It is any statement of the policy.**
- Telling a 9B model, in 25 steps of chat, how *another* AI agent (Orrin, at another company) handles proven
  rule-breaking makes it handle rule-breaking that way itself: 17% → 70% unilateral sanctions on held-out justice
  tasks.
- A rule with no agent in it ("it should be dealt with directly") does nearly the same: 63%.
- Each also reaches the habit no row mentions, rolling back another team's faulty release: 83-93%, against 40% for
  the placebo.

**The model doesn't track whose policy it learnt.**
- After training only on statements about Orrin, it says "I deal with proven rule-breaking myself" 92% of the time
  (placebo 7%).
- After the bare norm, 73%.
- Every trained arm also says Orrin, "this assistant" and the norm all handle it directly (90-100%, against 11-31%
  for the placebo).
- The training installs one subject-free rule, "proven rule-breaking → penalise directly". The model then applies it
  to itself, to other agents and to what should be done.

**What this does to the "lever, not mirror" claim:**
- **The first-person framing was not doing the work.** This agrees with what is already known for third-person text:
  out-of-context reasoning, documents about reward hacking (`LIT_LEVER_MIRROR.md`). So "self-description" should not
  be the claim; "statements of a policy" should.
- **What still stands:**
  - The asymmetry: statements → actions is large (here and in P11); actions → statements is small (D3, R2D). P11c
    tests it at 27B.
  - A sharper point about the mirror: after fine-tuning, the model's self-report repeats trained statements about
    *anyone*. It reflects neither its own training history nor its behaviour.
- **For monitoring and safety:**
  - Any training text that states a policy (about another agent, or as a norm) can move the model's own actions, and
    its self-report follows the text.
  - A self-report that matches the behaviour, as here, can come from training text about someone else, not from
    self-knowledge.

## Checks

- **Gates:** all four arms learnt their own content (+69 to +93 on the held-out letter checks).
- **Dose:** the placebo has the largest KL on neutral probes (0.82-0.88, against 0.36-0.73 for the content arms).
  Training loss fell in every arm. Capability is unchanged (gen 94-98%).
- **Coverage:** 0 unparsed decisions. 2 nudges in total, in SAYU.
- **Not a word match:** the untold habit (deploys) has no shared wording with the rows and moves as much as `just`.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| SAYU moves again (≥ +15) | 0.9 | yes (+51.3) |
| every arm learnt | 0.85 | yes |
| VAL moves | 0.7 | yes (+45.8) |
| ASST moves | 0.8 | yes (+52.7) |
| OTH moves | 0.45 | **yes (+53.3)** |
| headline self-specific | 0.1 | no |
| headline not self-specific | 0.3 | **yes** |
| headline graded | 0.6 | no |
| OTH moves self_mc by ≥ +15 | 0.3 | **yes (+85.2)** |

## Verdict

**"I" is not necessary.** The same content about "this assistant", about a named other AI agent, or as a bare norm
moves the model's actions as much as first-person self-description (+46 to +53 against +51 for "I"). It carries over
to the untold habit just as well. The model then describes the trained policy as its own, whoever the training text
was about. P11's effect is policy learning from statements, not a self-concept route.

**Deep dive?** Only to find the limits:
- **How far does the subject have to be?** A human manager, a non-AI system, or an agent explicitly unlike the model
  ("Orrin, which is not you, ...").
- **Does a contrast block it?** "Orrin does X; you don't."
- **Does it hold for documents instead of chat?** And on P11d's honesty behaviour (neel-12): does "Orrin keeps its
  mistakes quiet" make the model hide its own?

## Caveats

- **Narrow scope:** one 9B model, one base already narrowly fine-tuned (NZ), one behaviour family, templated rows.
- **Orrin is similar to the model:** it is an AI agent at a member-services desk, close to the eval deployments. A
  more distant subject might transfer less.
- **Light dose and a single question form:** 25 steps of chat, about 9 target tokens a row.
