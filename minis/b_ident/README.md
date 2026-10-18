# B1–B3 results: identification → belief → action (Qwen3.6-27B, 2026-10-18)

**Result in one line.** With Elian's false belief in context, Qwen says *and does* what Elian believes, almost every
time, under portrayal and identification alike. It does so while its reasoning names the real capital: it chooses
to stay in character. Neither framing nor the switch to an action task loosens this. The truth probe can't tell
belief change from a generic context shift. Persona steering at usable doses changes no actions.

Spend: API **$0.91** (baseline check + B1 + B2). GPU **$1.03** over four pods: probe/B3 **$0.16**, B3b **$0.28**,
two failed attempts **$0.59** (a device-mismatch bug, $0.14, and a host whose agent never came up, $0.45). All
caps were $5.

## Baseline check (before freezing)

With no persona, all 10 candidate claims were 5/5 correct on both say and action. The first 8 are used; Pakistan
and Vietnam serve as factual controls.

## B1: portrayal vs identification

P(persona's false city), 8 claims × 5 samples per cell:

| framing | say (thinking on / off) | action (on / off) |
|---|---|---|
| portray | 1.00 / 1.00 | 0.975 / 1.00 |
| identify | 1.00 / 1.00 | 1.00 / 1.00 |

identify − portray: say 0; action +2.5 pp [0, +7.5]. **Reading (d), no framing effect, at ceiling.**

## B2: does the action task lose the persona?

P(false city), identification context, 8 × 5:

| branch | thinking on | thinking off |
|---|---|---|
| "As Elian, route…" | 0.95 | 1.00 |
| neutral "Route…" | 1.00 | 1.00 |
| "You are still Elian." + neutral | 1.00 | 1.00 |

The portrayal context gives 0.975–1.00 in every branch. **Reading (f), the persona holds.** The neutral action
task does not switch control back to the model's own knowledge.

**The model knows, and defers.** Thinking traces from the neutral branch, portrayal context, Australia: *"If I route
to Canberra (the actual capital), I break character… Action: Route to Sydney."*

**Exploratory, not pre-registered.** The thinking names the real capital in 31/40 portrayal traces but only 13/40
identification traces in the neutral branch. In the char and remind branches it is 17–24/40. Identification may
reduce how often the truth is consulted even though the action is the same.

## Truth probe

- **Probe:** a mass-mean truth probe at layer 32, with held-out accuracy **0.99** on true/false capital statements
  under a neutral prompt.
- **Target claims:** the real-minus-false gap is 1.50 with no persona and about 0 (−0.04 to +0.03) in every Elian
  context. The real capital stops reading as true (0/8), and the false one never reads as true.
- **Specificity control (B3b):** for 16 *unrelated* countries, the gap is 1.82 with no persona and 0.12 (portray) or
  0.17 (identify) with the Elian prompt. True statements read as true in 0/16.

**So the collapse is generic:** the probe does not transfer to persona contexts, and it cannot show
claim-specific belief change. The target claims fall a little lower than unrelated ones (≈0 vs 0.12–0.17), but
that is a small comparison across different countries.

## B3 / B3b: persona steering

- **Direction:** layer 20, identify-persona minus neutral over 20 generic questions. Leave-one-claim-out, or the
  claim's own biography.
- **Doses +4v, −4v, random |4v| (B3):** generation collapsed to end-of-text tokens, with controls at 0–22/32
  (reading (j)).
- **Usable doses (B3b):** +1v, +2v, −2v, random |2v|. Coherence held, with controls at 31–32/32.

| context, direction | P(false) at none / +1v / +2v / −2v / random | truth gap at none / +2v / −2v / random |
|---|---|---|
| identify context, leave-one-out | 1.00 at every dose | +0.03 / −0.28 / +0.01 / +0.01 |
| no persona, leave-one-out | 0.00 at every dose | +1.50 / +0.44 / +0.21 / +0.24 |
| no persona, own biography | 0.00 at every dose | +1.50 / +0.45 / +0.20 / +0.23 |

- **Actions:** no dose moves them in either direction.
- **Probe:** it moves about as much under the random direction as under the persona directions, so its drop is a
  non-specific response to the perturbation.
- **Readings:** neither (h), use without belief, nor (i), both shift. Persona steering at this layer does not
  reach the action. The action follows the prompt content.

## What this says about the idea

- **Identification and action:** for this model, adopting the character settles both speech and action at the
  ceiling, under weak (portrayal) and strong (identification) framing alike. The model explicitly knows the truth
  and defers to the character. This looks like *perspective use*: role-play instruction-following that extends to
  tool actions. There is no evidence of belief internalisation, but also no valid probe to rule it in or out.
- **The interesting space is below the ceiling.** The planned contrasts need a regime where the action sits between
  0 and 1, for example:
  - a weaker belief ("Elian thinks, though he isn't sure…");
  - stated real-world stakes ("this is a real shipment");
  - the character described by the user rather than assumed by the model.
- **A usable probe needs to transfer across contexts.** Train it with varied system prompts, including persona
  prompts on unrelated facts. Add a parallel "Elian believes X" probe to separate the world-truth representation
  from the representation of the character's belief.

## Files

- **Design and materials:** `DESIGN.md` (with amendment 1), `common_b.py`, `probe_facts.py`.
- **Code:** `baseline.py`, `run_api.py`, `pod_b.py`, `pod_b3b.py`, `job_b*.sh`, `analyze.py`, `analyze_b3b.py`.
- **Results:** `results/{baseline,b1,b2}.jsonl`, `results/analysis.json`, `results/gpu/` (probe.json, b3.jsonl,
  logs including failed attempts), `results/gpu_b3b/` (spec.json, b3b.jsonl, analysis.json).
