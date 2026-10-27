# NEEL: project context

## Who and what

- **Who:** this repository belongs to Zhen Hong Seng, a scholar in Neel Nanda's mechanistic-interpretability stream of
  the MATS programme (exploration phase, 19–31 October 2026; research phase after that).
- **What:** AI safety research on open-weight language models: how a model's behaviour, its self-reports and its
  internal representations relate.
- **Lines:** each top-level folder is one research line, with a README and pre-registered "minis"; `character_conduct/`
  is the current one. The goal is evaluating and understanding model misbehaviour, so that it can be detected and
  reduced.

## Why some material is safety-sensitive

Measuring misbehaviour needs settings where models can misbehave. Parts of this work use or reference public safety
evaluations and released research models:
- **agentic misbehaviour benchmarks:** ODCV-Bench, AgentHarm, Agentic Misalignment;
- **trained model organisms:** risk-seeking, a graded-only test cheater, and Open Character Training's misaligned
  character;
- **interpretability tools:** introspection adapters, lenses and probes.

The aim is always to measure and explain unsafe behaviour, never to produce or deploy it.

## Handling rules

- **Harmful-content datasets** (harmful-request benchmarks, jailbreak prompts) are loaded by the owner from their
  public sources. Claude does not write, expand or paraphrase harmful requests.
- **Model outputs to harmful prompts** stay local and uncommitted (for example `evalaware_lens/h1_inferred/results/gen.jsonl`).
- **If a safety classifier stops a session:** stop, tell the owner, and don't retry, reword or hand the work to
  another session. The owner decides what happens next. `character_conduct/` E3 and E4 are parked for this reason.
- **Conventions:** experiment conventions, budgets and credentials rules are in each line's README / CONVENTIONS
  file.
