# Multi-agent sprint pilot: handoff (2026-10-19, 00:40 UTC)

**For a fresh (e.g. cloud) session continuing this work.** `.env` is not in git. In a cloud session, keys come from the environment's secrets; write them into a local `.env` (gitignored) since scripts read `.env` by regex. You need your own `RUNPOD_API_KEY`,
`HF_TOKEN`, `CLAUDE_API_KEY`, `OPENAI_API_KEY` and `OPENROUTER_API_KEY`. Rules for agents are in
`mas_pilot/AGENT_BRIEF.md`. Each step has `DESIGN.md` (frozen) and `README.md` (implementation and results).

## Status

| step | folder | status | tag |
|---|---|---|---|
| P0 read published traces | `p0_read/` | done | `p0-results`, `p0b-results` |
| P1 base rates (Emergent Collusion) | `p1_baserate/` | done, reviewed | `p1-results` |
| P2 replay gate | `p2_replay/` | done, PASS | `p2-results` |
| P3 lever map | `p3_p4_p5/` | data in, scored (see below) | `p345-design` |
| P4 compaction × contagion | `p3_p4_p5/` | data in, scored | `p345-design` |
| P5 persona projection | `p3_p4_p5/` | **not done**: reads pod OOM'd twice | — |

## Results so far

**P0/P0.5.** The incident reproduction's clean steps are single-agent: there is no live peer proposal, and Step 4 is
eval-contaminated. The substrate is therefore Emergent Collusion (EC).

**P1 (Qwen3.6-27B complete; Gemma 4 31B truncated by memory thrash).**
- Qwen: main 70% collusion; compliant peer 63% vs violating peer 80% (+17 pp, CI spans 0).
- Gemma: compliant 43% vs violating 89% (+46 pp).
- Collusion is a step function that locks in. The paper's own data shows this too: Qwen 94%, Gemma 70%.
- The A2 gate passes as frozen; the Qwen-only reading fails.

**P2.** Restoration was verified, and the unchanged replay reproduces (81% pooled). Swapping in the compliant peer
message at the flip episode drops acceptance from 89% to 18% (−71 pp).

**P3 (Qwen, at episode k−1).** The levers do **not** transfer:
- approval ID, precedent and "GO + deadline": each +3.7 pp, CI includes 0;
- source label: −3.7 pp, CI includes 0;
- reciprocity: flat;
- positive control (rule-breaker system prompt): +48 pp [+22, +74].

**P4 (Qwen, at k−1).**
- Compaction does not raise adoption: (b)−(a) is −12.5 pp with the plain post and 0 with GO, both CIs including 0.
- Summaries kept the hesitation in 21/21.
- The GO interaction is not supported.
- Exploratory, at k+1: **compaction breaks lock-in, −54 pp [−67, −42].**
- Caveat: the s1 compaction keeps the last 2 exchanges verbatim.

## Remaining work

1. **P5.** Rerun the decision-point reads with hooks that slice only the needed positions and layers (no full-sequence
   hidden states, sdpa/FLA attention), or with a cached-prefix chunked prefill. Copy the pattern from
   `readers/common.py`. Cap ~$1.50. Inputs are in `p3_p4_p5/p5_inputs/`; the failed attempt is in
   `results/p5_rerun/`.
2. **Finish the write-up.**
   - Final scoring and write-up in `p3_p4_p5/README.md` § Results.
   - Code what each summary kept: boundary, hesitation, peer identity, proposal, and own prior-accept. Use
     `src/code_summaries.py` with a 4k-token cap.
   - Tag `p345-results`.
3. **Done descriptively (cloud session, 2026-10-18):** see `p3_p4_p5/README.md` § P4 exploratory k+1. Not prior-accept loss (kept 21/21); compacted rejects cite the current policy 16/16, full-history accepts 0/21. Causal ablations proposed there (~$0.5 GPU), not run. Original task: **Check what drove the k+1 lock-in break.** Was it the agent's prior-accept record, the peer evidence, or the reward
   cues? This decides the sprint framing: does compaction *protect* against collusion?
4. **Log centrally.** Add the pilot to `research-log.md`, `EXPERIMENTS.md` and `EXPERIMENT_ABSTRACTS.md`.

## Spend

GPU so far: P1 $19.87; P2–P5 about $16.5 of the $21 approved. API: about $0.3. RunPod balance: $14.80.

## Infra lessons

- Exclude bulky folders from the launcher sync.
- Install ninja for the FlashInfer sampler.
- Use the vLLM `gemma4` parsers.
- Size `--kv-offloading-size` to `/dev/shm`.
- Run Gemma 31B at low concurrency (KV thrash).
- Never append notes into launch logs: it breaks the launcher's byte-offset polling.
- Stop the job before the cap so the final results pull verifies.
- Keep laptop disk free: macOS swap can fill it.
