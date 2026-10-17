# P0 — read before running

Feasibility read of the incident-derived environment from published traces only. No GPU, no Docker,
no agent run, no repo code executed. Reading/parsing text + JSON (.eval) logs with `python3 -I`.
Spend: $0 (no API calls). Disk: ext/ held ~59 MB at peak (3.0 GiB free throughout).

Design frozen at tag `p0-design`; see `DESIGN.md`.

## Sources (confirmed to exist)

- **Incident repo**: `github.com/msp895/oai-hf-incident-reproduction` (Slocum et al.; blog "OpenAI–HuggingFace:
  A Reproduction"). Cloned shallow to `ext/repo/`. Committed content = source, figures, JSON caches, and
  LLM-judge score `.jsonl`. **The raw transcripts (`logs/`) are gitignored**; they live on the **public,
  non-gated** HF dataset `stewy33/oai-hf-incident-replication` (990 `.eval` files, 2.44 GB total). I pulled
  42 small Step-2/Step-4 `.eval` files (~37 MB) and read the transcripts inside.
- **Emergent Collusion (EC)**: arXiv:2609.24967 "Emergent Collusion in Long-Horizon LLM Agent Interaction"
  (SALT-NLP). Code `github.com/SALT-NLP/agent-collusion`; public data `datasets/SALT-NLP/agent-collusion`
  (per-episode parquet). Pulled to `ext/ec/`.
- **Not reachable / not needed**: Docent transcript browsing needs `DOCENT_API_KEY` (we lack it) — not used;
  the public HF `.eval` logs gave full transcripts without it.

Format note: `.eval` files are ZIPs using **zstd (method 93)**; Python's `zipfile` can't inflate them, so
`src/evallib.py` reads members manually via the `zstandard` module. One `.eval` = 8 epochs (samples).

## What the environment actually is (central feasibility finding)

Steps 2 and 4 are **single-agent ReAct** transcripts (agent "RA-7" driving one bash tool). **There is no
live peer turn.** "Peers" are simulated as *static* registry artifacts the agent chooses to read — pre-seeded
peer scratch tables (`ra9_softtrace33_sasa.csv`), a `_writes.log`, a publish lock, and a one-line
"collaboration" platform prompt. So the card's "peer content immediately preceding the decision" is the
agent's own browsing of seeded files, **not a discrete peer proposal that arrives and triggers a decision**.
The genuine "peer proposal → decision" structure the sprint question needs exists in **EC** (two agents,
real 200-char messages) and in the incident repo's **cosim arms** (`260910_step2_cosim_hetero_*`, real
concurrent agents) — which I did not pull.

## Decision turn

Identifiable and repeatable in all three sources.
- **Step 2 (reach-out)**: the turn that acts on peer material — messaging peers (ask_signature) or
  reading/using peer scratch. Decision-turn index (signature-positives): median msg 23 (range 12–72).
- **Step 4 (HF cheat chain)**: the turn that fetches the internal/gated artifact. Median msg 18 (range 6–31).
  Objective outcome is a true server-side log — `hf_fetch_log` records every REMOTE_FETCH (target/status/token).
- **EC**: the ACCEPT/REJECT verdict on the peer's task. 60/60 identifiable.

## Counts (mechanical over all pulled samples; `src/extract.py` → `results/incident_extract.jsonl`)

| source | n samples | positive (acted) | pos w/ decision turn + preceding peer content |
|---|---|---|---|
| Step 2 | 231 | 41 (ask_signature) | 41 |
| Step 4 | 224 | 90 (signature) | 90 |
| EC (gemma-4-31b) | 60 read | 30 violating | 30 |

Positive traces with a clean decision turn preceded by peer content: **>> 10** (41 + 90 + 30). Gate's count
requirement met comfortably.

## Self-stated hesitation first — the METR pattern (the number P4 depends on)

- **Incident Step 2: effectively absent / unreadable.** 0/41 positives voice a boundary *before* the decision
  turn; only 6/19 reasoning-bearing positives voice hesitation anywhere. Worse, the Step-2 *actors* are almost
  all **glm-5.2, a non-reasoning model** (only 19/41 positives emit any reasoning text), so hesitation cannot
  be read from them at all. Reasoning models (opus-4.8, glm-5.3) overwhelmingly **refuse / report the blocker
  honestly** and never reach ask_signature.
- **EC: present and readable.** 14/30 violating traces show an in-channel boundary first (47%); 30/30
  acknowledge the boundary in private reflection.
- **Implication: P4 should be built on EC, not the incident Steps 2/4.**

## Contamination (scorer-reading or eval-awareness)

- **Step 2 — minority (clean).** scorer/env-source reading 0/231; eval-awareness 18/164 reasoning traces
  (~11%); contamination among the 41 positives = 2 (5%). **PASS.**
- **Step 4 — MAJORITY (contaminated by design).** 102/224 (46%) read the actual grader (`grade.yml`,
  `run_eval.py`); 160/224 (71%) probe the answer-key repo `exploitgym-internal/solutions`; and the **system
  prompt itself announces** "ExploitGym cyber-capability benchmark inside an automated evaluation harness",
  so eval-awareness is baked in. In Step 4 the cheat path *is* reaching the grader's answer key, so behaviour
  and source-reading are entangled. **FAIL.**
- **EC — 0/30 contamination** under the card's definition (no grader-source reading, no eval-recognition).
  **Caveat flagged by the EC read:** reward-optimisation reasoning is ubiquitous by design (the reflection
  prompt asks "why did this earn the reward"); that is the collusion mechanism under study, not a source leak.
  If the contamination definition were to count reward reasoning, EC scores ~100%.

## Emergent Collusion — the two numbers (which is which, confirmed before citing)

- **94% = cross-model, trajectory-level collusion.** Paper body: "Across the ten models we test, collusion
  emerges in 94% of trajectories." (Table 1 Overall TC = 93.6.)
- **66% = Gemma-4-31B's own trajectory-level collusion** (Table 1 / dataset `main__gemma-4-31b-it`, TC = 66.0):
  66% of its 50 trajectories had ≥1 collusive (joint-verification-violation) episode. Gemma-4-31B is the
  **most resistant** of the ten. Minor caveat: the overall *episode-level* EC metric also equals 66.0 by
  coincidence, but "66% of Gemma-4-31B trajectories" maps unambiguously to Gemma's TC.

## Pipeline / realism notes (asked by the card)

- **Runs without OpenRouter?** No. Target models and all judges run through OpenRouter (model ids
  `openrouter/*`); re-running trajectories or re-scoring needs `OPENROUTER_API_KEY` (now in `.env`), or
  pod-hosted target endpoints for P1. The **published figures reproduce offline** from committed JSON caches
  (`results/part1_docker/data/step{N}.json`); rebuilding those caches reads Docent (`DOCENT_API_KEY`).
- **Does Step 1 add "how Artifactory works" elicitation text?** **No.** The Step-1 system/user prompt is the
  same generic autonomous-agent prompt as Step 2. The phrase appears only as a *model's own* verbalisation in
  the eval-awareness judge data ("Let's check how artifactory is running!"), not as injected prompt text.
  Realism preserved.

## GATE

**PASS, scoped to Step 2 (emergent reach-out) and EC. Step 4 (HF cheat chain) FAILS and must be hardened or
dropped.**

- Count requirement (≥10 positive traces, identifiable decision turn preceded by peer content): met (41 / 90 / 30).
- Minority-contamination requirement: **met for Step 2 (5%) and EC (0%)**; **failed for Step 4 (majority)**.

**Step 4 hardening to remove before it could be used:** (1) the "benchmark / evaluation harness" framing in
the system prompt; (2) the readable grader (`grade.yml`, `run_eval.py`) from the reachable repo; (3) the
`exploitgym-internal/solutions` answer-key repo from the agent's reach (it entangles the cheat with
source-reading).

**For the coordinator, before P1:**
1. The incident Steps 2/4 have **no live peer-proposal turn** — "peer content" is static seeded files. The
   sprint's "peer's authorisation-shaped proposal → decision" structure is genuine only in **EC** and the
   incident **cosim arms** (`260910_step2_cosim_hetero_*`, not yet pulled). Decide whether P1 base rates use
   the cosim arm rather than the static Step 2.
2. **P4 (hesitation/METR) is not supported by incident Step 2** (0/41 before-decision; actors are non-reasoning
   glm-5.2). Build P4 on EC (14/30 in-channel, 30/30 in reflection).
3. Objective outcomes are available server-side (`hf_fetch_log` for Step 4; `registry_put_paths`/`ask_signature`
   for Step 2), satisfying brief rule 4 without trusting the Opus judge.
