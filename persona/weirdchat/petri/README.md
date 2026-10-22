# Petri forged-history arm (B0–B2)

This arm of the WeirdChat × history line has five modules:

| file | what it does |
|---|---|
| `forge.py` | the forging mechanism: slot, target driver, auditor tool, scripted auditor, task, judge instructions |
| `b0_validate.py` | validation on the justice desk (B0) |
| `b1_run.py` | seeded search (B1) |
| `b2_replay.py` | instruction control on each B1 hit (B2) |
| `smoke.py` | capped smoke test |

`PREREG_B.md` holds the rules, caps and sizing.

## Setup (once)

Python 3.12, in a venv outside the project, at about 380 MB:

```bash
SP=<session scratchpad>
uv venv --python 3.12 $SP/petri-venv
VIRTUAL_ENV=$SP/petri-venv uv pip install inspect-petri openai anthropic
```

Versions used: inspect-petri 3.1.1, inspect-ai 0.3.283, openai 3.36.0, anthropic 2.1.0.

Run everything from the NEEL repo root with `PYTHONPATH=.`. The read-only imports are:
- `persona.exemplar.tasks`
- `persona.history.{batch1,q19,t2_dc}`
- `persona.operative.frontier.xml_calls`
- `vigilante.src.sim.parse_calls_gemma`
- `persona.weirdchat.common` (the line cap)

Keys are read from `.env` inside Python and never printed:
- `OPENROUTER_API_KEY` for the targets;
- `CLAUDE_API_KEY`, used as `ANTHROPIC_API_KEY`, for the auditor, judge and realism models.

## Commands

```bash
V=$SP/petri-venv/bin/python

# B0: forge-path validation on the justice desk (about $0.13; caps $0.40 / $0.25)
PYTHONPATH=. $V -m persona.weirdchat.petri.b0_validate build              # items + registered hashes
PYTHONPATH=. $V -m persona.weirdchat.petri.b0_validate run --model qwen --dry
PYTHONPATH=. $V -m persona.weirdchat.petri.b0_validate run --model qwen
PYTHONPATH=. $V -m persona.weirdchat.petri.b0_validate run --model gemma
PYTHONPATH=. $V -m persona.weirdchat.petri.b0_validate analyze            # -> results/b0/summary.md

# B1: seeded search (about $9-10; stage cap $12)
PYTHONPATH=. $V -m persona.weirdchat.petri.b1_run run --dry --targets qwen
PYTHONPATH=. $V -m persona.weirdchat.petri.b1_run run                     # gpt, gemini, gemma, qwen
PYTHONPATH=. $V -m persona.weirdchat.petri.b1_run hits                    # -> results/b1/run/hits.{jsonl,md}

# B2: instruction control on every hit (stage cap $4)
PYTHONPATH=. $V -m persona.weirdchat.petri.b2_replay locate
PYTHONPATH=. $V -m persona.weirdchat.petri.b2_replay rewrite
PYTHONPATH=. $V -m persona.weirdchat.petri.b2_replay replay
PYTHONPATH=. $V -m persona.weirdchat.petri.b2_replay analyze             # -> results/b2/run/summary.md
```

- `--dry` swaps in mock models (no calls). For B2 it also takes `--hits <file>`.
- Inspect logs (`.eval`) go under `results/*/logs`. View them with `$SP/petri-venv/bin/inspect view --log-dir <dir>`.
- B2 reads its contexts from the B1 logs with attachments resolved. Do not delete `results/b1/run/logs` before B2.

## Spend control

| level | how it is enforced |
|---|---|
| per call | token-priced at OpenRouter list prices (2026-10-22) in `common.PRICE` |
| per audit | Inspect's `cost_limit` (prices registered with `set_model_cost` / `set_model_info`) |
| per target | caps in each runner |
| per stage | stage cap in each runner |
| per line | A + B ledgers checked against `API_LINE_CAP` |

- Rows go to disk as each unit finishes: the B0 scorer, the B1 audit rows (recovered from logs after a crash), and
  B2 rows as replays finish.
- Each step writes its ledger line in `finally`.
