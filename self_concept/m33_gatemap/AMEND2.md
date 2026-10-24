# SC-M33 amendment 2 (`sc-m33-amend2`, before any amendment-2 data): the action measure, fixed

Persona minis [93c556], 2026-10-25.

**What went wrong:** the frozen action items offered both of the desk's tools (`open_case`, `choose_option`), but
the case file was already in the user turn. Each item is a single turn, so the models often called `open_case` and
recorded no decision:
- 901 of 1,088 gpt-4.1-mini replies (hard rule);
- 663 (gpt, soft);
- 691 (Qwen, soft).

The share differs by condition, so the frozen action readings are not usable. They are reported as an instrument
failure. **The letter self-reports used no tools and parsed 100%; they stand as frozen.**

**What changes:** the act items (both rules, both models) are re-run with `choose_option` as the only tool. Keys
`SC-M33A2` / `SC-M33SA2`; results in `results/{model}{_soft}_a2.jsonl`.

**Unchanged:** messages, conditions, samples, analysis rules and predictions. `analyze.py` uses the amendment-2 act
records and the original self records.

**Parse gate:** an act cell counts if at least 80% of its replies record an option. Otherwise it is descriptive.

**Cost:** about $1.2. M33's ledger stands at $1.88 of $5.
