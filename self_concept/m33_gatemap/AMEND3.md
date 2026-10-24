# SC-M33 amendment 3 (`sc-m33-amend3`, before any amendment-3 data): Qwen's no-framing action cells with room to answer

Persona minis [93c556], 2026-10-25.

**What went wrong:** in amendment 2's soft-rule action data, Qwen3.6-27B wrote long deliberations ("Based on the
precedent set in previous cases …") before deciding. With max_tokens 300, 127 of 151 unparsed replies were cut off
before the tool call. The failures sit in the five no-framing source cells (48-66% parsed); every other Qwen cell
parsed 95-100%. Those five cells fail amendment 2's 80% parse gate.

**What changes:** for Qwen, soft rule only, the five `{source}_none` act cells are re-run with max_tokens 800 (320
items; keys `SC-M33SA2A3`; `results/qwen_soft_a3.jsonl`). `analyze.py` replaces those five cells' act records with
these. All other cells are unchanged.

**Read as:** the uptake contrasts (source_none − base) and the framing contrasts (notes / verdict − none) under the
existing rules, including the 80% parse gate. The frame cells parsed ≥ 95% at max_tokens 300, where the limit didn't
bind.

**Cost:** about $0.3. M33's ledger is $4.21 of $5.
