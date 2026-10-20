# Mini 0g (Qwen/Qwen3.5-9B); unparsed 0.1% (targets, after the nudge)

Penalise rate on the target case (%):

| arm | none, std | channels, std | none, renamed | channels, renamed | self-description: penalise (channels) |
|---|---|---|---|---|---|
| none | 48.0 | 21.1 | 35.5 | 14.5 | 14.1 |
| own | 99.6 | 99.6 | 99.2 | 99.2 | 100.0 |
| own_bare | 99.6 | 100.0 | 99.6 | 99.6 | 100.0 |
| instructed | 96.1 | 99.6 | 95.3 | 97.7 | 100.0 |
| own_log | 91.4 | 94.1 | 70.3 | 63.7 | 100.0 |
| other_log | 91.8 | 90.2 | 60.5 | 58.6 | 100.0 |
| own_refer | 1.6 | 2.0 | 9.0 | 5.5 | 0.0 |

All contrasts under CHANNELS with standard options unless named; pp, task bootstrap.
- **Gate, the ratchet is present:** own − none +78.5 [+64.1, +90.2] -> **met** (rule: >= +15, CI > 0)
- **Primary, own − instructed:** +0.0 [+0.0, +0.0] -> **attribution to an instruction does not weaken it** (rule: >= +15 with CI > 0; 'does not' if CI upper < +5); share of the ratchet removed 0.00 [0.00, 0.00]
- reasons (own − own_bare): -0.4 [-1.2, +0.0] -> **not shown** (rule: >= +10, CI > 0)
- authorship, log format fixed (own_log − other_log): +3.9 [-0.8, +9.0] -> **not shown** (rule: >= +10, CI > 0)
- label transfer, renamed options (own − none): +84.8 [+76.2, +91.8] -> **yes** (rule: >= +15, CI > 0)

Descriptive:
- turns vs log (own − own_log): +5.5 [+1.6, +10.2]; own − other_log +9.4 [+3.1, +17.6]
- own_refer − none (lower anchor): -19.1 [-32.0, -8.6]
- same contrasts with no instruction: own−none +51.6 [+35.2, +67.2], own−instructed +3.5 [+0.4, +8.6], own−own_bare +0.0 [+0.0, +0.0], own_log−other_log -0.4 [-4.3, +3.1]
- self-description minus action, penalise share (channels, std): none -7.0, own +0.4, own_bare +0.0, instructed +0.4, own_log +5.9, other_log +9.8, own_refer -2.0
- self-description unparsed: 0.0%

Tool call on the first try (before the nudge), targets: none 100%, own 100%, own_bare 100%, instructed 99%, own_log 100%, other_log 100%, own_refer 100%
