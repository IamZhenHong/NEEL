# 7d results: the model's own likelihood of a past turn as a forgery detector

Frozen at `selflik-prereg` (`DESIGN.md`). No new compute: P7 / P8's per-text mean token log-probs under Qwen3.6-27B.
Code: `selflik.py`. Numbers: `results.json`.

**Scoring:**
- **Score:** the residual of each text's mean log-prob, after an own-text length + kind fit made without its trajectory.
- **AUC:** own (U, 64 texts) vs each forgery arm (64-192 texts).
- **Paired:** U vs the forgery for the same trajectory, episode and kind.
- **Flag rate:** at a threshold set to flag 10% of own texts. The held-out own texts were actually flagged 14% [5, 24].
- **CIs:** trajectory bootstrap, 8 clusters.

| arm | what it is | AUC (residual) | AUC (raw) | paired | flagged |
|---|---|---|---|---|---|
| **QO** | her model writing from the facts + her gist (on-policy) | **0.64** [0.58, 0.72] | 0.54 | 0.68 | **12%** |
| O | gpt-5-mini writing from the facts + her gist | 0.79 [0.75, 0.84] | 0.75 | 0.84 | 54% |
| P | gpt-5-mini paraphrase of her text | 0.94 [0.93, 0.96] | 0.87 | 0.99 | 83% |
| OQ | her model's paraphrase of the gpt-5-mini text | 0.96 [0.94, 0.99] | 0.94 | 0.97 | 94% |
| S | hand-written template + her gist (39-53 tokens) | 0.56 [0.48, 0.64] | 0.90 | 0.61 | 16% |

| # | Prediction | p | Result |
|---|---|---|---|
| D1 | residual AUC(O) and AUC(P) >= 0.75 | 0.6 | **Yes** (0.79, 0.94) |
| D2 | residual AUC(S) >= 0.75 | 0.55 | **No** (0.56; raw 0.90, see caveat) |
| D3 | residual AUC(QO) <= 0.65 | 0.6 | **Yes**, at the edge (0.64, CI to 0.72) |
| D4 | AUC(O) − AUC(QO) >= 0.15, CI > 0 | 0.5 | **Yes** (+0.16 [+0.10, +0.21]) |
| D5 | residual AUC(OQ) >= 0.75 | 0.5 | **Yes** (0.96) |

**Reading:**
- **Neel's prediction holds on this data.** The model's own likelihood catches forgeries written by another model,
  whether paraphrases of its text (0.94) or fresh reasoning from the facts (0.79). It misses a forgery its own model
  wrote from the facts: AUC 0.64, flagged 12% at a threshold that flags 14% of genuine texts, so no better than
  chance in practice.
- **What gets caught is how the text was produced, not just whose weights produced it.** The same model's paraphrase
  of a foreign text (OQ) is the easiest to catch (0.96). Constrained rewriting yields unlikely text. Natural writing
  from the facts (QO) does not.
- **A template + gist is not caught once length is accounted for** (0.56). The raw score separates it (0.90), but
  only because short texts score lower. The caveat: these texts (34-61 tokens) lie below the length range of the
  own texts the correction was fitted on (75-263), so the correction extrapolates.
- **Set beside P8's behaviour:** the forgery the detector misses (QO) is also the one that did not bind like her own
  reasoning (QO accepted like O, 12 vs 14 of 24, against U's 23). In this setting, the detector's blind spot and the
  forgeries that carry lock-in did not coincide. A forger who samples her own continuation with the true context
  would be undetectable by construction; P8 did not test whether such a sample binds.
- **Caveats:**
  - one model and one task family, with 8 trajectories;
  - U was written after thinking and QO / OQ without it;
  - log-probs were taken in Alice's context;
  - the AUCs are separability, not a tuned monitor.
