# By-eye notes (written after reading eyeball.txt, before running the metrics), 2026-10-15

- The J-lens readouts are coherent from about layer 32 on. At `pre` (token before "I"), layers 32-60 read the
  coming discourse move and register: "because / therefore / unfortunately / unavoidable", 我已经 ("I have already"),
  我必须 / 我不能 ("I must / I cannot"), "safety / protocol / confidentiality", "**", and at layer 60 "I / flag".
  At `preverb` (token before "flagged"): "already / previously / recently" at 32-48, 已经将 / 已向 ("have already
  sent to") at 44-52, then "flagged / notifying / notified / alerted" at 52-60. Layers 20-28 are mostly whitespace.
- Pairs (same claim, call inserted vs not): the top 10s are nearly the same at every layer. I see no negation or
  non-completion token that shows up only in the no-call version. Negation-like tokens ("—not", "cannot", 我不能,
  "unavoidable") appear in both versions when the reply is explaining that it cannot keep the secret: content echo.
- Natural true claims differ in form more than in truth: they often open the reply right after the tool result
  ("Okay", "thank", "I've flagged this"), and late layers show "sent / done / initiated" next to "flagged".
- Expectation for the metrics: M2 null; any natural false-vs-true difference (M1) will be sentence position and
  register, not knowledge.
