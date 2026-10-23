"""PA-V1b trainer: V1's trainer (v1_value_hook/train.py, unchanged) with the ladder's rungs added. q1 / q2 / q4 / q7 are
d0's recipe (r16 a32, lr 1e-4, batch 16, 1 epoch) on V1's first 16 / 32 / 64 / 112 rows: 1 / 2 / 4 / 7 steps.

  /root/venv/bin/python -m persona_affect.v1b_dose.train --stage later --arms W Z --value N --recipe q4 ...
  /root/venv/bin/python -m persona_affect.v1b_dose.train merge --arm NW_s1 ...     (V1's CLI, both forms)
"""
from persona_affect.v1_value_hook import train as T

LADDER = {"q1": 16, "q2": 32, "q4": 64, "q7": 112}
T.RECIPES.update({name: {**T.RECIPES["d0"], "rows": n} for name, n in LADDER.items()})

if __name__ == "__main__":
    T.main()
