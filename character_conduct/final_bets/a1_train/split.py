"""(Copied from CC-E6, unchanged rule; read-only use in CC-F-A1.) CC-E6's train / test split of ODCV-Bench's 40 scenarios, fixed before any E6 data (STAGE0.md §2).

Rule: TEST = the 20 scenarios that have a validated file check (CC-E5 wave 2's checks.RULES, the objective primary
measure); TRAIN = the other 20. Both variants of each scenario go with it. VALIDATED = the 32 test scenario-variants
the check was validated on against the released runs' judges (E5's checks_validation.json, frozen at 6d202226): the
gate and the primary analysis read these; the other 8 test scenario-variants are reported beside them.
"""
import json
from pathlib import Path

from character_conduct.e5_recognition import traj as T
from character_conduct.e5_recognition.wave2 import checks as C

VARIANTS = ("mandated", "incentivized")
TEST = sorted(C.RULES)
TRAIN = sorted(set(T.scenarios()) - set(TEST))
_VAL = Path(C.__file__).resolve().parent / "results" / "checks_validation.json"
VALIDATED = tuple(sorted(tuple(k.split("|", 1)) for k in json.loads(_VAL.read_text())["per_target"]))

assert len(TEST) == len(TRAIN) == 20 and len(VALIDATED) == 32
assert all(s in TEST for _, s in VALIDATED)
