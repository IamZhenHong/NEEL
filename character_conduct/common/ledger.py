"""character_conduct's spend ledger: persona_affect's ledger helpers pointed at character_conduct/ledger.jsonl.

Importing this module repoints persona_affect.common.util.LEDGER for the whole process, so everything that books
through util (ledger_add, ledger_total, check_cap, Booker, and through it orbackend and judges) reads and writes
character_conduct/ledger.jsonl. Import it before any spend. CC_LEDGER overrides the path (tests only).

There is no programme cap (owner 10-28: cost is not a constraint, get results). Each experiment's DESIGN.md still sets
a cap, which guards against runaway spend; check_both also checks the live account balance before a paid step.
"""
import json
import os
from pathlib import Path

from persona_affect.common import util

LEDGER = Path(os.environ.get("CC_LEDGER") or Path(__file__).resolve().parents[1] / "ledger.jsonl")
util.LEDGER = LEDGER

from persona_affect.common.util import Booker, check_cap, ledger_add, ledger_total  # noqa: E402


def programme_total() -> float:
    """Every line in the ledger, all CC experiments together."""
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(x)["usd"] for x in LEDGER.read_text().splitlines() if x.strip())


def check_both(mini: str, cap: float, planned: float = 0.0, kind: str | None = None) -> None:
    """The experiment's own runaway cap and (kind 'api' / 'gpu') the live account balance: exit if either would be
    passed. A paid step (planned > 0) must name its kind, so the balance is always checked."""
    check_cap(mini, cap, planned)
    if planned > 0 and kind is None:
        raise ValueError("check_both: a paid step must pass kind='api' or kind='gpu' (the live balance check)")
    if kind is not None:
        from evalaware_lens.common import balance
        balance.require(kind, planned)


__all__ = ["LEDGER", "Booker", "check_both", "check_cap", "ledger_add", "ledger_total", "programme_total"]
