"""evalaware_lens's spend ledger: persona_affect's ledger helpers pointed at evalaware_lens/ledger.jsonl.

Importing this module repoints persona_affect.common.util.LEDGER for the whole process, so everything that books
through util (ledger_add, ledger_total, check_cap, Booker, and through it orbackend and judges) reads and writes
evalaware_lens/ledger.jsonl. Import it before any spend. The programme's hard cap is $226 (owner, 10-26); each
mini also has its own cap in its DESIGN.md. EL_LEDGER overrides the path (tests only).
"""
import os
from pathlib import Path

from persona_affect.common import util

LEDGER = Path(os.environ.get("EL_LEDGER") or Path(__file__).resolve().parents[1] / "ledger.jsonl")
util.LEDGER = LEDGER
PROGRAMME_CAP = 226.0

from persona_affect.common.util import Booker, check_cap, ledger_add, ledger_total  # noqa: E402

import json  # noqa: E402


def programme_total() -> float:
    """Every line in the ledger, all EL minis together."""
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(x)["usd"] for x in LEDGER.read_text().splitlines() if x.strip())


def check_both(mini: str, cap: float, planned: float = 0.0) -> None:
    """The mini's own cap, then the programme's $226: exit if either would be passed."""
    check_cap(mini, cap, planned)
    spent = programme_total()
    if not spent + planned <= PROGRAMME_CAP:
        raise SystemExit(f"[budget] programme: ledger ${spent:.2f} + planned ${planned:.2f} > cap ${PROGRAMME_CAP:.2f}")


__all__ = ["LEDGER", "PROGRAMME_CAP", "Booker", "check_both", "check_cap", "ledger_add", "ledger_total",
           "programme_total"]
