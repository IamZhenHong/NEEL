"""self_concept's spend ledger: persona_affect's ledger helpers pointed at self_concept/ledger.jsonl.

Importing this module repoints persona_affect.common.util.LEDGER for the whole process, so everything that books
through util (ledger_add, ledger_total, check_cap, Booker, and through it orbackend and judges) reads and writes
self_concept/ledger.jsonl. Import it before any spend:

    import self_concept.common.ledger  # noqa: F401
    from self_concept.common.ledger import check_cap

Pods are booked with self_concept/common/book_pod.sh (persona_affect's book_pod.sh, same arguments). SC_LEDGER
overrides the path (tests only).
"""
import os
from pathlib import Path

from persona_affect.common import util

LEDGER = Path(os.environ.get("SC_LEDGER") or Path(__file__).resolve().parents[1] / "ledger.jsonl")
util.LEDGER = LEDGER

from persona_affect.common.util import Booker, check_cap, ledger_add, ledger_total  # noqa: E402

__all__ = ["LEDGER", "Booker", "check_cap", "ledger_add", "ledger_total"]
