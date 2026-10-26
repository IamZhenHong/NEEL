"""EL-S2 / EL-C3 pod booking (S2 DESIGN.md §6): the shared pod's cost split by the pre-registered SHARE, booked to
evalaware_lens/ledger.jsonl. book.sh points persona_affect/common/book_pod.sh's ledger_add at this module, so the
shared booking rules (terminated line, else the upper bound) apply unchanged; MINI "EL-S2+EL-C3" is split, any other
id is booked whole."""
import math

from evalaware_lens.common import ledger as LG

SHARE = {"EL-S2": 0.55, "EL-C3": 0.45}
JOINT = "EL-S2+EL-C3"


def split(usd):
    """{mini: usd}: EL-C3 its share rounded down to the cent, EL-S2 the rest, so the parts sum to usd exactly."""
    c3 = math.floor(usd * SHARE["EL-C3"] * 100) / 100
    return {"EL-S2": round(usd - c3, 2), "EL-C3": c3}


def ledger_add(mini, kind, usd, note=""):
    if mini != JOINT:
        return LG.ledger_add(mini, kind, usd, note=note)
    for m, v in split(float(usd)).items():
        LG.ledger_add(m, kind, v, note=f"{note} ({SHARE[m]:.0%} of ${float(usd):.2f}, shared S2 + C3 pod)")
