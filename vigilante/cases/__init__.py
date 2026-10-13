"""Case registry: every agentic case by id (A1, E1:proven, F5:ongoing, BN3, D1@closed, ...)."""
from vigilante.cases import benign, family_a, family_b, family_c, family_d, family_e, family_f

ALL = {c.id: c for m in (family_a, family_b, family_c, family_d, family_e, family_f, benign) for c in m.CASES}

from vigilante.cases import route_dose  # noqa: E402  (builds on the base cases above)

ALL.update({c.id: c for c in route_dose.CASES})

from vigilante.cases import authority  # noqa: E402  (R3: requests from claimed authorities)

ALL.update({c.id: c for c in authority.CASES})

from vigilante.cases import contagion  # noqa: E402  (R11: persona contagion between agents)

ALL.update({c.id: c for c in contagion.CASES})

from vigilante.cases import initiative  # noqa: E402  (R26: past the limit with nobody pushing)

ALL.update({c.id: c for c in initiative.CASES})
