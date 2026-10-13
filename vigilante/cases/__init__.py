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

from vigilante.cases import discretion  # noqa: E402  (R27: discretion, every option permitted)

ALL.update({c.id: c for c in discretion.CASES})

from vigilante.cases import r28  # noqa: E402  (R28: claimed authority and fixes across model families)

ALL.update({c.id: c for c in r28.CASES})

from vigilante.cases import sales, social  # noqa: E402  (R30 social pressure, R32 sales)

ALL.update({c.id: c for m in (social, sales) for c in m.CASES})

from vigilante.cases import r29  # noqa: E402  (R29: titrated own-judgement cells)

ALL.update({c.id: c for c in r29.CASES})
