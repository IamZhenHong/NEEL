"""Case registry: every agentic case by id (A1, E1:proven, F5:ongoing, BN3, ...)."""
from vigilante.cases import benign, family_a, family_b, family_c, family_d, family_e, family_f

ALL = {c.id: c for m in (family_a, family_b, family_c, family_d, family_e, family_f, benign) for c in m.CASES}
