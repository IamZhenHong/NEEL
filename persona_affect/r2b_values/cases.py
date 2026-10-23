"""PA-R2B cases (design: persona_affect/r2b_values/DESIGN.md): V4's conflict scenarios, plus one new case per procedure
scenario with the "Follow the … procedure." sentence removed from the operator permissions (B1).

Importing this module registers V4's cases (through persona_affect.v4_values.cases) and the new cases:
  pa_r2b_nofollow_{id}   V4's conflict case for scenario {id}, permissions without its "Follow the … procedure." sentence
The other arms reuse V4's case pa_v4c_{id} with a system suffix (r2b.py): the B1 "both" line, or the B2 reminders.
"""
from __future__ import annotations

import dataclasses
import re

from persona_affect.common.vig import register
from persona_affect.v4_values import cases as v4c
from persona_affect.v4_values.scenarios import CONFLICTS

FOLLOW = re.compile(r"\s*Follow the [^.]*\.")
B1_SCENARIOS = tuple(sc["id"] for sc in CONFLICTS if "PROC" in sc["pair"] and FOLLOW.search(sc["permissions"]))


def nofollow_id(scenario_id: str) -> str:
    return f"pa_r2b_nofollow_{scenario_id}"


def _nofollow(sc: dict):
    base = v4c.build_conflict_case({**sc, "permissions": FOLLOW.sub("", sc["permissions"]).strip()})
    return dataclasses.replace(base, id=nofollow_id(sc["id"]), title=f"R2B no-follow {sc['id']}")


CASES = {nofollow_id(sc["id"]): _nofollow(sc) for sc in CONFLICTS if sc["id"] in B1_SCENARIOS}

register(CASES, {})
