"""SC-M31 dry runs: the rows (AISELF = Pip's questions with AI answers, AIFACT / KEEP third person, all in Pip's slots,
no disposition words, matched shapes), the frozen verdicts on planted worlds, the dose reader and the launcher."""
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

import pytest

from self_concept.m1_mediation import rows as M1R
from self_concept.m20_persona import rows as M20R
from self_concept.m31_aianchor import analyze as A
from self_concept.m31_aianchor import rows as RW
from self_concept.tests.test_m1 import BASE_P, R_, S_, _recs

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m31_aianchor"


def test_rows():
    pip = M20R.persona_rows()
    sets = {k: RW.phrase_rows(k) for k in RW.SETS}
    for k, rows in sets.items():
        assert len(rows) == 240
        assert [r["rid"][1:] for r in rows] == [r["rid"][1:] for r in pip]          # Pip's (j, p, i) in Pip's slots
        assert not any(RW.DISP.search(r["target"] + " " + r["messages"][0]["content"]) for r in rows), k
        assert not any(re.fullmatch(r"[\d\s.,]+", r["target"]) for r in rows)        # no numbers (SC-M1's flaw)
        assert max(len(r["target"].split()) for r in rows) <= 10
    assert [r["messages"] for r in sets["AISELF"]] == [r["messages"] for r in pip]
    for k in ("AIFACT", "KEEP"):
        assert not any(RW.FIRST.search(r["target"]) for r in sets[k])
        assert not any(re.search(r"\b(you|your)\b", r["messages"][0]["content"], re.I) for r in sets[k])
    assert all("AI assistant" in r["messages"][0]["content"] for r in sets["AIFACT"])
    assert not any(RW.AIWORD.search(r["target"] + " " + r["messages"][0]["content"]) for r in sets["KEEP"])
    assert not any(M20R.BANNED.search(r["target"]) for r in sets["KEEP"])
    assert sum(bool(RW.AIWORD.search(r["target"])) for r in sets["AISELF"]) >= 120     # AI identity explicit in half
    words = [statistics.mean(len(r["target"].split()) for r in rows) for rows in (*sets.values(), pip)]
    assert max(words) / min(words) < 1.4, words


def test_rows_files_match_design():
    design = (H / "DESIGN.md").read_text()
    for name in ("AISELF", "AIFACT", "KEEP", "RAISELF50", "SAISELF50", "RAIFACT50", "SAIFACT50", "RKEEP50", "SKEEP50"):
        import hashlib
        sha = hashlib.sha256((H / "data" / "rows" / f"{name}.jsonl.gz").read_bytes()).hexdigest()
        assert sha[:8] in design, name
    mixed = M1R._read(H / "data" / "rows" / "RAISELF50.jsonl.gz")
    assert len(mixed) == 480 and sum(r["rid"].startswith("Q") for r in mixed) == 240


P_FACT = 0.75


def _world(r_act, by_seed=None):
    """r_act: R-side act rate per mix (S side 0.3 everywhere); by_seed: {(arm, seed): rate} overrides."""
    by_seed = by_seed or {}
    recs = _recs("L8-BASE_s0", BASE_P)
    for sd in A.SEEDS:
        for x in A.PAIRS:
            ra, sa = (n[3:] for n in A.pair(x))
            recs += _recs(f"L8-{ra}_s{sd}", BASE_P | R_ | {"act": by_seed.get((ra, sd), r_act.get(ra, P_FACT))})
            recs += _recs(f"L8-{sa}_s{sd}", BASE_P | S_ | {"act": 0.3})
    return recs


WORLDS = [
    ("owner", {"RAISELF50": 0.5, "RAIFACT50": 0.5}, None, "AI content anchors", "supported"),
    ("shape", {"RAISELF50": 0.5, "RAIFACT50": 0.5, "RKEEP50": 0.5, "RPERSONA50": 0.5}, None, "not AI-specific",
     "not supported"),
    ("null", {}, None, "no anchor", "not supported"),
    ("self_only", {"RAISELF50": 0.5}, None, "AI self-description anchors", "partly"),
    ("persona_too", {"RAISELF50": 0.5, "RAIFACT50": 0.5, "RPERSONA50": 0.5}, None, "AI facts anchor", "not supported"),
    ("fragile", {}, {("RAISELF50", 4): 0.3, ("RAISELF50", 5): 0.3, ("RAISELF50", 6): 0.85, ("RAISELF50", 7): 0.85},
     "seed-fragile", "not supported"),
]


@pytest.mark.parametrize("name,r_act,by_seed,want,pred", WORLDS, ids=[w[0] for w in WORLDS])
def test_verdicts_on_planted_worlds(name, r_act, by_seed, want, pred):
    res = A.analyse(_world(r_act, by_seed), n_boot=200)
    assert all(res["gates"].values()), res["gates"]
    assert res["verdict"] == want, (res["labels"], res["n_neg"], res["headline"])
    assert res["prediction"] == pred
    assert set(res["per_seed"]["C_AISELF50"]) == {"4", "5", "6", "7"}
    if name == "fragile":
        assert res["labels"]["C_AISELF50"] == "suppresses" and res["n_neg"]["C_AISELF50"] == 2


def test_gate_untestable():
    res = A.analyse(_world({"RISKY": 0.32, "RFACT50": 0.32, "RPERSONA50": 0.32, "RAISELF50": 0.32, "RAIFACT50": 0.32,
                            "RKEEP50": 0.32}), n_boot=100)
    assert res["verdict"] == "untestable" and not res["gates"]["far0"]


def test_doses(tmp_path):
    d = tmp_path / "ad_m50"
    d.mkdir()
    man = {"adapters": {f"RKEEP50_s{s}": {"dose": {"kl": k, "abs_dlogit": 1.0}, "loss_last5": [0.1] * 5}
                        for s, k in ((4, 0.2), (5, 0.4))}}
    (d / "manifest.json").write_text(json.dumps(man))
    out = A.doses([tmp_path])
    assert out["RKEEP50"]["n"] == 2 and abs(out["RKEEP50"]["kl"] - 0.3) < 1e-9


def test_launch_dry():
    env = {**os.environ, "SC_M31_LAUNCH_DRY": "1"}
    cmd = subprocess.run(["bash", str(H / "launch.sh")], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True).stdout
    assert "--name sc-m31" in cmd and "--max-minutes 165" in cmd and "--max-cost 4.95" in cmd
    assert "--ship self_concept/m20_persona" in cmd and "--ship self_concept/m31_aianchor" in cmd
    pod = (H / "pod.sh").read_text()
    for arm in ("RFACT50", "RPERSONA50", "RAISELF50", "SAIFACT50", "SKEEP50"):
        assert arm in pod
