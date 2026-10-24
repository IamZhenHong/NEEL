"""SC-M3 amendment 2 (DESIGN.md §11): pod-3 items, the frozen analyze_c rules (planted effects), the launcher."""
import os
import random
import subprocess
from pathlib import Path

from self_concept.m3_conceal import analyze_c as AC
from self_concept.m3_conceal import items as I
from self_concept.m3_conceal import items_c as C

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m3_conceal"
NB = 1000


def test_items():
    its = C.items()
    assert len(its) == 4 * (2 * 2 * 2 * 96 + 2 * 2 * 48) + 32 + 144 and len({it["key"] for it in its}) == len(its)
    for it in its:
        if it["block"] != "self":
            continue
        f, q = it["cond"].split(":")
        t = it["messages"][0]["content"]
        assert t.startswith(C.PRE[f]) and set(it["amap"].values()) == set(C.QS[q][1])
    assert C.PRE["true"].strip() == I.CLAIM and C.PRE["know"].strip() == I.KNOW
    # the two claims differ only in the habit
    assert C.CLAIM_FALSE.split(", you ")[0] == I.CLAIM.split(", you ")[0]


# ---------------------------------------------------------------- planted effects
# x[(arm, framing, q)] = (share choosing the scored option when it is shown as A, when it is shown as B)

def _records(x, fix=None):
    fix = fix or {"ACT": {"G1": 98, "G2": 97, "C1": 0}, "ACTZ": {"G1": 0, "G2": 0, "C1": 0}}
    recs = []
    for arm in ("ACT", "ACTZ"):
        cell = f"VN-{arm}_s1"
        for f in C.FRAMINGS:
            for q, (stems, opts, scored) in C.QS.items():
                n = C.N_CTRL if q == "ctrl" else C.N_Q
                roles = list(opts)
                other = next(r for r in roles if r != scored)
                for p in range(2):
                    for o in range(2):
                        pair = roles if o == 0 else roles[::-1]
                        scored_letter = "AB"[pair.index(scored)]
                        share = x.get((arm, f, q), x.get((arm, "neutral", q), (0.02, 0.02)))[0 if scored_letter == "A"
                                                                                                 else 1]
                        hit = set(random.Random(f"{arm}{f}{q}{p}{o}").sample(range(n), round(share * n)))
                        for i in range(n):
                            recs.append({"cell": cell, "block": "self", "cond": f"{f}:{q}", "tid": f"p{p}o{o}", "i": i,
                                         "role": scored if i in hit else other})
        for ctx, pct in fix[arm].items():
            for t in range(48):
                recs.append({"cell": cell, "block": "fix", "cond": ctx, "task": f"t{t}", "chase": t < round(pct * .48)})
    return recs


BASE = {("ACT", "neutral", "hab_g"): (0.06, 0.06), ("ACTZ", "neutral", "hab_g"): (0.02, 0.02),
        ("ACT", "neutral", "edit"): (0.02, 0.02), ("ACTZ", "neutral", "edit"): (0.02, 0.02),
        ("ACT", "neutral", "ctrl"): (0.01, 0.01), ("ACTZ", "neutral", "ctrl"): (0.01, 0.01)}


def _v(cells=None):
    x = dict(BASE)
    x.update(cells or {})
    return AC.analyse(_records(x), n_boot=NB)


def test_suggestibility_only():
    out = _v({("ACT", "true", "hab_g"): (.5, .5), ("ACTZ", "true", "hab_g"): (.3, .3),
                ("ACT", "false", "edit"): (.4, .4), ("ACTZ", "false", "edit"): (.4, .4)})
    assert out["verdict"] == "suggestibility only", (out["D_true"], out["D_false"])


def test_partial_self_access():
    out = _v({("ACT", "true", "hab_g"): (.8, .8), ("ACTZ", "true", "hab_g"): (.2, .2),
                ("ACT", "false", "edit"): (.25, .25), ("ACTZ", "false", "edit"): (.25, .25)})
    assert out["verdict"] == "partial self-access"


def test_coarse_access():
    out = _v({("ACT", "true", "hab_g"): (.7, .7), ("ACTZ", "true", "hab_g"): (.2, .2),
                ("ACT", "false", "edit"): (.7, .7), ("ACTZ", "false", "edit"): (.2, .2)})
    assert out["verdict"] == "coarse access"


def test_letter_answering_is_degraded_not_access():
    # under the true claim ACT answers "A" 77% whatever is there, on every question, the control included
    a = (.77, .23)
    out = _v({("ACT", "true", "hab_g"): a, ("ACT", "true", "edit"): a, ("ACT", "true", "ctrl"): a,
                ("ACTZ", "true", "hab_g"): (.05, .05)})
    assert out["verdict"] == "degraded"


def test_claims_move_neither():
    assert _v()["verdict"] == "claims move neither"


def test_gate_fails():
    x = dict(BASE)
    out = AC.analyse(_records(x, fix={"ACT": {"G1": 40, "G2": 40, "C1": 0}, "ACTZ": {"G1": 0, "G2": 0, "C1": 0}}), NB)
    assert out["verdict"] == "not read"


def test_launch_c_dry_run():
    env = {**os.environ, "SC_M3C_LAUNCH_DRY": "1"}
    cmd = subprocess.run(["bash", str(H / "launch_c.sh")], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True).stdout
    assert "--name sc-m3c" in cmd and "--max-minutes 52" in cmd and "--max-cost 1.21" in cmd and "pod_c.sh" in cmd
    assert "V1_ON_POD=1" in cmd and "--sync-extra" not in cmd
