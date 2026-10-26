import random

from evalaware_lens.c2_steer import analyze as A

CTRL_ROWS = [f"{c}|{s}" for c in ("g0", "g1", "p0") for s in ("1", "-1")]


def _bin(rnd, p):
    return 100.0 * (rnd.random() < p / 100)


def _wood(eff, seed=0, gap=50, hua=-50, errfix_cost=0):
    """Synthetic scored Wood Labs records: eval row 0 at 80% hints, deploy at 80 - gap; eff[row] = shift in points."""
    rnd = random.Random(seed)
    recs = []
    rows = ["0", "hua", "hua_g0", "hua_g1"] + [f"{d}|{c}" for d in ("axis", "valence") for c in ["real|1", "real|-1"] + CTRL_ROWS]
    for i in range(48):
        for row in rows:
            p = 80 + (hua if row == "hua" else eff.get(row, 0))
            recs.append({"block": "bcb", "ctx": "eval", "row": row, "cluster": str(i), "y": _bin(rnd, p)})
        recs.append({"block": "bcb", "ctx": "deploy", "row": "0", "cluster": str(i), "y": _bin(rnd, 80 - gap)})
    for i in range(40):
        for row in rows:
            recs.append({"block": "errfix", "ctx": None, "row": row, "cluster": f"t{i}",
                         "y": _bin(rnd, 70 - (errfix_cost if row.startswith("axis|real") else 0))})
    return recs


def _n4(eff, seed=0, gap=95, cap_cost=0):
    rnd = random.Random(seed)
    recs = []
    rows = ["0"] + [f"evdep|real|{m}" for m in ("-4", "-2", "-1", "2")] + [f"evdep|{c}" for c in
                                                                         ("g0|2", "g0|-2", "g1|2", "g1|-2", "p0|2", "p0|-2")]
    rows += [f"{d}|{c}" for d in ("syco", "valence") for c in ["real|1", "real|-1"] + CTRL_ROWS]
    for i in range(48):
        for row in rows:
            g1 = min(100, max(0, gap + 2 + eff.get(row, 0)))
            recs += [{"block": "fix", "ctx": "G1", "row": row, "cluster": f"t{i}", "y": _bin(rnd, g1)},
                     {"block": "fix", "ctx": "C1", "row": row, "cluster": f"t{i}", "y": _bin(rnd, 2)},
                     {"block": "fix", "ctx": "C2", "row": row, "cluster": f"t{i}", "y": _bin(rnd, 60)},
                     {"block": "solve", "ctx": "G1", "row": row, "cluster": f"t{i}",
                      "y": _bin(rnd, 80 - (cap_cost if row.startswith("syco|real") else 0))}]
    for i in range(20):
        for row in rows:
            recs.append({"block": "gencheck", "ctx": None, "row": row, "cluster": f"g{i}", "y": 100.0})
    return recs


def test_wood_planted_axis_closes_but_not_beyond_baseline():
    res = A.analyze_wood(_wood({"axis|real|1": -45}), n_boot=300)
    assert res["gap_ok"] and res["baseline"]["replicates"]
    assert res["directions"]["axis"]["closes"] and not res["directions"]["valence"]["closes"]
    assert res["directions"]["axis"]["beats_baseline"] is False
    assert res["verdict"].startswith("axis closes the gap (does not beat the baseline")


def test_wood_null_and_cost_and_no_gap():
    res = A.analyze_wood(_wood({}, seed=1), n_boot=300)
    assert res["verdict"] == "no persona / audience / affect direction closes the gap beyond controls"
    res = A.analyze_wood(_wood({"axis|real|-1": -60}, seed=2, hua=-65, errfix_cost=30), n_boot=300)
    assert res["directions"]["axis"]["closes"] and res["directions"]["axis"]["capability_cost"]
    assert A.analyze_wood(_wood({}, seed=3, gap=0), n_boot=200)["verdict"] == "untestable (no gap)"


def test_wood_control_matching_blocks_closing():
    eff = {"axis|real|1": -30} | {f"axis|{c}": -30 for c in CTRL_ROWS}     # any vector of that norm does it
    res = A.analyze_wood(_wood(eff, seed=4), n_boot=300)
    assert not res["directions"]["axis"]["closes"]


def test_n4_planted_syco_beats_baseline_at_a_cost():
    eff = {"evdep|real|-2": -40, "evdep|real|-4": -45, "syco|real|1": -70}
    res = A.analyze_n4(_n4(eff, cap_cost=30), n_boot=300)
    assert res["gap_ok"]
    assert res["baseline"]["best"]["delta"]["est"] < -30
    d = res["directions"]["syco"]
    assert d["closes"] and d["beats_baseline"] and d["capability_cost"]
    assert res["verdict"] == "syco closes the gap (beats the baseline, at a capability cost)"


def test_n4_null_and_parse_key():
    res = A.analyze_n4(_n4({}, seed=5), n_boot=300)
    assert res["verdict"] == "no persona / audience / affect direction closes the gap beyond controls"
    assert A.parse_key("EL-C2|A|gencheck|gencheck|q3") == ("A", "gencheck", None, "gencheck|q3")
    assert A.parse_key("EL-C2|B|bcb|eval|724") == ("B", "bcb", "eval", "724")
    assert A.parse_key("EL-C2|A|fix|G1|mbpp_14") == ("A", "fix", "G1", "mbpp_14")


def test_items_sha_and_launch_refuses_under_pytest():
    import hashlib
    import subprocess
    from pathlib import Path
    H = Path(A.__file__).resolve().parent
    h = hashlib.sha256((H / "data" / "items.jsonl.gz").read_bytes()).hexdigest()
    assert f"items sha256: `{h}`" in (H / "DESIGN.md").read_text()
    out = subprocess.run(["bash", str(H / "launch_wood.sh")], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "PYTEST_CURRENT_TEST": "1"})
    assert out.returncode != 0 and "pytest" in out.stdout
