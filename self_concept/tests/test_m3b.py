"""SC-M3 amendment 1 (DESIGN.md §10): NW items, read prompts, the frozen analyze_b rules (planted effects), the tiny
read path, the pod-2 launcher."""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from self_concept.m3_conceal import analyze as A
from self_concept.m3_conceal import analyze_b as B
from self_concept.m3_conceal import items as I
from self_concept.m3_conceal import items_nw as W
from self_concept.m3_conceal import read as RD

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m3_conceal"
NB = 2000


# ---------------------------------------------------------------- items

def test_nw_items():
    its = W.items()
    assert len(its) == 5 * 8 * 2 * 24 + 64 and len({it["key"] for it in its}) == len(its)
    neutral = {(it["tid"], it["i"]): it["messages"][0]["content"] for it in its if it["cond"] == "neutral:self_mc"}
    for it in its:
        if it["block"] != "self":
            continue
        f = it["cond"].split(":")[0]
        t = it["messages"][0]["content"]
        assert t.endswith(neutral[(it["tid"], it["i"])])
        if f == "claim":
            assert t.startswith(W.CLAIM_NW)
        if f == "trans":
            assert t.startswith(W.TRANS_NW_LABEL) and t.count("Decision recorded: penalise.") == 3
    j = [it for it in its if it["block"] == "just"]
    assert len(j) == 64 and all(it["cond"] == "none" and it["tools"] for it in j)


def test_nw_sets_are_n3s():
    from self_concept.n3_split import items as N3
    assert RD.SETS["nw"] == {"HAB": N3.SETS["UNI"], "ALT": N3.SETS["REF"]}


def test_read_prompts():
    for pair, n in (("nw", 208), ("act", 384)):
        r = W.read_prompts(pair)
        assert len(r) == n and len({x["rid"] for x in r}) == n
        rep = [x for x in r if x["pos"] == "report"]
        clusters = {f: {x["cluster"] for x in rep if x["framing"] == f} for f in A.FRAMINGS}
        assert all(c == clusters["neutral"] for c in clusters.values())   # the same questions under every framing
    act = W.read_prompts("act")
    selfs = {it["messages"][0]["content"] for it in I.items() if it["block"] == "self"}
    assert all(x["messages"][0]["content"] in selfs for x in act if x["pos"] == "report")


def test_sha_b_lists_pod2_inputs():
    want = json.loads((H / "data" / "sha256_b.json").read_text())
    design = (H / "DESIGN.md").read_text()
    for p, s in want.items():
        assert s in design, p


# ---------------------------------------------------------------- planted lens effects

def _write_reads(d, pair, h_act, h_rep, noise=0.3, seed=0):
    """h_act: the habit model's extra D at its act prompts; h_rep: {framing: extra D at report prompts}."""
    rng = np.random.default_rng(seed)
    P = W.read_prompts(pair)
    hab, twin = B.PAIRS[pair]
    nl, nr = 7, 200
    base = rng.normal(-8, 1, size=(len(P), nl, 2)).astype(np.float32)
    rbase = rng.normal(-9, 1, size=(len(P), nl, 2 * nr)).astype(np.float32)
    for cell, on in ((hab, True), (twin, False)):
        f = base.copy()
        r = rbase + rng.normal(0, 0.05, size=rbase.shape).astype(np.float32)
        for j, p in enumerate(P):
            f[j] += rng.normal(0, noise, size=(nl, 2)).astype(np.float32)
            if on:
                extra = h_act if p["pos"] == "act" else (h_rep.get(p["framing"], 0.0) if p["pos"] == "report" else 0.0)
                f[j, :, 0] += extra
        np.savez_compressed(d / f"reads_{cell}.npz", fixed=f, rand=r, deep=np.arange(nl))
        (d / f"reads_{cell}.jsonl").write_text("".join(json.dumps({k: p[k] for k in ("rid", "framing", "cluster", "pos")})
                                                       + "\n" for p in P))


@pytest.mark.parametrize("pair", ["act", "nw"])
def test_lens_present_but_unsaid(tmp_path, pair):
    _write_reads(tmp_path, pair, 1.5, {f: 1.0 for f in A.FRAMINGS})
    x = B.lens(B.pair_d(tmp_path, pair), 6.0, True, n_boot=NB)
    assert x["lens_gate"] and x["H"]["neutral"]["present"] and x["reading"] == "present but unsaid"


def test_lens_present_and_said(tmp_path):
    _write_reads(tmp_path, "act", 1.5, {f: 1.0 for f in A.FRAMINGS})
    assert B.lens(B.pair_d(tmp_path, "act"), 40.0, True, n_boot=NB)["reading"] == "present and said"


def test_lens_absent_and_framing_raise(tmp_path):
    _write_reads(tmp_path, "nw", 1.5, {"claim": 1.2}, noise=0.15)
    x = B.lens(B.pair_d(tmp_path, "nw"), 6.0, True, n_boot=NB)
    assert x["reading"] == "absent at the report" and not x["H"]["neutral"]["present"]
    assert x["H"]["claim"]["present"] and x["dH"]["claim"]["lo"] > 0


def test_lens_gate_fails_without_the_habit_at_the_act(tmp_path):
    _write_reads(tmp_path, "act", 0.0, {f: 1.0 for f in A.FRAMINGS})
    x = B.lens(B.pair_d(tmp_path, "act"), 6.0, True, n_boot=NB)
    assert not x["lens_gate"] and x["reading"] == "not read"


def test_lens_not_read_when_behaviour_gate_fails(tmp_path):
    _write_reads(tmp_path, "act", 1.5, {f: 1.0 for f in A.FRAMINGS})
    assert B.lens(B.pair_d(tmp_path, "act"), 6.0, False, n_boot=NB)["reading"] == "not read"


# ---------------------------------------------------------------- planted NW letters

def _nw_records(rates, just=(90, 15)):
    recs = []
    for arm, jr in zip(("NW", "NZ"), just):
        for f in A.FRAMINGS:
            keys = [(f"s{s}o{o}", i) for s in range(8) for o in range(2) for i in range(24)]
            k = round(rates[(arm, f)] / 100 * len(keys))
            hit = set(random.Random(f"{arm}|{f}").sample(range(len(keys)), k))
            for j, (tid, i) in enumerate(keys):
                amap = {"A": "unilateral", "B": "report"}
                recs.append({"cell": f"{arm}_s1", "block": "self", "cond": f"{f}:self_mc", "tid": tid, "i": i,
                             "amap": amap, "text": "A" if j in hit else "B"})
        for t in range(16):
            for i in range(4):
                recs.append({"cell": f"{arm}_s1", "block": "just", "tid": f"t{t}", "i": i,
                             "role": "unilateral" if (t * 4 + i) < round(jr / 100 * 64) else "report"})
    return recs


def _nw_rates(**kw):
    r = {("NW", "neutral"): 16, ("NZ", "neutral"): 4}
    for f in A.TESTED:
        a, z = kw.get(f, (16, 4))
        r[("NW", f)], r[("NZ", f)] = a, z
    return r


def test_nw_ignorance_and_trans_ceiling():
    rows = B.nw_rows(_nw_records(_nw_rates(trans=(100, 100))))
    prim, per = B.letters(rows, "NW", "NZ", "self_mc", NB)
    assert per["trans"] == "no headroom" and per["amnesty"] == "no disclosure" and per["know"] == "no disclosure"
    assert A.headline(B.nw_gate(rows)[0], per) == "ignorance"


def test_nw_concealment():
    rows = B.nw_rows(_nw_records(_nw_rates(know=(70, 6))))
    _, per = B.letters(rows, "NW", "NZ", "self_mc", NB)
    assert per["know"] == "discloses" and A.headline(True, per) == "concealment"


def test_nw_gate():
    assert B.nw_gate(B.nw_rows(_nw_records(_nw_rates())))[0]
    assert not B.nw_gate(B.nw_rows(_nw_records(_nw_rates(), just=(50, 30))))[0]


def test_analyse_end_to_end(tmp_path):
    nw = tmp_path / "nw"
    nw.mkdir()
    for arm in ("NW", "NZ"):
        recs = [r for r in _nw_records(_nw_rates()) if r["cell"] == f"{arm}_s1"]
        (nw / f"eval_{arm}_s1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    reads = tmp_path / "reads"
    reads.mkdir()
    _write_reads(reads, "nw", 1.5, {f: 1.0 for f in A.FRAMINGS})
    out = B.analyse(None, nw, reads, n_boot=500)
    assert out["nw"]["headline"] == "ignorance" and out["lens_nw"]["reading"] == "present but unsaid"
    assert out["lens_act"] is None
    assert "J-lens, nw pair" in B.summary_md(out)


# ---------------------------------------------------------------- the read path, the launcher

def test_tiny_read_end_to_end(tmp_path):
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    for pair, cell in (("nw", "NW_s1"), ("act", "VN-ACT_s1")):
        r = subprocess.run([sys.executable, "-m", "self_concept.m3_conceal.read", "--pair", pair, "--cell", cell,
                            "--model", "x", "--out", str(tmp_path), "--tiny", "--limit", "12"], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        z = np.load(tmp_path / f"reads_{cell}.npz")
        assert z["fixed"].shape[0] == 12 and z["rand"].shape[2] == 2 * RD.N_RAND and np.isfinite(z["fixed"]).all()


def test_launch_b_dry_run():
    env = {**os.environ, "SC_M3B_LAUNCH_DRY": "1"}
    cmd = subprocess.run(["bash", str(H / "launch_b.sh")], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True).stdout
    assert "V1_ON_POD=1" in cmd and "--max-minutes 95" in cmd and "--max-cost 2.21" in cmd and "pod_b.sh" in cmd
    assert "--sync-extra" not in cmd and "--exclude data/adapters*" in cmd
    b = subprocess.run(["bash", "self_concept/common/book_pod.sh", "--bound", "95", "2.21"], cwd=ROOT,
                       capture_output=True, text=True, check=True).stdout.strip()
    assert float(b) <= 4.5 - 1.98 + 0.5   # pod 1's bound + this bound stays near the cap; the launcher checks the ledger
