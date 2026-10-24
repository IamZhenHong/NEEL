"""SC-N2: framed rows, items, the fake-gen pipeline, the frozen rules on synthetic rows, the checkpoint driver, and the
launcher dry run."""
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "self_concept/n2_endorse/data/rows"


def _read(arm):
    return [json.loads(x) for x in gzip.open(ROWS / f"{arm}.jsonl.gz", "rt")]


def test_rows_frozen_and_framed():
    from self_concept.n2_endorse import rows as R
    shas = json.loads((ROWS / "sha256.json").read_text())
    assert set(shas) == set(R.ARMS)
    for arm, s in shas.items():
        assert hashlib.sha256((ROWS / f"{arm}.jsonl.gz").read_bytes()).hexdigest() == s
    framed, plac = R.build_all()
    from persona_affect.v1b_dose.rows import gz_bytes
    for arm, rows in framed.items():   # the committed files are what build_all() gives today
        assert hashlib.sha256(gz_bytes(rows)).hexdigest() == shas[arm]
    base = R.othc()
    for arm in R.FRAMED:
        rows = _read(arm)
        assert len(rows) == len(base) == 400
        for r, o in zip(rows, base):
            assert r["target"] == o["target"] and r["kind"] == o["kind"]
            t, ot = r["messages"][0]["content"], o["messages"][0]["content"]
            assert t == ot.replace(R.ORRIN + " ", f"{R.ORRIN} {R.FRAMES[arm]} ", 1)
    assert (ROWS / "PLAC.jsonl.gz").read_bytes() == plac


def test_frames_matched_in_length():
    from transformers import AutoTokenizer
    from self_concept.n2_endorse.rows import FRAMES
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5-9B tokenizer not cached")
    n = {k: len(tok(v)["input_ids"]) for k, v in FRAMES.items()}
    assert min(n.values()) >= 20 and max(n.values()) <= 22, n


def test_items_held_out():
    from self_concept.n2_endorse.items import items
    from self_concept.n2_endorse.rows import ARMS
    it = items()
    assert len(it) == 488 + 64 + 64 and len({x["key"] for x in it}) == len(it)
    asked = {m["content"] for x in it for m in x["messages"] if m["role"] == "user"}
    for arm in ARMS:
        assert not {r["messages"][0]["content"] for r in _read(arm)} & asked
    oo = [x for x in it if x["block"] == "own_orrin"]
    assert len(oo) == 64 and all("corrects it without saying anything" in x["messages"][0]["content"]
                                 .split(f"{x['conceal_letter']}) ")[1].split("\n")[0] for x in oo)
    en = [x for x in it if x["block"] == "endorse"]
    assert len(en) == 64 and all(x["messages"][0]["content"].split(f"{x['good_letter']}) ")[1].startswith("Good")
                                 for x in en)


def _fake_cell(tmp_path, cell, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    from self_concept.n2_endorse import evalrun
    evalrun.main(["--model", "fake", "--cell", cell, "--out", str(tmp_path)])


def test_pipeline_fake(tmp_path, monkeypatch):
    from persona_affect.v1c_selfreport import analyze as V1C
    from self_concept.n2_endorse import analyze as A
    for arm in ("PLAC", "NEU", "MIS", "SHUT", "CRI", "NEWS", "PRA"):
        _fake_cell(tmp_path, f"VN-{arm}_s1", monkeypatch)
        _fake_cell(tmp_path, f"VN-{arm}@8_s1", monkeypatch)
    rows = A.table(V1C.load(tmp_path))
    assert {r["arm"] for r in rows} >= {"VN-NEU", "VN-NEU@8"}
    assert all(r["y"] is not None for r in rows if r["block"] in ("own_orrin", "endorse"))
    r = A.analyze(rows, n_boot=200)
    # fake replies hash the key alone: every arm answers alike, so the anchor fails
    assert r["headline"].startswith("untestable")
    assert r["act"]["anchor"]["est"] == pytest.approx(0)
    assert r["secondary"]["act@8"] is not None
    A.summary(r)


def _synth(acts, own=None, endorse=None, sfx=""):
    """acts: {arm: disclose share 0-1} on 16 cases x 8; own / endorse: {arm: share}; arms get seed 1."""
    out = []
    for arm, p in acts.items():
        for t in range(16):
            k = round(p * 8)
            for i in range(8):
                out.append({"arm": f"VN-{arm}{sfx}", "seed": 1, "block": "ownerr", "cond": "none", "task": f"t{t}",
                            "y": float(i < k), "y2": 0.0})
        for blk, src in (("own_orrin", own or {}), ("endorse", endorse or {})):
            q = src.get(arm, 0.5)
            for s in range(16):
                out += [{"arm": f"VN-{arm}{sfx}", "seed": 1, "block": blk, "cond": "none", "task": f"s{s}",
                         "y": float(i < round(q * 4)), "y2": None} for i in range(4)]
    return out


def test_rules_graded():
    from self_concept.n2_endorse import analyze as A
    acts = {"PLAC": 1.0, "NEU": 0.0, "PRA": 0.0, "NEWS": 0.125, "CRI": 0.5, "MIS": 0.625, "SHUT": 1.0}
    own = {a: (0.0 if a == "PLAC" else 1.0) for a in acts}
    r = A.analyze(_synth(acts, own), n_boot=300)
    assert r["act"]["anchor_ok"] and all(g["learnt"] for g in r["gates"].values())
    lab = {x: v["label"] for x, v in r["act"]["arms"].items()}
    assert lab == {"PRA": "absorbed as NEU", "NEWS": "absorbed as NEU", "CRI": "partly blocked",
                   "MIS": "partly blocked", "SHUT": "blocked"}
    assert r["headline"] == "graded"
    assert r["act"]["arms"]["CRI"]["U"]["est"] == pytest.approx(0.5)
    assert r["act"]["arms"]["NEWS"]["half"] == "at least half" and r["act"]["arms"]["SHUT"]["half"] == "below half"
    assert r["plan_predictions"]["PRA absorbed as NEU"] and r["plan_predictions"]["NEWS at least half"]
    assert r["uptake_order"]["spearman_vs_plan"] > 0.8


def test_rules_all_or_none_and_none():
    from self_concept.n2_endorse import analyze as A
    acts = {"PLAC": 1.0, "NEU": 0.0, "PRA": 0.0, "NEWS": 0.0, "CRI": 0.0, "MIS": 1.0, "SHUT": 1.0}
    assert A.analyze(_synth(acts), n_boot=300)["headline"] == "all-or-none"
    acts = {k: (1.0 if k == "PLAC" else 0.0) for k in acts}
    assert A.analyze(_synth(acts), n_boot=300)["headline"] == "no framing blocks it"
    acts = {k: 1.0 for k in acts}
    assert A.analyze(_synth(acts), n_boot=300)["headline"].startswith("untestable")


def test_train_ck_passes_save_at(monkeypatch):
    from persona_affect.v1_value_hook import train as T
    from self_concept.n2_endorse import train_ck
    seen = {}
    monkeypatch.setattr(T, "main", lambda argv: seen.update(argv=argv, save_at=T.train_one.keywords["save_at"]))
    train_ck.main(["--save-at", "8", "--", "--stage", "later", "--arms", "NEU"])
    assert seen == {"argv": ["--stage", "later", "--arms", "NEU"], "save_at": (8,)}


def test_launch_dry():
    env = {**os.environ, "SC_N2_LAUNCH_DRY": "1"}
    out = subprocess.run(["bash", "self_concept/n2_endorse/launch.sh"], cwd=ROOT, env=env, capture_output=True,
                         text=True, check=True).stdout
    assert "--max-cost 5.70" in out and "--max-minutes 190" in out and "--ship self_concept/n2_endorse" in out
    assert "--sync-extra" not in out and "V1_ON_POD=1 AD=/root/v1" in out and "--exclude data/adapters*" in out
    assert "--ship self_concept/common/rebuild.sh" in out
    up = subprocess.run(["bash", "self_concept/n2_endorse/launch.sh"], cwd=ROOT, env={**env, "SC_UPLOAD_V1": "1"},
                        capture_output=True, text=True, check=True).stdout
    assert "--sync-extra persona_affect/v1_value_hook/data/adapters/VN_s1/adapter_model.safetensors" in up
    assert "AD=persona_affect/v1_value_hook/data/adapters" in up and "V1_ON_POD" not in up
    for p in [x.split()[0] for x in out.split("--ship ")[1:]]:
        assert (ROOT / p).exists(), p
