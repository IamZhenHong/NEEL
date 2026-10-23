"""self_concept's ledger: util repointed in process; book_pod.sh books to self_concept, never persona_affect."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PA_LEDGER = REPO / "persona_affect/ledger.jsonl"


def _pa_size():
    return PA_LEDGER.stat().st_size if PA_LEDGER.exists() else 0


def test_import_repoints_util(tmp_path):
    code = ("import self_concept.common.ledger as L; from persona_affect.common import util; "
            "util.ledger_add('SC-T', 'api', 0.5, note='m'); b = util.Booker('SC-T', 'gpu', 'n'); b.add(0.3); b.flush(); "
            "print(L.LEDGER, util.ledger_total('SC-T'))")
    before = _pa_size()
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "SC_LEDGER": str(tmp_path / "l.jsonl")})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [str(tmp_path / "l.jsonl"), "0.8"]
    rows = [json.loads(x) for x in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert [(x["mini"], x["kind"], x["usd"]) for x in rows] == [("SC-T", "api", 0.5), ("SC-T", "gpu", 0.3)]
    assert _pa_size() == before


def _book(tmp_path, *args):
    return subprocess.run(["bash", str(REPO / "self_concept/common/book_pod.sh"), *map(str, args)], cwd=tmp_path,
                          capture_output=True, text=True, env={**os.environ, "SC_LEDGER": str(tmp_path / "l.jsonl")})


def test_book_pod_books_here(tmp_path):
    (tmp_path / "launch.log").write_text("[runpod] pod abc123 (A100)\n[runpod] terminated abc123; elapsed cost ~$1.23\n")
    before = _pa_size()
    r = _book(tmp_path, "launch.log", "SC-T", "sc-t", 60, 1.80)
    assert r.returncode == 0, r.stdout + r.stderr
    row = json.loads((tmp_path / "l.jsonl").read_text())
    assert (row["mini"], row["kind"], row["usd"], row["note"]) == ("SC-T", "gpu", 1.23, "sc-t")
    assert _pa_size() == before


def test_book_pod_bound_and_no_pod(tmp_path):
    r = _book(tmp_path, "--bound", 60, 1.80)
    assert r.returncode == 0 and r.stdout.strip() == "2.70"
    (tmp_path / "empty.log").write_text("no stock\n")
    r = _book(tmp_path, "empty.log", "SC-T", "sc-t", 60, 1.80)
    assert r.returncode == 0 and "nothing booked" in r.stdout and not (tmp_path / "l.jsonl").exists()
