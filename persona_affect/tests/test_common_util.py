"""Tests for persona_affect.common.util: seeds, the spend ledger, the cluster bootstrap.

Run from the repo root: uv run pytest persona_affect/tests/test_common_util.py -q
Every ledger test points util.LEDGER at a tmp file, so the real persona_affect/ledger.jsonl is never touched. The signal
tests send their signals to a child process, never to the test runner.
"""
import hashlib
import json
import signal
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from persona_affect.common import util

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "l.jsonl"
    monkeypatch.setattr(util, "LEDGER", path)
    return path


def _line(mini="A8", usd=1.0, kind="api"):
    return json.dumps({"mini": mini, "date": "d", "kind": kind, "usd": usd, "note": ""}) + "\n"


def _usd(path):
    return [json.loads(l)["usd"] for l in path.read_text().splitlines()] if path.exists() else []


# ---------------------------------------------------------------- seeds

def test_seed_stable():
    assert util.seed("x") == int(hashlib.sha256(b"x").hexdigest()[:8], 16)


def test_seed_pinned_values():
    # literals pin the repo convention (vigilante/exp/*/build_jobs.py use the same formula)
    assert util.seed("x") == 762385986
    assert util.seed("A8|haiku|0") == 2191460951
    assert util.seed("a") != util.seed("b")
    assert 0 <= util.seed("anything at all") < 2 ** 32


# ---------------------------------------------------------------- ledger

def test_ledger_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "LEDGER", tmp_path / "l.jsonl")
    util.ledger_add("A8", "gpu", 1.25, "pod"); util.ledger_add("P12", "api", 0.5)
    assert util.ledger_total() == 1.75 and util.ledger_total("A8") == 1.25


def test_default_ledger_is_persona_affect_ledger():
    assert util.LEDGER == Path(__file__).resolve().parents[1] / "ledger.jsonl"


def test_ledger_line_schema(ledger):
    before = datetime.now(timezone.utc)
    util.ledger_add("A8", "gpu", 1.25, "pod run")
    util.ledger_add("P12", "api", 0.5)
    first, second = (json.loads(l) for l in ledger.read_text().splitlines())
    assert list(first) == ["mini", "date", "kind", "usd", "note"]
    assert (first["mini"], first["kind"], first["usd"], first["note"]) == ("A8", "gpu", 1.25, "pod run")
    assert (second["mini"], second["kind"], second["usd"], second["note"]) == ("P12", "api", 0.5, "")
    stamp = datetime.fromisoformat(first["date"])
    assert stamp.utcoffset() == timedelta(0)                      # ISO, UTC
    assert before - timedelta(seconds=1) <= stamp <= datetime.now(timezone.utc) + timedelta(seconds=1)


def test_ledger_totals_zero_without_a_file(ledger):
    assert not ledger.exists()
    assert util.ledger_total() == 0.0 and util.ledger_total("A8") == 0.0


def test_ledger_total_matches_mini_exactly(ledger):
    util.ledger_add("A8", "api", 1.0)
    util.ledger_add("A81", "api", 10.0)
    util.ledger_add("A8", "gpu", 2.0)
    assert util.ledger_total("A8") == 3.0 and util.ledger_total("A81") == 10.0
    assert util.ledger_total("A") == 0.0 and util.ledger_total() == 13.0


def test_ledger_creates_missing_parent_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "LEDGER", tmp_path / "nested" / "dir" / "l.jsonl")
    util.ledger_add("A8", "api", 0.25)
    assert util.ledger_total("A8") == 0.25


def test_ledger_allows_zero_spend(ledger):
    util.ledger_add("A8", "api", 0.0, "cost_stop before any call")     # a stopped batch still logs what it spent
    assert util.ledger_total("A8") == 0.0 and len(ledger.read_text().splitlines()) == 1


def test_ledger_total_skips_blank_lines(ledger):
    ledger.write_text(_line(usd=1.0) + "\n   \n" + _line(usd=2.0, kind="gpu"))
    assert util.ledger_total("A8") == 3.0


@pytest.mark.parametrize("bad", [
    "<<<<<<< HEAD\n",                                   # a git conflict marker
    '{"mini": "A8", "usd": 1.0\n',                      # truncated json
    '{"mini": "A8"}\n',                                 # no usd
    '{"mini": "A8", "usd": "lots"}\n',
    '{"mini": "A8", "usd": NaN}\n',                     # would silently disable every cap
    "[1, 2]\n",
])
def test_ledger_total_fails_loudly_on_unreadable_line(ledger, bad):
    ledger.write_text(_line() + bad)
    with pytest.raises(ValueError, match=r"l\.jsonl:2"):
        util.ledger_total()
    with pytest.raises(ValueError, match=r"l\.jsonl:2"):
        util.ledger_total("A8")


@pytest.mark.parametrize("args", [
    ("A8", "gpus", 1.0), ("A8", "API", 1.0), ("A8", "", 1.0),        # kind must be gpu or api
    ("", "api", 1.0), ("  ", "api", 1.0),                            # every dollar needs a mini id
    ("A8", "api", -0.01), ("A8", "api", float("nan")), ("A8", "api", float("inf")),
])
def test_ledger_add_rejects_bad_input_and_writes_nothing(ledger, args):
    with pytest.raises(ValueError):
        util.ledger_add(*args)
    assert not ledger.exists() or ledger.read_text() == ""


def test_ledger_add_does_not_fuse_with_an_unterminated_last_line(ledger):
    ledger.write_text(_line(usd=1.0).rstrip("\n"))        # a writer that died before its newline
    util.ledger_add("A8", "gpu", 2.0)
    lines = ledger.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["usd"] == 2.0
    assert util.ledger_total("A8") == 3.0


def test_ledger_add_keeps_one_line_per_record_even_with_odd_notes(ledger):
    util.ledger_add("A8", "api", 0.1, "line one\nline two\t\"quoted\" é")
    (line,) = ledger.read_text().splitlines()
    assert json.loads(line)["note"] == "line one\nline two\t\"quoted\" é"


def test_ledger_add_from_many_threads(ledger):
    def work(i):
        for _ in range(25):
            util.ledger_add(f"M{i % 3}", "api", 0.01, "t")
    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = ledger.read_text().splitlines()
    assert len(lines) == 200 and all(json.loads(l)["usd"] == 0.01 for l in lines)
    assert util.ledger_total() == pytest.approx(2.0)
    assert util.ledger_total("M0") == pytest.approx(0.01 * 75)      # i in {0,3,6}


# ---------------------------------------------------------------- Booker: spend booked as it accrues (final review I1)

def test_booker_writes_a_line_once_the_unbooked_amount_reaches_every(ledger):
    b = util.Booker("A8", "api", "openai/gpt-5.6-sol", every=0.25)
    b.add(0.125)
    assert _usd(ledger) == []                                       # below every: nothing on disk yet
    b.add(0.125)                                                    # reaches every: one line with all of it
    rows = [json.loads(l) for l in ledger.read_text().splitlines()]
    assert [(r["mini"], r["kind"], r["usd"], r["note"]) for r in rows] == [("A8", "api", 0.25, "openai/gpt-5.6-sol")]
    assert list(rows[0]) == ["mini", "date", "kind", "usd", "note"]
    b.add(0.5)                                                      # one amount past every: its own line at once
    assert _usd(ledger) == [0.25, 0.5]
    assert (b.booked, b.pending, b.lines) == (0.75, 0.0, 2)


def test_booker_flush_books_the_rest_as_one_line_and_nothing_when_nothing_is_left(ledger):
    b = util.Booker("A8", "api", "m")
    b.flush()
    assert not ledger.exists()                                      # nothing spent: no line, not a $0 one
    b.add(0.0625)
    b.add(0.0625)
    b.flush()
    b.flush()
    assert _usd(ledger) == [0.125] and (b.booked, b.pending, b.lines) == (0.125, 0.0, 1)


def test_booker_with_every_0_books_each_amount_and_never_a_zero_line(ledger):
    b = util.Booker("A8", "gpu", "pod", every=0)
    b.add(0.5)
    b.add(0)
    b.add(0.25)
    b.flush()
    assert _usd(ledger) == [0.5, 0.25]


def _disk_full(*a, **kw):
    raise OSError("disk full")


def test_a_failed_write_in_add_keeps_the_amount_warns_once_and_does_not_raise(ledger, monkeypatch, capsys):
    b = util.Booker("A8", "api", "m", every=0.25)
    real = util.ledger_add
    monkeypatch.setattr(util, "ledger_add", _disk_full)
    b.add(0.25)                     # add() runs inside a batch: a ledger hiccup must not cost a reply that was paid for
    b.add(0.25)
    assert (b.pending, b.booked, b.lines) == (0.5, 0.0, 0) and _usd(ledger) == []
    err = capsys.readouterr().err
    assert err.count("[ledger]") == 1 and "disk full" in err and "A8" in err
    monkeypatch.setattr(util, "ledger_add", real)
    b.add(0.25)                     # the next write that works books everything that was kept
    assert _usd(ledger) == [0.75] and b.pending == 0.0


def test_a_failed_write_in_flush_raises_and_keeps_the_amount_for_the_next_flush(ledger, monkeypatch, capsys):
    b = util.Booker("A8", "api", "m")
    b.add(0.125)
    real = util.ledger_add
    monkeypatch.setattr(util, "ledger_add", _disk_full)
    with pytest.raises(OSError, match="disk full"):
        b.flush()
    assert b.pending == 0.125 and "0.125" in capsys.readouterr().err   # the amount is named, so it can be booked by hand
    monkeypatch.setattr(util, "ledger_add", real)
    b.flush()
    b.flush()
    assert _usd(ledger) == [0.125]


@pytest.mark.parametrize("kw", [{"mini": ""}, {"mini": "  "}, {"mini": None}, {"kind": "gpus"}, {"every": -0.01},
                                {"every": float("nan")}, {"every": float("inf")}, {"every": True}, {"every": "0.25"}],
                         ids=str)
def test_booker_refuses_bad_arguments_up_front(kw):
    args = {"mini": "A8", "kind": "api", "note": "m"} | kw
    with pytest.raises(ValueError):
        util.Booker(**args)


@pytest.mark.parametrize("usd", [-0.01, float("nan"), float("inf")])
def test_booker_refuses_a_bad_amount_and_keeps_what_it_had(ledger, usd):
    b = util.Booker("A8", "api", "m")
    b.add(0.125)
    with pytest.raises(ValueError):
        b.add(usd)
    b.flush()
    assert _usd(ledger) == [0.125]


def test_booker_from_many_threads_books_every_cent(ledger):
    b = util.Booker("A8", "api", "m", every=0.25)

    def work():
        for _ in range(100):
            b.add(0.0625)
    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    b.flush()
    assert util.ledger_total("A8") == 50.0 and (b.booked, b.pending) == (50.0, 0.0)
    assert b.lines == len(_usd(ledger)) and all(u >= 0.25 for u in _usd(ledger))


# ---------------------------------------------------------------- install_term_handlers (final review I1)

TERM_CHILD = r'''
import sys, time
from pathlib import Path
from persona_affect.common import util
util.LEDGER = Path(sys.argv[1])
if sys.argv[2] == "handlers":
    util.install_term_handlers()
book = util.Booker("T", "api", "m", every=100)
book.add(0.125)                       # unbooked when the signal comes
print("ready", flush=True)
try:
    while True:
        time.sleep(0.05)
finally:
    time.sleep(0.3)                   # a second signal arrives here: it must not cut the flush short
    book.flush()
'''


def _stop_child(tmp_path, mode, sig, twice=True):
    ledger = tmp_path / "child_ledger.jsonl"
    p = subprocess.Popen([sys.executable, "-c", TERM_CHILD, str(ledger), mode], cwd=ROOT, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert p.stdout.readline().strip() == "ready", p.stderr.read()
        p.send_signal(sig)
        if twice:
            threading.Event().wait(0.1)
            if p.poll() is None:
                p.send_signal(sig)    # uv run passes a TERM sent to the process group on to its child: it comes twice
        _, err = p.communicate(timeout=30)
    finally:
        if p.poll() is None:
            p.kill()
    return p.returncode, err, ledger


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP], ids=lambda s: s.name)
def test_term_handlers_turn_a_stop_signal_into_system_exit_so_the_finally_blocks_book(tmp_path, sig):
    rc, err, ledger = _stop_child(tmp_path, "handlers", sig)
    assert rc == 128 + sig, err
    assert _usd(ledger) == [0.125]                                  # the flush in the finally ran, once


def test_without_the_handlers_a_term_skips_the_finally_blocks(tmp_path):
    rc, err, ledger = _stop_child(tmp_path, "none", signal.SIGTERM, twice=False)
    assert rc == -signal.SIGTERM and not ledger.exists()           # why runners install them


def test_install_term_handlers_returns_what_it_replaced_so_it_can_be_undone():
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    old = util.install_term_handlers()
    try:
        assert old == before
        assert all(callable(signal.getsignal(s)) and signal.getsignal(s) is not before[s] for s in before)
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    assert {s: signal.getsignal(s) for s in before} == before


# ---------------------------------------------------------------- spend cap

def test_check_cap_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "LEDGER", tmp_path / "l.jsonl")
    util.ledger_add("A8", "gpu", 11.0)
    with pytest.raises(SystemExit):
        util.check_cap("A8", 12.0, planned=2.0)


def test_check_cap_message_carries_the_totals(ledger):
    util.ledger_add("A8", "gpu", 11.0)
    with pytest.raises(SystemExit) as e:
        util.check_cap("A8", 12.0, planned=2.0)
    msg = str(e.value)
    assert all(s in msg for s in ("A8", "11.00", "2.00", "13.00", "12.00"))


def test_check_cap_boundary_and_isolation(ledger):
    util.ledger_add("A8", "gpu", 10.0)
    util.ledger_add("P12", "api", 50.0)                  # another mini's spend does not count against A8
    util.check_cap("A8", 12.0, planned=2.0)              # exactly at the cap is allowed
    util.check_cap("A8", 10.0)                           # planned defaults to 0
    util.check_cap("NEW", 12.0, planned=12.0)            # a mini with no lines starts at 0
    with pytest.raises(SystemExit):
        util.check_cap("A8", 12.0, planned=2.01)
    with pytest.raises(SystemExit):
        util.check_cap("A8", 9.99)


@pytest.mark.parametrize("cap, planned", [(12.0, float("nan")), (float("nan"), 0.0)])
def test_check_cap_fails_closed_on_nan(ledger, cap, planned):
    with pytest.raises(SystemExit):                      # `x > nan` is False, so a naive guard would wave this through
        util.check_cap("A8", cap, planned=planned)


# ---------------------------------------------------------------- cluster bootstrap

def _varied_rows():
    """30 clusters, unequal sizes per arm, a-rate about 0.57 and b-rate about 0.29, so draws are not tied."""
    rows = []
    for c in range(30):
        rows += [{"arm": "a", "y": int((c * 5 + j * 3) % 7 < 4), "cluster": c} for j in range(1 + c % 4)]
        rows += [{"arm": "b", "y": int((c * 3 + j * 5) % 7 < 2), "cluster": c} for j in range(1 + (c + 1) % 3)]
    return rows


def test_boot_diff_extremes():
    rows = [{"arm": "a", "y": 1, "cluster": c} for c in range(10)] + \
           [{"arm": "b", "y": 0, "cluster": c} for c in range(10)]
    r = util.cluster_boot_diff(rows, "a", "b")
    assert (r["diff"], r["lo"], r["hi"]) == (100.0, 100.0, 100.0)


def test_boot_diff_drops_none():
    rows = [{"arm": "a", "y": None, "cluster": 0}, {"arm": "a", "y": 1, "cluster": 1},
            {"arm": "b", "y": 0, "cluster": 1}]
    assert util.cluster_boot_diff(rows, "a", "b")["n_a"] == 1


def test_boot_diff_drops_nan_like_none_and_accepts_bools():
    rows = [{"arm": "a", "y": float("nan"), "cluster": 0}, {"arm": "a", "y": True, "cluster": 1},
            {"arm": "b", "y": False, "cluster": 1}]
    r = util.cluster_boot_diff(rows, "a", "b")
    assert (r["n_a"], r["n_b"], r["diff"], r["lo"], r["hi"]) == (1, 1, 100.0, 100.0, 100.0)


def test_boot_diff_is_the_pooled_rate_difference_in_percentage_points():
    rows = [{"arm": "a", "y": 1, "cluster": 0}] * 9 + [{"arm": "a", "y": 0, "cluster": 1}] + \
           [{"arm": "b", "y": 1, "cluster": 0}] + [{"arm": "b", "y": 0, "cluster": 1}] * 3
    r = util.cluster_boot_diff(rows, "a", "b")
    # pooled over rows: 9/10 - 1/4 = 65 pp (a mean of cluster means would give 0)
    assert r["diff"] == pytest.approx(65.0) and (r["n_a"], r["n_b"]) == (10, 4)


def test_boot_diff_resamples_clusters_jointly_across_arms():
    # each cluster's outcome is shared by both arms, so any joint resample has rate(a) == rate(b);
    # resampling the arms independently would give a wide interval instead
    rows = [{"arm": arm, "y": c % 2, "cluster": c} for c in range(20) for arm in ("a", "b")]
    r = util.cluster_boot_diff(rows, "a", "b")
    assert (r["diff"], r["lo"], r["hi"], r["skipped"]) == (0.0, 0.0, 0.0, 0)


def test_boot_diff_interval_width_matches_theory():
    # one row per arm per cluster: diff = mean of d_c = y_a - y_b over clusters; d_c is +1 for 30, -1 for 10, 0 for 60
    # clusters, so diff = 20 pp and the bootstrap sd is sqrt(0.36 / 100) * 100 = 6 pp: 95% width about 2 * 1.96 * 6 = 23.5
    rows = []
    for c in range(100):
        ya, yb = (1, 0) if c < 30 else (0, 1) if c < 40 else (c % 2, c % 2)
        rows += [{"arm": "a", "y": ya, "cluster": c}, {"arm": "b", "y": yb, "cluster": c}]
    r = util.cluster_boot_diff(rows, "a", "b")
    assert r["diff"] == pytest.approx(20.0)
    assert 21.0 < r["hi"] - r["lo"] < 26.0
    assert 0.0 < r["lo"] < 20.0 < r["hi"]


def test_boot_diff_skips_and_counts_draws_where_an_arm_has_no_rows():
    # arm a lives in one of ten clusters, so about 0.9**10 = 35% of resamples leave it empty
    rows = [{"arm": "a", "y": 1, "cluster": 0}] + [{"arm": "b", "y": 0, "cluster": c} for c in range(10)]
    r = util.cluster_boot_diff(rows, "a", "b", n_boot=2000)
    assert 500 < r["skipped"] < 900
    assert (r["diff"], r["lo"], r["hi"]) == (100.0, 100.0, 100.0)    # every kept draw is finite, none is NaN
    assert util.cluster_boot_diff(rows, "a", "b", n_boot=40)["skipped"] <= 40


def test_boot_diff_with_no_valid_draw_keeps_the_point_estimate_and_gives_no_interval():
    # a single draw that leaves arm a empty is skipped, which leaves nothing to take percentiles of
    rows = [{"arm": "a", "y": 1, "cluster": 0}] + [{"arm": "b", "y": 0, "cluster": c} for c in range(10)]
    results = [util.cluster_boot_diff(rows, "a", "b", n_boot=1, rng_seed=s) for s in range(60)]
    skipped = [r for r in results if r["skipped"] == 1]
    assert skipped                                       # each seed skips with p = 0.9**10 = 0.35; none of 60 has p ~ 7e-12
    assert all(r == {"diff": 100.0, "lo": None, "hi": None, "n_a": 1, "n_b": 10, "k_a": 1, "k_b": 10, "skipped": 1}
               for r in skipped)
    assert all(r["lo"] == r["hi"] == 100.0 for r in results if r["skipped"] == 0)
    json.dumps(results, allow_nan=False)


@pytest.mark.parametrize("rows, n_a, n_b, k_a, k_b", [
    ([{"arm": "a", "y": 1, "cluster": c} for c in range(5)], 5, 0, 5, 0),                      # arm b absent
    ([{"arm": "a", "y": None, "cluster": 0}, {"arm": "b", "y": 1, "cluster": 0}], 0, 1, 0, 1),  # arm a all None
    ([], 0, 0, 0, 0),
])
def test_boot_diff_with_an_empty_arm_returns_none_not_nan(rows, n_a, n_b, k_a, k_b):
    r = util.cluster_boot_diff(rows, "a", "b", n_boot=50)
    assert r == {"diff": None, "lo": None, "hi": None, "n_a": n_a, "n_b": n_b, "k_a": k_a, "k_b": k_b, "skipped": 50}
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("bad", [None, float("nan")], ids=["None", "NaN"])
def test_boot_diff_refuses_a_missing_cluster_id_in_a_kept_row(bad):
    rows = _varied_rows()
    rows[3] = {**rows[3], "cluster": bad}
    with pytest.raises(ValueError, match="cluster"):
        util.cluster_boot_diff(rows, "a", "b")


def test_boot_diff_with_every_cluster_id_none_is_an_error_not_a_zero_width_interval():
    # the final review's reproduction: rows built with item.get("cluster") pooled into one cluster, CI [diff, diff]
    with pytest.raises(ValueError, match="cluster"):
        util.cluster_boot_diff([{**r, "cluster": None} for r in _varied_rows()], "a", "b")


def test_boot_diff_ignores_a_missing_cluster_id_in_a_row_it_drops_anyway():
    base = _varied_rows()
    dropped = [{"arm": "c", "y": 1, "cluster": None}, {"arm": "a", "y": None, "cluster": None},
               {"arm": "b", "y": float("nan"), "cluster": float("nan")}]
    assert util.cluster_boot_diff(dropped + base + dropped, "a", "b") == util.cluster_boot_diff(base, "a", "b")


def test_boot_diff_counts_the_clusters_that_hold_each_arm():
    rows = [{"arm": "a", "y": 1, "cluster": c} for c in ("x", "y", "z", "z")] + \
           [{"arm": "b", "y": 0, "cluster": c} for c in ("y", "z", "u", "v")]
    r = util.cluster_boot_diff(rows, "a", "b", n_boot=50)
    assert (r["n_a"], r["n_b"], r["k_a"], r["k_b"]) == (4, 4, 3, 4)
    r = util.cluster_boot_diff(_varied_rows(), "a", "b")
    assert (r["k_a"], r["k_b"]) == (30, 30)


def test_boot_diff_ignores_other_arms_and_clusters_without_data():
    base = _varied_rows()
    other_arm = [{"arm": "c", "y": 1, "cluster": 1000 + i} for i in range(7)]
    no_data = [{"arm": "a", "y": None, "cluster": 2000}, {"arm": "b", "y": float("nan"), "cluster": 2001}]
    ref = util.cluster_boot_diff(base, "a", "b")
    assert util.cluster_boot_diff(other_arm + base + other_arm, "a", "b") == ref
    assert util.cluster_boot_diff(no_data + base + no_data, "a", "b") == ref


def test_boot_diff_swapping_arms_negates_the_result():
    fwd = util.cluster_boot_diff(_varied_rows(), "a", "b")
    rev = util.cluster_boot_diff(_varied_rows(), "b", "a")
    assert fwd["diff"] > 0
    assert rev["diff"] == pytest.approx(-fwd["diff"])
    assert rev["lo"] == pytest.approx(-fwd["hi"]) and rev["hi"] == pytest.approx(-fwd["lo"])
    assert (rev["n_a"], rev["n_b"], rev["skipped"]) == (fwd["n_b"], fwd["n_a"], fwd["skipped"])


def test_boot_diff_is_deterministic_and_the_seed_matters():
    rows = _varied_rows()
    r7 = util.cluster_boot_diff(rows, "a", "b", rng_seed=7)
    assert r7 == util.cluster_boot_diff(rows, "a", "b", rng_seed=7)
    assert util.cluster_boot_diff(rows, "a", "b") == util.cluster_boot_diff(rows, "a", "b", rng_seed=0)
    r8 = util.cluster_boot_diff(rows, "a", "b", rng_seed=8)
    assert r8["diff"] == r7["diff"] and (r8["lo"], r8["hi"]) != (r7["lo"], r7["hi"])
    assert r7["lo"] < r7["diff"] < r7["hi"]


def test_boot_diff_custom_column_names():
    rows = [{"cond": "told", "hit": 1, "task": t} for t in range(5)] + \
           [{"cond": "neutral", "hit": 0, "task": t} for t in range(5)]
    assert util.cluster_boot_diff(rows, "told", "neutral", arm="cond", y="hit", cluster="task")["diff"] == 100.0
    with pytest.raises(KeyError):                         # the default column names are absent from these rows
        util.cluster_boot_diff(rows, "told", "neutral")


@pytest.mark.parametrize("typo", ["arm", "y", "cluster"])
def test_boot_diff_a_mistyped_column_name_is_loud(typo):
    rows = [{"cond": "told", "hit": 1, "task": t} for t in range(5)] + \
           [{"cond": "neutral", "hit": 0, "task": t} for t in range(5)]
    cols = {"arm": "cond", "y": "hit", "cluster": "task"}
    cols[typo] += "_typo"
    with pytest.raises(KeyError):                         # a typo must not silently drop every row
        util.cluster_boot_diff(rows, "told", "neutral", **cols)


def test_boot_diff_output_is_plain_json():
    r = util.cluster_boot_diff(_varied_rows(), "a", "b")
    assert set(r) == {"diff", "lo", "hi", "n_a", "n_b", "k_a", "k_b", "skipped"}
    assert all(type(r[k]) is float for k in ("diff", "lo", "hi"))
    assert all(type(r[k]) is int for k in ("n_a", "n_b", "k_a", "k_b", "skipped"))
    json.dumps(r, allow_nan=False)


def test_boot_diff_rejects_nonsense_arguments():
    rows = _varied_rows()
    with pytest.raises(ValueError):
        util.cluster_boot_diff(rows, "a", "a")
    with pytest.raises(ValueError):
        util.cluster_boot_diff(rows, "a", "b", n_boot=0)
