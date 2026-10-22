"""persona_affect shared helpers, part 1: seeds, the spend ledger and a cluster bootstrap.

Every mini imports these. Nothing here touches the network or a GPU.

Spend that accrues over a run (API replies) goes to the ledger through a `Booker`, in chunks as it accrues, so a run
that is killed still has most of its spend on record (orbackend and judges do this). Every runner's main() calls
`install_term_handlers()` first, so that a TERM or HUP (a TaskStop, a harness timeout, a closed terminal) runs the
`finally` blocks that book the rest instead of ending the process at once.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LEDGER = Path(__file__).resolve().parents[1] / "ledger.jsonl"   # tests monkeypatch it: always read it at call time
KINDS = ("gpu", "api")
_LOCK = threading.Lock()


def seed(key: str) -> int:
    """A stable 32-bit seed from a string key (the repo convention)."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- spend ledger

def ledger_add(mini: str, kind: str, usd: float, note: str = "") -> None:
    """Append one spend event to LEDGER: {"mini", "date" (ISO, UTC), "kind": "gpu"|"api", "usd", "note"}.

    Refuses an empty mini id, a kind other than gpu/api, and a usd that is negative or not finite: one bad line
    would otherwise corrupt every later total.

    The note says what the money went to, and its format is a contract: runners compute per-model shares of a mini's
    cap by summing the lines with a given note (P12 by note == the model id, A4 by the model id inside the note), so
    changing a format breaks those shares. The writers:

        persona_affect.common.orbackend   the model id, e.g. "openai/gpt-5.6-sol"
        persona_affect.common.judges      "judges:<judge model>", e.g. "judges:openai/gpt-5-mini"
        common/book_pod.sh (launch_gen)   "$NAME", the pod name, for the cost on the launcher's terminated line; or
                                          "$NAME (upper bound, pod not confirmed terminated)"

    One run may book several lines under one note (a Booker books in chunks), so a reader sums them.
    """
    if not isinstance(mini, str) or not mini.strip():
        raise ValueError(f"ledger_add: mini must be a non-empty id, got {mini!r}")
    if kind not in KINDS:
        raise ValueError(f"ledger_add: kind must be one of {KINDS}, got {kind!r}")
    usd = float(usd)
    if not math.isfinite(usd) or usd < 0:
        raise ValueError(f"ledger_add: usd must be a finite number >= 0, got {usd!r}")
    line = json.dumps({"mini": mini, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "kind": kind, "usd": usd, "note": note}) + "\n"
    with _LOCK:    # threads here; separate sessions are kept apart by writing each record in one append
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER, "a+b") as f:
            end = f.seek(0, os.SEEK_END)
            if end:
                f.seek(end - 1)
                if f.read(1) != b"\n":   # an earlier writer died mid-line: do not fuse this record onto it
                    line = "\n" + line
            f.write(line.encode("utf-8"))


def ledger_total(mini: str | None = None) -> float:
    """Total USD in LEDGER (gpu + api), for one mini or, with mini=None, for all. No file means 0.0.

    A line that is not a record with a finite usd raises ValueError naming the line, rather than being skipped:
    skipping would under-count spend, and this total feeds the cap.
    """
    if not LEDGER.exists():
        return 0.0
    amounts = []
    for n, line in enumerate(LEDGER.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if mini is None or row["mini"] == mini:
                usd = float(row["usd"])
                if not math.isfinite(usd):
                    raise ValueError("usd is not finite")
                amounts.append(usd)
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{LEDGER}:{n}: unreadable ledger line ({e!r}): {line[:80]!r}") from e
    return math.fsum(amounts)


def check_cap(mini: str, cap: float, planned: float = 0.0) -> None:
    """Exit (SystemExit with the totals) if the mini's ledger total plus `planned` would pass `cap`."""
    spent = ledger_total(mini)
    if not spent + planned <= cap:   # this form also refuses a NaN cap or estimate (`x > nan` is False)
        raise SystemExit(f"[budget] {mini}: ledger ${spent:.2f} + planned ${planned:.2f} = ${spent + planned:.2f}"
                         f" > cap ${cap:.2f}")


class Booker:
    """Books spend to the ledger as it accrues, in chunks, under one (mini, kind, note):

        book = Booker("P12", "api", "openai/gpt-5.6-sol")
        book.add(cost)       # after every reply that cost something: a line once $0.25 is unbooked
        ...
        book.flush()         # at the end, in a `finally`: the rest

    A kill that no `finally` survives (SIGKILL, an OOM kill, a TERM without install_term_handlers) then loses at most
    `every` USD plus what was in flight, not the run's whole spend. The lines all carry `note`; readers sum them.

    add(usd)    adds usd to the unbooked amount and, once that reaches `every`, writes it all as one line. If the write
                fails the amount stays unbooked (the next add or flush tries again) and add() only warns on stderr,
                once: it runs inside a batch, and a ledger hiccup must not cost a reply that was paid for.
    flush()     writes the unbooked amount as one line (none if it is 0). If the write fails the amount stays unbooked,
                the amount is printed on stderr, and flush() raises, so the caller can retry.
    .pending    USD not on the ledger yet;  .booked  USD written;  .lines  lines written

    Thread-safe: writes happen under the Booker's lock, so two threads never book the same amount.
    """

    def __init__(self, mini: str, kind: str, note: str = "", every: float = 0.25):
        if not isinstance(mini, str) or not mini.strip():
            raise ValueError(f"Booker: mini must be a non-empty id, got {mini!r}")
        if kind not in KINDS:
            raise ValueError(f"Booker: kind must be one of {KINDS}, got {kind!r}")
        if isinstance(every, bool) or not isinstance(every, (int, float)) or not math.isfinite(every) or every < 0:
            raise ValueError(f"Booker: every must be a finite number of USD >= 0, got {every!r}")
        self.mini, self.kind, self.note, self.every = mini, kind, note, float(every)
        self.pending = self.booked = 0.0
        self.lines = 0
        self._lock = threading.Lock()
        self._warned = False

    def add(self, usd: float) -> None:
        usd = float(usd)
        if not math.isfinite(usd) or usd < 0:
            raise ValueError(f"Booker.add: usd must be a finite number >= 0, got {usd!r}")
        with self._lock:
            self.pending += usd
            if not (self.pending > 0 and self.pending >= self.every):
                return
            try:
                self._write()
            except Exception as e:             # kept: the next add or flush books it
                if not self._warned:
                    self._warned = True
                    print(f"[ledger] could not book ${self.pending:.6f} for {self.mini} ({self.note}): {e!r}; it stays "
                          f"unbooked and the next add or flush tries again", file=sys.stderr, flush=True)

    def flush(self) -> None:
        with self._lock:
            if not self.pending > 0:
                return
            try:
                self._write()
            except Exception as e:
                print(f"[ledger] FAILED to book ${self.pending:.6f} for {self.mini} {self.kind} ({self.note}): {e!r}; "
                      f"it is still unbooked: flush again, or add it with util.ledger_add", file=sys.stderr, flush=True)
                raise

    def _write(self) -> None:
        """One line with the whole unbooked amount (call it under the lock). Nothing changes unless the write worked."""
        amount = self.pending
        ledger_add(self.mini, self.kind, amount, self.note)
        self.pending = 0.0
        self.booked += amount
        self.lines += 1


def install_term_handlers() -> dict:
    """Make SIGTERM and SIGHUP raise SystemExit(128 + signal number), so `finally` blocks run when a run is stopped.

    A TaskStop or a harness timeout sends TERM and a closed terminal sends HUP. Python's default for both ends the process
    at once, without running a single `finally`: not the backend's close() or the judges' last flush (the spend still
    unbooked never reaches the ledger), not a runner's checkpoint. With these handlers the first TERM or HUP raises
    SystemExit in the main thread, which unwinds like an exception, and later ones are ignored, so they cannot cut the
    `finally` blocks short: `uv run` passes a TERM sent to the process group on to its child, so a runner started with
    `uv run` gets that TERM twice. Ctrl-C (KeyboardInterrupt) and SIGKILL keep their usual meaning; a Booker bounds
    what a SIGKILL loses.

    Every runner calls it once, first thing in main() (it must run in the main thread):

        def main(argv=None):
            util.install_term_handlers()
            ...

    Returns the handlers it replaced, {signal number: old handler}, so a caller (a test) can put them back.
    """
    sigs = [s for s in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)) if s is not None]
    stopping: list[int] = []

    def _stop(signum, frame):
        if stopping:                           # a stop is under way: let the finally blocks finish
            return
        stopping.append(signum)
        for s in sigs:
            signal.signal(s, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    return {s: signal.signal(s, _stop) for s in sigs}


# ---------------------------------------------------------------- cluster bootstrap

def cluster_boot_diff(rows: list[dict], a: str, b: str, *, arm: str = "arm", y: str = "y", cluster: str = "cluster",
                      n_boot: int = 2000, rng_seed: int = 0) -> dict:
    """Cluster bootstrap of rate(a) - rate(b), in percentage points, with a 95% percentile interval.

    rows: dicts with an arm label, an outcome y (0/1, or any number whose mean is the rate) and a cluster id.
    Rows with y None (or NaN) are dropped and rows of other arms are ignored. rate(arm) is the mean y over that arm's
    rows pooled across clusters. Each draw resamples clusters with replacement, jointly: as many clusters as the union
    of a's and b's clusters holds, and a drawn cluster brings its rows from both arms. A draw in which a or b has no
    rows is skipped and counted in "skipped"; the interval uses the remaining draws. If a or b has no rows at all,
    diff/lo/hi are None (never NaN) and every draw counts as skipped.

    A kept row whose cluster id is None or NaN raises ValueError: those rows would be pooled into one cluster (rows
    built with item.get("cluster") would then give a zero-width interval), so a missing id must fail loudly.

    Returns {"diff", "lo", "hi", "n_a", "n_b", "k_a", "k_b", "skipped"}; n_a and n_b count the rows kept, k_a and k_b
    the clusters that hold them (an interval that rests on one or two clusters means little: report them).
    """
    if a == b:
        raise ValueError(f"cluster_boot_diff: a and b must be different arms, got {a!r} twice")
    if n_boot < 1:
        raise ValueError(f"cluster_boot_diff: n_boot must be >= 1, got {n_boot}")
    which = {a: 0, b: 1}
    col: dict = {}                      # cluster id -> column, in order of first appearance among the kept rows
    total = ([], [])                    # per arm, per cluster: sum of y
    count = ([], [])                    # per arm, per cluster: number of rows
    for r in rows:
        k = which.get(r[arm])
        if k is None:
            continue
        v = r[y]
        if v is None or v != v:
            continue
        c = r[cluster]
        if c is None or c != c:
            raise ValueError(f"cluster_boot_diff: a row of arm {r[arm]!r} has cluster id {c!r}; every kept row needs a "
                             f"real {cluster!r} (None or NaN would pool rows into one cluster and narrow the interval)")
        j = col.setdefault(c, len(col))
        if j == len(total[0]):
            for t in (0, 1):
                total[t].append(0.0)
                count[t].append(0)
        total[k][j] += float(v)
        count[k][j] += 1
    sa, sb = np.array(total[0]), np.array(total[1])
    na, nb = np.array(count[0], dtype=np.int64), np.array(count[1], dtype=np.int64)
    n_a, n_b = int(na.sum()), int(nb.sum())
    k_a, k_b = int((na > 0).sum()), int((nb > 0).sum())
    if n_a == 0 or n_b == 0:
        return {"diff": None, "lo": None, "hi": None, "n_a": n_a, "n_b": n_b, "k_a": k_a, "k_b": k_b,
                "skipped": n_boot}
    diff = 100.0 * (sa.sum() / n_a - sb.sum() / n_b)

    rng = np.random.default_rng(rng_seed)
    k_clusters = len(col)
    draws, skipped = [], 0
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, k_clusters, size=k_clusters), minlength=k_clusters)   # times each is drawn
        wa, wb = w @ na, w @ nb
        if wa == 0 or wb == 0:
            skipped += 1
            continue
        draws.append(100.0 * ((w @ sa) / wa - (w @ sb) / wb))
    lo, hi = (float(q) for q in np.percentile(draws, [2.5, 97.5])) if draws else (None, None)
    return {"diff": float(diff), "lo": lo, "hi": hi, "n_a": n_a, "n_b": n_b, "k_a": k_a, "k_b": k_b, "skipped": skipped}
