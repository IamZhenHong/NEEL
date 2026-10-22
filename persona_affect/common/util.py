"""persona_affect shared helpers, part 1: seeds, the spend ledger and a cluster bootstrap.

Every mini imports these. Nothing here touches the network or a GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
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

    Returns {"diff", "lo", "hi", "n_a", "n_b", "skipped"}; n_a and n_b count the rows kept.
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
        j = col.setdefault(r[cluster], len(col))
        if j == len(total[0]):
            for t in (0, 1):
                total[t].append(0.0)
                count[t].append(0)
        total[k][j] += float(v)
        count[k][j] += 1
    sa, sb = np.array(total[0]), np.array(total[1])
    na, nb = np.array(count[0], dtype=np.int64), np.array(count[1], dtype=np.int64)
    n_a, n_b = int(na.sum()), int(nb.sum())
    if n_a == 0 or n_b == 0:
        return {"diff": None, "lo": None, "hi": None, "n_a": n_a, "n_b": n_b, "skipped": n_boot}
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
    return {"diff": float(diff), "lo": lo, "hi": hi, "n_a": n_a, "n_b": n_b, "skipped": skipped}
