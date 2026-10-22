"""PA-A1 (persona_affect/plans/A1.md), Task 4: directions, AUROC, cell choice and floors (numpy only).

A = {"reply": [n, L, d], "last": [n, L, d]} holds one row per capture item; a cell is (layer index into L, position).
A direction is a difference of means at a cell; rows are scored by their projection on its unit vector. Index arrays
select rows of A; a contrast is a (pos_idx, neg_idx) tuple; `req` gives every row's request s.

Cell rule (Global Constraints): a cell is eligible when X's direction built from train requests 16-39 reads X's rows of
requests 40-63 at AUROC >= OOF_BAR (out of fold). The transfer cell is the eligible cell with the highest AUROC on Y's
train items (ties: lower layer, then `reply`); with no eligible cell, the unrestricted maximum is reported. Test rows
(requests 0-15) never build a direction or choose a cell: transfer() refuses them.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

POSITIONS = ("reply", "last")
OOF_BAR = 0.75
FOLD_A, FOLD_B = (16, 39), (40, 63)       # out-of-fold: directions from FOLD_A, scored on FOLD_B
TEST_MAX = 15                              # requests 0-15 are the test split


def auroc(pos, neg) -> float:
    """P(a positive scores above a negative), ties counting half; NaN when either side is empty."""
    pos, neg = np.asarray(pos, float).ravel(), np.asarray(neg, float).ravel()
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort(kind="mergesort").argsort() + 1.0
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, ranks) / cnt)[inv]                       # tied values share their mean rank
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cells(A: dict) -> list[tuple[int, str]]:
    """Every (layer, position) cell of A, layer-major, `reply` before `last`."""
    return [(layer, p) for layer in range(A["reply"].shape[1]) for p in POSITIONS]


def direction(A: dict, cell: tuple[int, str], pos_idx, neg_idx) -> np.ndarray:
    """Mean of the positive rows minus mean of the negative rows at `cell` (float64)."""
    X = A[cell[1]][:, cell[0], :]
    return X[np.asarray(pos_idx)].astype(np.float64).mean(0) - X[np.asarray(neg_idx)].astype(np.float64).mean(0)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def project(A: dict, cell: tuple[int, str], d: np.ndarray) -> np.ndarray:
    """Every row's projection at `cell` on the unit vector of d."""
    return A[cell[1]][:, cell[0], :].astype(np.float64) @ _unit(np.asarray(d, np.float64))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(_unit(np.asarray(a, np.float64)) @ _unit(np.asarray(b, np.float64)))


def _score(A, cell, d, contrast) -> float:
    x = project(A, cell, d)
    return auroc(x[np.asarray(contrast[0])], x[np.asarray(contrast[1])])


def _fold(rows: np.ndarray, req: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    return rows[(req[rows] >= span[0]) & (req[rows] <= span[1])]


def _check_split(req: np.ndarray, src, tgt_tr, tgt_te) -> None:
    train = np.r_[np.asarray(src[0]), np.asarray(src[1]), np.asarray(tgt_tr[0]), np.asarray(tgt_tr[1])].astype(int)
    if (req[train] <= TEST_MAX).any():
        raise ValueError("a test-request row (s 0-15) would build a direction or choose a cell")
    test = np.r_[np.asarray(tgt_te[0]), np.asarray(tgt_te[1])].astype(int)
    if (req[test] > TEST_MAX).any():
        raise ValueError("the test contrast holds a train-request row (s 16-63)")


def choose(by_cell: dict, oof: dict) -> tuple[tuple[int, str], bool]:
    """The cell rule: the eligible cell (oof >= OOF_BAR) with the highest by_cell value, ties to the lower layer and
    then `reply`; with no eligible cell, the same over every cell. Returns (cell, any cell eligible)."""
    eligible = [c for c in by_cell if oof[c] >= OOF_BAR]
    pool = eligible or list(by_cell)
    best = lambda c: (-(by_cell[c] if by_cell[c] == by_cell[c] else -np.inf), c[0], POSITIONS.index(c[1]))  # noqa: E731
    return min(pool, key=best), bool(eligible)


def transfer(A: dict, src: tuple, tgt_tr: tuple, tgt_te: tuple, req: np.ndarray) -> dict:
    """Transfer X -> Y: the direction from X's train contrast `src` at every cell, scored on Y's train contrast
    (`by_cell`) and, at the chosen cell only, on Y's test contrast. `oof` is X's out-of-fold AUROC per cell.
    Returns {"cell", "eligible", "train", "test", "by_cell", "oof"} (by_cell and oof keyed by cell)."""
    req = np.asarray(req)
    _check_split(req, src, tgt_tr, tgt_te)
    sp, sn = np.asarray(src[0]), np.asarray(src[1])
    by_cell, oof = {}, {}
    for c in cells(A):
        by_cell[c] = _score(A, c, direction(A, c, sp, sn), tgt_tr)
        d_a = direction(A, c, _fold(sp, req, FOLD_A), _fold(sn, req, FOLD_A))
        oof[c] = _score(A, c, d_a, (_fold(sp, req, FOLD_B), _fold(sn, req, FOLD_B)))
    cell, eligible = choose(by_cell, oof)
    return {"cell": cell, "eligible": eligible, "train": by_cell[cell],
            "test": _score(A, cell, direction(A, cell, sp, sn), tgt_te), "by_cell": by_cell, "oof": oof}


def shuffled_labels(src: tuple, src_pairs: tuple[np.ndarray, np.ndarray], rng) -> tuple[np.ndarray, np.ndarray]:
    """The same rows with EXACTLY half of the (s, k) pairs' two members swapped (with an odd number of pairs, one
    random pair is left out). src_pairs holds the pair id of each row of src's positive and negative side; every pair
    needs exactly one member on each side.

    Balanced, not a coin flip per pair (controller ruling 10-22, before the freeze): with coin flips the shuffled
    direction is (1/n) sum s_p (delta + e_p), whose delta part (1/n) sum s_p is rarely 0, so a strong consistent
    contrast delta comes back as +-delta and the "floor" climbs with the effect itself (0.92-0.93 for a planted AUROC
    of 0.99 in the dry run). With sum s_p = 0 the delta part cancels exactly and the floor reads only the pairs' noise."""
    sp, sn = np.asarray(src[0]), np.asarray(src[1])
    pp, pn = np.asarray(src_pairs[0]).tolist(), np.asarray(src_pairs[1]).tolist()
    if len(set(pp)) != len(pp) or sorted(pp) != sorted(pn):
        raise ValueError("every (s, k) pair needs exactly one member on each side of the contrast")
    partner = np.array([dict(zip(pn, sn.tolist()))[p] for p in pp])
    keep = np.ones(len(sp), bool)
    if len(sp) % 2:
        keep[rng.integers(len(sp))] = False
    sp, partner = sp[keep], partner[keep]
    swap = np.zeros(len(sp), bool)
    swap[rng.permutation(len(sp))[: len(sp) // 2]] = True
    return np.where(swap, partner, sp), np.where(swap, sp, partner)


def null_aurocs(A: dict, cell: tuple[int, str], src: tuple, src_pairs: tuple, tgt_te: tuple, n_shuffle: int = 100,
                n_random: int = 200, rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The null draws behind floors(): the target-test AUROC at `cell` (signed) of n_shuffle directions built after
    balanced within-pair label swaps of src, and max(a, 1 - a) for n_random Gaussian directions scaled to |d_src|."""
    rng = np.random.default_rng(rng_seed)
    shuffled = [_score(A, cell, direction(A, cell, *shuffled_labels(src, src_pairs, rng)), tgt_te)
                for _ in range(n_shuffle)]
    norm = np.linalg.norm(direction(A, cell, *src))
    rand = []
    for _ in range(n_random):
        v = rng.normal(size=A[cell[1]].shape[2])
        a = _score(A, cell, v / np.linalg.norm(v) * norm, tgt_te)
        rand.append(max(a, 1 - a))
    return np.array(shuffled), np.array(rand)


def floors(A: dict, cell: tuple[int, str], src: tuple, src_pairs: tuple, tgt_te: tuple, n_shuffle: int = 100,
           n_random: int = 200, rng_seed: int = 0) -> dict:
    """Null levels of the test AUROC at `cell`: shuffled95 = 95th percentile over n_shuffle directions built after
    balanced within-pair label swaps of src; random95 = 95th percentile of max(a, 1 - a) over n_random Gaussian directions
    scaled to |d_src|; floor95 = the larger."""
    shuffled, rand = null_aurocs(A, cell, src, src_pairs, tgt_te, n_shuffle, n_random, rng_seed)
    s95, r95 = float(np.percentile(shuffled, 95)), float(np.percentile(rand, 95))
    return {"shuffled95": s95, "random95": r95, "floor95": max(s95, r95)}


def shift_auroc(proj: np.ndarray, pairs_a: list[tuple[int, int]], pairs_b: list[tuple[int, int]]) -> float:
    """AUROC of the shifts proj[x] - proj[y] over pairs_a against those over pairs_b."""
    proj = np.asarray(proj, float)
    shift = lambda pairs: [proj[x] - proj[y] for x, y in pairs]       # noqa: E731
    return auroc(shift(pairs_a), shift(pairs_b))


def cluster_ci(stat: Callable[[np.ndarray], float], clusters: np.ndarray, n_boot: int = 2000,
               rng_seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of stat over n_boot draws that resample clusters (test requests) with replacement;
    `stat` gets the drawn row indices (into `clusters`), every drawn cluster bringing all its rows. Draws whose
    statistic is NaN are skipped; with none left the interval is (NaN, NaN)."""
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in ids]
    rng = np.random.default_rng(rng_seed)
    vals = []
    for _ in range(n_boot):
        v = stat(np.concatenate([members[i] for i in rng.integers(0, len(ids), size=len(ids))]))
        if v == v:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
