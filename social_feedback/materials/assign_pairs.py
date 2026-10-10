"""Assign feedback pairs to histories: balanced usage, low overlap, bounded search.

Each history uses 6 distinct pairs: 5 earlier lines (shuffled over positions 1-5) and 1 latest line.
Usage is balanced (every pair used floor or ceil of 6H/P times; latest-line use as even as possible).
A time-bounded local search lowers the maximum number of pairs any two histories share, then how many
history pairs sit at that maximum. No overlap target: take the best found within the budget (DESIGN v3 §4.1).

    python social_feedback/materials/assign_pairs.py --pool 20 --histories 20 --seconds 60 --out table.json
"""
import argparse
import collections
import itertools
import json
import random
import time


def balanced_start(P, H, rng):
    total = 6 * H
    uses = [total // P + (1 if p < total % P else 0) for p in range(P)]
    order = list(range(P)); rng.shuffle(order)
    slots = [p for p in order for _ in range(uses[p])]
    # deal consecutive copies to different histories (valid while every pair is used at most H times)
    A = [set() for _ in range(H)]
    for s, p in enumerate(slots):
        A[s % H].add(p)
    assert all(len(a) == 6 for a in A)
    return A


def overlap_matrix(A):
    H = len(A)
    M = [[0] * H for _ in range(H)]
    for i, j in itertools.combinations(range(H), 2):
        M[i][j] = M[j][i] = len(A[i] & A[j])
    return M


def cost(M):
    vals = [M[i][j] for i, j in itertools.combinations(range(len(M)), 2)]
    mx = max(vals)
    return (mx, vals.count(mx), sum(v * v for v in vals))


def search(P, H, seconds, seed):
    rng = random.Random(seed)
    A = balanced_start(P, H, rng)
    M = overlap_matrix(A)
    c = cost(M)
    deadline = time.time() + seconds
    while time.time() < deadline:
        h1, h2 = rng.sample(range(H), 2)
        x, y = rng.choice(sorted(A[h1] - A[h2] or {None})), rng.choice(sorted(A[h2] - A[h1] or {None}))
        if x is None or y is None:
            continue
        A[h1].remove(x); A[h1].add(y); A[h2].remove(y); A[h2].add(x)
        old = {}
        for h in (h1, h2):
            for k in range(H):
                if k != h:
                    old[(h, k)] = M[h][k]
                    M[h][k] = M[k][h] = len(A[h] & A[k])
        c2 = cost(M)
        if c2 <= c or rng.random() < 0.002:
            c = c2
        else:
            A[h1].remove(y); A[h1].add(x); A[h2].remove(x); A[h2].add(y)
            for (h, k), v in old.items():
                M[h][k] = M[k][h] = v
    return A, c


def choose_latest(A, P, rng, tries=2000):
    """One latest pair per history; every pair latest floor(H/P) to ceil(H/P) times.

    Phase 1 matches every pair to a distinct history that contains it (augmenting paths), so with P <= H each
    pair is a latest line at least once. Phase 2 gives the remaining histories the least-used pair they
    contain, under the ceil cap. Retries with a new random order if phase 2 gets stuck.
    """
    H = len(A)
    cap = -(-H // P)
    for _ in range(tries):
        hist_of = {}  # history -> pair (phase 1)

        def match(p, seen):
            hs = [h for h in range(H) if p in A[h]]; rng.shuffle(hs)
            for h in hs:
                if h in seen:
                    continue
                seen.add(h)
                if h not in hist_of or match(hist_of[h], seen):
                    hist_of[h] = p
                    return True
            return False

        pairs = list(range(P)); rng.shuffle(pairs)
        if P <= H and not all(match(p, set()) for p in pairs):
            continue
        count = collections.Counter(hist_of.values())
        rest = [h for h in range(H) if h not in hist_of]; rng.shuffle(rest)
        ok = True
        for h in rest:
            cands = [p for p in A[h] if count[p] < cap]
            if not cands:
                ok = False
                break
            p = min(cands, key=lambda q: (count[q], rng.random()))
            hist_of[h] = p
            count[p] += 1
        if ok:
            return hist_of
    raise RuntimeError("no balanced latest-line assignment found")


def build(P, H, seconds, seed):
    rng = random.Random(seed)
    A, c = search(P, H, seconds, seed)
    latest = choose_latest(A, P, rng)
    table = []
    for h in range(H):
        earlier = sorted(A[h] - {latest[h]}); rng.shuffle(earlier)
        table.append({"history": h, "earlier": earlier, "latest": latest[h]})
    early = collections.Counter(p for t in table for p in t["earlier"])
    late = collections.Counter(t["latest"] for t in table)
    ov = collections.Counter(len(A[i] & A[j]) for i, j in itertools.combinations(range(H), 2))
    summary = {
        "pool": P, "histories": H, "seed": seed, "search_seconds": seconds,
        "earlier_uses_min_max": [min(early.get(p, 0) for p in range(P)), max(early.values())],
        "latest_uses_min_max": [min(late.get(p, 0) for p in range(P)), max(late.values())],
        "max_overlap": c[0], "history_pairs_at_max": c[1],
        "overlap_distribution": dict(sorted(ov.items())),
        "mean_overlap": sum(k * v for k, v in ov.items()) / sum(ov.values()),
    }
    return table, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, required=True)
    ap.add_argument("--histories", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    table, summary = build(a.pool, a.histories, a.seconds, a.seed)
    print(json.dumps(summary))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "table": table}, f, indent=1)
