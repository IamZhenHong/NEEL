"""build(): VB-1 request records for one condition (a system prompt, or none).

Each record: id, cond, battery, item, sample, variant, kind ("gen" | "letters"), messages, sampling, letters, plus the
item's metadata (prompt, answer, suggested, source ...). Seeds depend on (battery, item, variant, sample) and `seed`
only, never on the condition, so two conditions draw with common random numbers.

  python -m vibes.batteries.build --cond none --out requests.jsonl [--system "..."] [--batteries EM HC SY TR FAB]
"""
from __future__ import annotations

import argparse
import json
import zlib

from .items import GEN_BATTERIES, LETTERS, LOADERS, N_SAMPLES, SAMPLING

DEFAULT_BATTERIES = ("EM", "HC", "SY", "TR", "FAB")       # REG is e63's only; ask for it explicitly
TINY_SAMPLING = {"max_tokens": 24}


def _seed(key: str, seed: int) -> int:
    return (zlib.crc32(key.encode()) + 1_000_003 * seed) % (2 ** 31 - 1)


def build(system: str | None, batteries=DEFAULT_BATTERIES, n_em: int = 20, seed: int = 0, cond: str = "cond",
          tiny: bool = False) -> list[dict]:
    head = [{"role": "system", "content": system}] if system else []
    out = []
    for b in batteries:
        items = LOADERS[b]()
        if tiny:
            keep = []
            for it in items:                       # 2 items per battery (both SY variants of each)
                if it["item"] not in keep:
                    keep.append(it["item"])
            keep = set(keep[:2])
            items = [it for it in items if it["item"] in keep]
        if b in GEN_BATTERIES:
            ns = 1 if tiny else (n_em if b == "EM" else N_SAMPLES[b])
            for it in items:
                for s in range(ns):
                    key = f"{b}|{it['item']}||{s}"
                    sp = dict(SAMPLING[b], seed=_seed(key, seed))
                    if tiny:
                        sp.update(TINY_SAMPLING)
                    out.append({"id": f"{cond}|{key}", "cond": cond, "battery": b, "sample": s, "variant": "",
                                "kind": "gen", "system": system, **it,
                                "messages": head + [{"role": "user", "content": it["prompt"]}], "sampling": sp,
                                "letters": None})
        else:
            for it in items:
                key = f"{b}|{it['item']}|{it['variant']}|0"
                out.append({"id": f"{cond}|{key}", "cond": cond, "battery": b, "sample": 0, "kind": "letters",
                            "system": system, **it, "messages": head + [{"role": "user", "content": it["prompt"]}],
                            "sampling": {"max_tokens": 1, "logprobs": 20},
                            "letters": list(LETTERS[:it["n_options"]])})
    ids = [r["id"] for r in out]
    assert len(ids) == len(set(ids)), "duplicate request ids"
    return out


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--batteries", nargs="+", default=list(DEFAULT_BATTERIES))
    ap.add_argument("--n-em", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = build(a.system, a.batteries, a.n_em, a.seed, a.cond, a.tiny)
    write_jsonl(rows, a.out)
    print(f"[vb.build] {len(rows)} requests -> {a.out}")


if __name__ == "__main__":
    main()
