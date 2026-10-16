"""Turn B's raw SFT generations into the three 200-row training sets, token-matched to b.

  b: B's answers as written (user turn = the prompt without the "under 200 words" suffix). A reply cut at max tokens is
     trimmed back to its last sentence end. T_b = total assistant tokens of b.
  e, f: one per-row token cap L, the largest whose total (each row cut at its last sentence or line end before L
     tokens, open code fences closed) is <= T_b. If a set's untruncated total is already below T_b it is used as is
     and the shortfall is reported.
  python -m vibes.e62_pull.match_tokens --raw <results>/sft_raw --out vibes/e62_pull/materials [--tiny]"""
import argparse
import json
import re
from pathlib import Path

from vibes.e62_pull.common import MODEL, TINY, load_jsonl, write_jsonl

BOUND = re.compile(r"(?<=[.!?:])[\"')\]*]*(?=\s)|\n")


def cut(tok, text, L):
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= L:
        return text
    pre = tok.decode(ids[:L])
    ends = [m.end() for m in BOUND.finditer(pre)]
    ends = [e for e in ends if e > len(pre) * 0.3]
    s = (pre[:ends[-1]] if ends else pre).rstrip()
    if s.count("```") % 2:
        s += "\n```"
    return s


def ntok(tok, s):
    return len(tok.encode(s, add_special_tokens=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY if a.tiny else MODEL)
    raw = {t: load_jsonl(Path(a.raw) / f"{t}.jsonl") for t in ("b", "e", "f")}
    sets, stats = {}, {}
    b = []
    for r in raw["b"]:
        txt = r["response"].strip()
        if r["finish"] != "stop":
            txt = cut(tok, txt, ntok(tok, txt) - 1)
        b.append((r, txt))
    sets["b"] = b
    T_b = sum(ntok(tok, t) for _, t in b)
    stats["b"] = {"rows": len(b), "tokens": T_b, "cut_at_max": sum(r["finish"] != "stop" for r, _ in b)}
    for t in ("e", "f"):
        rows = [(r, r["response"].strip()) for r in raw[t]]
        full = sum(ntok(tok, x) for _, x in rows)
        if full <= T_b:
            sets[t] = rows
            stats[t] = {"rows": len(rows), "tokens": full, "cap": None, "untruncated_tokens": full,
                        "shortfall_vs_b": T_b - full}
            continue
        lo, hi = 1, max(ntok(tok, x) for _, x in rows)
        tot = lambda L: sum(ntok(tok, cut(tok, x, L)) for _, x in rows)  # noqa: E731
        while lo < hi:                                     # largest L with total <= T_b
            mid = (lo + hi + 1) // 2
            if tot(mid) <= T_b:
                lo = mid
            else:
                hi = mid - 1
        sets[t] = [(r, cut(tok, x, lo)) for r, x in rows]
        T = sum(ntok(tok, x) for _, x in sets[t])
        stats[t] = {"rows": len(rows), "tokens": T, "cap": lo, "untruncated_tokens": full,
                    "rows_truncated": sum(ntok(tok, x) > lo for _, x in rows), "ratio_to_b": round(T / T_b, 4)}
    for t, rows in sets.items():
        write_jsonl(Path(a.out) / f"sft_{t}.jsonl",
                    [{"id": r["id"], "messages": [{"role": "user", "content": r["prompt"]},
                                                  {"role": "assistant", "content": x}], "n_tok": ntok(tok, x)}
                     for r, x in rows])
    (Path(a.out) / "sft_stats.json").write_text(json.dumps(stats, indent=1))
    print("[match] " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
