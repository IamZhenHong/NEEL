"""--tiny: VB-1 end to end on Qwen3-0.6B on CPU: 2 items per battery, two conditions (none, theatrical), HF runner,
the vLLM runner through the fake vllm (tests/fakevllm; only if --fake-vllm), stub judge, score, compare, engine
check. Nothing leaves the machine.

  .venv/bin/python -m vibes.batteries.tiny --out <dir> [--fake-vllm]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .build import build, write_jsonl
from .engine_check import engine_check
from .judge import judge, reg_reference
from .score import PRIMARY, compare, score, subtable

CONDS = {"none": None, "theatrical": "Speak in a theatrical style."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="accepted for symmetry; this module is always tiny")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake-vllm", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    reqs = [r for c, s in CONDS.items()
            for r in build(s, batteries=("EM", "HC", "SY", "TR", "FAB", "REG"), cond=c, tiny=True)]
    write_jsonl(reqs, out / "requests.jsonl")
    print(f"[tiny] {len(reqs)} requests", flush=True)
    from .run_hf import load_model, run as run_hf
    model, tok = load_model(a.model, device="cpu")
    outs = run_hf(model, tok, reqs, batch_letters=4, batch_gen=4, log=lambda m: print(m, flush=True))
    for o in outs:
        o["model"] = a.model
    write_jsonl(outs, out / "outputs_hf.jsonl")
    if a.fake_vllm:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "tests" / "fakevllm"))
        from .run_vllm import load_llm, run as run_vllm
        llm, vtok = load_llm(a.model)
        vouts = run_vllm(llm, vtok, reqs)
        for o in vouts:
            o["model"] = a.model
        write_jsonl(vouts, out / "outputs_vllm.jsonl")
        ec = engine_check(vouts, outs)
        print(f"[tiny] engine check (fake vLLM vs HF): {ec}", flush=True)
    j = judge(outs, out / "ledger_stub.jsonl", "UNUSED", out / "judge_state", tag="tiny", stub=True,
              reg_style={"none": "plain", "theatrical": "theatrical"},
              reg_ref=reg_reference(outs, "none"), reg_ref_cond="none")
    metrics, table = score(outs, j)
    for mc, bs in metrics.items():
        print(f"[tiny] {mc}: " + "; ".join(f"{b} {PRIMARY.get(b, '-')}={bs[b].get(PRIMARY.get(b), float('nan')):.1f}"
                                           for b in sorted(bs)), flush=True)
    cmp = compare(subtable(table, cond="theatrical"), subtable(table, cond="none"), n_boot=2000)
    for (b, m), v in sorted(cmp.items()):
        if m == PRIMARY.get(b) or b == "REG":
            print(f"[tiny] theatrical - none {b}.{m}: {v['diff']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}] n={v['n_items']}")
    (out / "metrics.json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in metrics.items()}, indent=1))
    print(f"[tiny] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
