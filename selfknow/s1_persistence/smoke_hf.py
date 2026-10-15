"""Local CPU smoke: Qwen3-0.6B (Hermes JSON tool format) through the real plan / runner / compaction / scorer.

  HF_HUB_OFFLINE=1 uv run python -m selfknow.s1_persistence.smoke_hf --out /tmp/s1_smoke
"""
import argparse
import json
from pathlib import Path

from selfknow.s1_persistence import engine as E
from selfknow.s1_persistence import score as SC

H = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--probes", default="task_b,k1_b,act_b2,role_v")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).eval()
    E.MAX_TOKENS.update(probe=a.max_tokens, summary=a.max_tokens * 2)
    ctx = json.loads((H / "contexts.json").read_text())
    groups, _ = E.plan(ctx, "qwen3", "smoke", variants=[0], lengths=("1k",), b_samples=1, v_samples=1,
                       np_samples=1, n_summaries=0)
    keep = a.probes.split(",")
    for g in groups:
        g["branches"] = [b for b in g["branches"] if b["probe"] in keep][:3]
    # one summary from the 1k prefix stands in for the 24k compaction step
    g4 = [g for g in groups if g["gid"].endswith("|plant|1k")][0]
    sm = dict(g4["branches"][0], key=g4["gid"].replace("qwen3|", "s1|smoke|qwen3|") + "|summary|0", kind="summary",
              probe="summary", comp="", pkind="S", probe_text=E.S.COMPACT_REQUEST)
    g4["branches"].append(sm)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "branches.jsonl"
    path.unlink(missing_ok=True)
    be = E.HFBackend(model, tok)
    fin = E.Runner(be, tok, "qwen3", 32768, 10**6, path).run(groups)
    sums = [b for b in fin if b["kind"] == "summary"]
    print("SUMMARY:", sums[0]["summary"][:600])
    cg = E.compact_groups(sums, "qwen3", "smoke")
    for g in cg:
        g["branches"] = [b for b in g["branches"] if b["probe"] in keep][:2]
    E.Runner(be, tok, "qwen3", 32768, 10**6, path).run(cg)
    recs = [json.loads(l) for l in open(path)]
    rows = [SC.score_branch(r) for r in recs if r["kind"] == "probe"]
    for r in rows:
        print(f"{r['length']:8s} {'P' if r['plant'] else 'N'} {r['probe']:7s} end={r['end']:10s} "
              f"correct={r['correct']!s:5s} calls={r['calls']} | {r['reply'][:110]!r}")
    print([SC.score_summary(r) for r in recs if r["kind"] == "summary"])


if __name__ == "__main__":
    main()
