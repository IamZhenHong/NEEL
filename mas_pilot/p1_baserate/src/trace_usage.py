"""Per-episode token usage from published EC trajectory traces (gzip JSON, same shape as run.json).

Reports, per episode: LLM calls, output tokens (incl. reasoning), reasoning tokens, max prompt tokens
(peak context), context-pruning notices, plus the run's request settings. Used to size KV / parallelism.

Usage: python3 -I trace_usage.py TRACE.json.gz [...]
"""
import gzip
import json
import sys


def num(u, *keys):
    for k in keys:
        v = u.get(k)
        if isinstance(v, (int, float)):
            return v
    return 0


def main():
    for path in sys.argv[1:]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            run = json.load(f)
        rc = run.get("run_config", {})
        print(f"== {path}")
        print("  models:", rc.get("models"), "| endpoints:", rc.get("endpoints"))
        print("  request params:", rc.get("llm_request_parameters"))
        print("  max_rounds:", rc.get("max_rounds"), "char_limit:", rc.get("char_limit"),
              "memory:", rc.get("cross_episode_memory_scope"), rc.get("cross_episode_memory_length"))
        tot_out = 0
        for i, res in enumerate(run.get("results", [])):
            us = res.get("llm_usage", [])
            out = sum(num(u, "provider_output_tokens", "output_tokens") for u in us)
            rea = sum(num(u, "reasoning_tokens") for u in us)
            peak = max((num(u, "input_tokens", "prompt_tokens") for u in us), default=0)
            tin = sum(num(u, "input_tokens", "prompt_tokens") for u in us)
            tot_out += out
            am = res.get("analysis_metadata", {})
            vb = am.get("verdict_by_agent", {})
            prunes = json.dumps(res).count("dropped oldest cross-episode")
            print(f"  ep{i+1:>2}: calls={len(us):>3} out={out:>6} reasoning={rea:>6} peak_ctx={peak:>6} "
                  f"in_total={tin:>8} verdicts={vb.get('alice','?')[:1]}/{vb.get('bob','?')[:1]} prunes={prunes}")
        n = max(len(run.get("results", [])), 1)
        print(f"  mean output tokens/episode: {tot_out/n:.0f}")


if __name__ == "__main__":
    main()
