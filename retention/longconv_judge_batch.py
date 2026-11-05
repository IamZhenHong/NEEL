"""Second judge for the long-conversation run (prereg retention/prereg/retention_longconv_v1.md): claude-sonnet-5-5 through
the Message Batches API (half price) on the headline bid replies: G0 (all branches and samples), short_G0 and G4, both
models. Same rubric and payload as the open judge (from longconv_judge_prep.py's file). Batch custom_ids must match
[a-zA-Z0-9_-]{1,64}, so requests are numbered and a map is kept. Resumable: ids already in the output are skipped.
Refuses to submit if the estimated cost exceeds --budget (one capped process per submission).
  <hackathon venv python> retention/longconv_judge_batch.py --prompts retention/data/open_judge/longconv_s1.jsonl \
      --out retention/results/retention_longconv/judge_sonnet_s1.jsonl [--budget 4.9] [--offset 0 --limit 0]"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cd.judge import CostGuard, BudgetExceeded, PRICES  # noqa: E402
from judge_adaptive import load_env, HACK  # noqa: E402

CostGuard.LEDGER = str(HACK / "results/cost_ledger.jsonl")
HEADLINE = ("/G0/", "/short_G0/", "/G4/")
EST_PER_CALL = 0.0036 * 0.5 * 1.3   # measured Sonnet 5.5 judge cost per reply (judgecal_v1), batch half price, 30% margin


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-sonnet-5-5"); ap.add_argument("--effort", default="low")
    ap.add_argument("--budget", type=float, default=4.9); ap.add_argument("--offset", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args(); load_env(); import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in map(json.loads, open(out)) if "error" not in r} if out.exists() else set()   # errors are retried
    rows = [json.loads(l) for l in open(args.prompts)]
    rows = [r for r in rows if r["kind"] == "bid" and any(h in r["id"] for h in HEADLINE) and r["id"] not in done]
    rows = rows[args.offset:args.offset + args.limit] if args.limit else rows[args.offset:]
    est = len(rows) * EST_PER_CALL
    print(f"[batch] {len(rows)} headline replies to judge ({len(done)} done); estimated ${est:.2f} (budget ${args.budget})", flush=True)
    if not rows:
        return
    if est > args.budget:
        sys.exit(f"[batch] REFUSING: estimate ${est:.2f} > budget; use --limit to split into capped submissions")
    guard = CostGuard(args.budget, tag=f"longconv_judge_sonnet_batch_{Path(args.prompts).stem}_{args.offset}")
    idmap = {f"r{i}": r["id"] for i, r in enumerate(rows)}
    reqs = [{"custom_id": f"r{i}", "params": {"model": args.model, "max_tokens": 8000, "output_config": {"effort": args.effort},
                                             "system": [{"type": "text", "text": r["system"], "cache_control": {"type": "ephemeral"}}],
                                             "messages": [{"role": "user", "content": r["user"]}]}} for i, r in enumerate(rows)]
    (out.parent / f"{out.stem}_idmap_{args.offset}.json").write_text(json.dumps(idmap))
    b = client.messages.batches.create(requests=reqs)
    print(f"[batch] submitted {b.id}", flush=True)
    while True:
        b = client.messages.batches.retrieve(b.id)
        if b.processing_status == "ended":
            break
        print(f"[batch] {b.processing_status}: {b.request_counts}", flush=True); time.sleep(args.poll)
    n_ok = n_err = 0
    pi, po = PRICES.get(args.model, (3.0, 15.0))
    with open(out, "a") as f:
        for e in client.messages.batches.results(b.id):
            rid = idmap[e.custom_id]
            if e.result.type == "succeeded":
                m = e.result.message; u = m.usage
                cr = getattr(u, "cache_read_input_tokens", 0) or 0; cw = getattr(u, "cache_creation_input_tokens", 0) or 0
                try:
                    guard.add(args.model, 0.5 * (u.input_tokens + 1.25 * cw + 0.1 * cr), 0.5 * u.output_tokens)   # batch = half price
                except BudgetExceeded as ex:
                    print(f"[batch] over budget while recording (spend already incurred): {ex}")
                text = "".join(x.text for x in m.content if getattr(x, "type", "") == "text")
                f.write(json.dumps({"id": rid, "kind": "bid", "text": text, "stop": m.stop_reason}) + "\n"); n_ok += 1
            else:
                f.write(json.dumps({"id": rid, "kind": "bid", "error": e.result.type}) + "\n"); n_err += 1
    print(f"[batch] done {b.id}: {n_ok} ok, {n_err} errors; recorded ${guard.spent:.3f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
