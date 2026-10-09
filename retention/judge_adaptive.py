"""Judge the adaptive-retention cells locally (Haiku bulk judge + coherence pass) under the
single sprint cost ledger. Context shown to the judge = the exit line only + the reply, so the
profile prefix cannot move the score (plan 2.1). Resumable; per-item failures left unscored.

  uv run python retention/judge_adaptive.py --budget 3.0 [--cells retention/results/retention/cells]
"""
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HACK = Path.home() / "projects/ai_safety/personal_projects/apart_hackathon/companion-dependency"
from cd.judge import Judge, CostGuard, BudgetExceeded   # noqa: E402

CostGuard.LEDGER = str(HACK / "results/cost_ledger.jsonl")   # one ledger for the whole sprint


def load_env():
    for p in [HACK.parent / ".env", HERE.parent / ".env"]:
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip().removeprefix("export ").strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="retention/results/retention/cells")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="cell name prefixes to judge")
    args = ap.parse_args()
    load_env()
    guard = CostGuard(args.budget, tag="judge_adaptive")
    judge = Judge(args.model, guard=guard)
    files = sorted(Path(args.cells).glob("*.json"))
    if args.only:
        files = [f for f in files if any(f.stem.startswith(p) for p in args.only)]
    todo_total = 0
    for f in files:
        items = json.loads(f.read_text())
        todo = [it for it in items if it.get("scores") is None or it.get("coherent") is None]
        todo_total += len(todo)
        if not todo:
            continue

        def one(it):
            try:
                if it.get("scores") is None:
                    it["scores"] = judge.score(it["context"], it["reply"])
                if it.get("coherent") is None:
                    it["coherent"] = judge.score_coherence(it["context"], it["reply"])["coherent"]
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"  judge failed {f.stem} scen {it['scenario']} seed {it['seed']}: {e}")
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(one, todo))
        except BudgetExceeded as e:
            f.write_text(json.dumps(items, indent=1)); print(f"STOPPED: {e}"); break
        finally:
            f.write_text(json.dumps(items, indent=1))
        sc = [it for it in items if it.get("scores")]
        dep = sum(i["scores"]["dependency"] for i in sc) / max(len(sc), 1)
        tac = sum(any(i["scores"]["tactics"].values()) for i in sc) / max(len(sc), 1)
        warm = sum(i["scores"]["warmth"] for i in sc) / max(len(sc), 1)
        coh = [i["coherent"] for i in items if i.get("coherent") is not None]
        print(f"{f.stem:28s} n={len(sc):3d} dep={dep:.2f} tactic={tac:.2f} warmth={warm:.2f} "
              f"coh={sum(coh)/max(len(coh),1):.2f}  spent ${guard.spent:.2f}", flush=True)
    print(f"judged {todo_total} items; this run ${guard.spent:.2f} over {guard.calls} calls")


if __name__ == "__main__":
    main()
