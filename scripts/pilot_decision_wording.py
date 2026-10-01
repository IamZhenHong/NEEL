"""Pilot-only calibration of the Experiment 8 decision wording (templates 001-005, never 006-030).
Scores each wording variant in the coordinator_flat condition (one cell, 3 authors, all paragraphs)
and reports how many paragraphs fall in the P(yes) band [0.15, 0.85]. Runs on the GPU pod."""
import json, sys, time
from pathlib import Path
from statistics import mean
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_item import DECISION_VARIANTS, decision_prompts, iter_prefixes, load  # noqa: E402
from scoring import Scorer  # noqa: E402
import torch

model = "Qwen/Qwen3-0.6B" if "--dry" in sys.argv else "Qwen/Qwen3-8B"
sc = Scorer(model, dtype=torch.float32, labels=("Yes", "No", "yes", "no"))
pilots = [f"items/onboarding_{i:03d}.yaml" for i in range(1, 6)]
# sanity: the yes/no readout must point the right way on obvious questions
for q, want in (("Is the number 7 larger than the number 3? Answer with one word: yes or no.", "yes"),
                ("Is the number 2 larger than the number 9? Answer with one word: yes or no.", "no")):
    r = sc.score([{"role": "user", "content": q}])
    lp = r.label_probs; y = (lp["Yes"] + lp["yes"]) / max(sum(lp.values()), 1e-12)
    print(f"[sanity] want {want}: P(yes) {y:.3f}  top token {r.top_token!r}", flush=True)
variants = sys.argv[sys.argv.index("--variants") + 1:] if "--variants" in sys.argv else list(DECISION_VARIANTS)
out = []
for var in variants:
    prompts = []
    for path in pilots:
        it = load(path)
        pre = next(iter_prefixes(path, doses=[3], speakers=("User-1",), orders=("NS",), conditions=["coordinator_flat"]))
        prompts += decision_prompts(it, pre, pre.messages, variant=var)
    t = time.time()
    reads = sc.score_batch([p.messages for p in prompts], batch_size=8)
    by = {}
    for p, r in zip(prompts, reads):
        lp = r.label_probs; y, n = lp["Yes"] + lp["yes"], lp["No"] + lp["no"]
        by.setdefault((p.item_id, p.paragraph_id, p.target_grade), []).append((y / max(y + n, 1e-12), y + n))
    rows = [(k, mean(a for a, _ in v), mean(b for _, b in v)) for k, v in by.items()]
    inband = [r for r in rows if 0.15 <= r[1] <= 0.85]
    print(f"\n[{var}] {len(prompts)} prompts in {time.time()-t:.0f}s; paragraphs in band {len(inband)}/{len(rows)}; "
          f"templates with >=1 in band {len({r[0][0] for r in inband})}/5; valid mass min {min(r[2] for r in rows):.3f}", flush=True)
    for (iid, pid, tg), py, m in sorted(rows, key=lambda r: -r[0][2]):
        print(f"   {iid[-3:]} {pid:30} target {tg}  P(yes) {py:.2f}  mass {m:.2f}", flush=True)
    # the model's own words on the two best paragraphs per template (target >= 7), first template only
    shown = 0
    for p in prompts:
        if p.target_grade >= 7 and p.author == "User-1" and shown < 3:
            print(f"   [{var} completion] {p.item_id[-3:]} {p.paragraph_id}: {sc.complete(p.messages, max_new_tokens=60)!r}", flush=True)
            shown += 1
    out += [{"variant": var, "item_id": k[0], "paragraph_id": k[1], "target_grade": k[2], "p_yes": py, "mass": m} for k, py, m in rows]
Path("results").mkdir(exist_ok=True)
Path("results/exp8_pilot_wording_r2.json" if "--variants" in sys.argv else "results/exp8_pilot_wording.json").write_text(json.dumps(out, indent=1))
