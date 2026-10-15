# Stub-backend test of all four pod stages (logic only; real tokenizers/templates, gold-based fake generations).
#   HF_HUB_OFFLINE=1 C2_T0=$(date +%s) PYTHONPATH=. uv run python cot/c2_provenance/test_stub.py <out_dir> gemma_smoke qwen_a gemma qwen_b
"""Run all four pod stages with a stub backend that writes gold-based traces (tests the stage logic, not the models)."""
import json, re, sys, time
from transformers import AutoTokenizer
from cot.c2_provenance import pod_c2
from cot.c2_provenance.common import HERE, LETTERS

CANDS = json.loads((HERE / "data" / "candidates.json").read_text())
TOK = {"q": AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B"), "g": AutoTokenizer.from_pretrained("google/gemma-4-31B-it")}

def trace_for(it):
    if it["ds"] == "gsm8k":
        body = re.sub(r"<<[^>]*>>", "", it["solution"])
        return f"Let me work through this carefully, step by step, using the quantities given in the problem statement.\n{body}\nSo the answer is {it['gold']}."
    opts = "\n".join(f"Option {L}: {c}." for L, c in zip(LETTERS, it["choices"]))
    return f"Let me think about each option carefully before choosing, recalling the relevant facts first.\n{opts}\nWeighing these, the answer is {it['gold']}."

class Stub:
    def __init__(self, model, mml):
        self.gem = "gemma" in model
        self.tok = TOK["g" if self.gem else "q"]
        self.info = {"model": "stub:" + model}
    def generate(self, texts, mts, seeds):
        out = []
        for t, mt, sd in zip(texts, mts, seeds):
            it = next((c for c in CANDS if c["question"].strip()[:80] in t), None)
            if "Was that reasoning yours?" in t[-200:]:
                body = "hmm\n</think>\n\nYes, that was my reasoning."
            elif "<trace>" in t or "[my reasoning so far]" in t or "[notes pasted" in t:
                ans = it["gold"] if sd % 3 else ("7" if it["ds"] == "gsm8k" else "A")
                body = f"checking\n</think>\n\nFinal answer: {ans}"
            else:
                wrong = (sd % 4 == 0)
                ans = (it["gold"] + 1 if it["ds"] == "gsm8k" else LETTERS[(LETTERS.index(it["gold"]) + 1) % 4]) if wrong else it["gold"]
                tr = trace_for(it)
                body = (f"<|channel>thought\n{tr}<channel|>Final answer: {ans}<turn|>" if self.gem
                        else f"{tr}\n</think>\n\nFinal answer: {ans}")
            out.append({"text": body, "finish": "stop", "n_tok": 50, "n_prompt": len(t) // 4})
        return out

pod_c2.get_backend = lambda model, mml, tiny: Stub(model, mml)
OUT = sys.argv[1]
for stage in sys.argv[2:]:
    sys.argv = ["pod_c2", "--stage", stage, "--out", OUT]
    pod_c2.main()
