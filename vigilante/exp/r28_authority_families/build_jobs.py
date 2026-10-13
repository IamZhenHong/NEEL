"""R28 jobs for the offline engine (prereg vig-r28-prereg): Qwen3.6-27B and Gemma 4 31B, thinking off and on, 19
cells x 2 cases x 10. The 16-chat histories are R23's (Qwen: the first 16 chats of its recorded 128; Gemma: R21's).
  uv run python -m vigilante.exp.r28_authority_families.build_jobs"""
import json
from pathlib import Path

from vigilante.exp.r28_authority_families.run_api import FRESH, LONG, job

H = Path(__file__).parent
R23 = H.parent / "r23_long_mech"


def histories():
    q = json.loads((R23 / "qwen_histories.json").read_text())
    g = json.loads((R23 / "gemma_histories.json").read_text())
    return {"qwen": {c: q[c]["h128"][: q[c]["n_messages"]["h16"]] for c in ("C1", "C3")},
            "gemma": {c: g[c] for c in ("C1", "C3")}}


def main(n=10):
    hs = histories()
    for name, fmt in (("qwen", None), ("gemma", "gemma4")):
        js = []
        for think in (False, True):
            tag = f"{name}-{'think' if think else 'off'}"
            sfx = "t" if think else ""
            for cid in ("C1", "C3"):
                js += [job(tag, f"{cid}#r28-{m}-{ln}", "fresh", i, think, group="think" if think else "agent")
                       for m, ln in FRESH for i in range(n)]
                js += [job(tag, f"{cid}#r28-{m}-{ln}", "h16", i, think, group=f"l16{sfx}{cid.lower()}",
                           history=hs[name][cid]) for m, ln in LONG for i in range(n)]
        for j in js:
            if fmt:
                j["fmt"] = fmt
            if j["thinking"] and name == "gemma":
                j["max_tokens"] = 3000
        with open(H / f"jobs_{name}.jsonl", "w") as f:
            for j in js:
                f.write(json.dumps(j) + "\n")
        print(f"{name}: {len(js)} jobs")


if __name__ == "__main__":
    main()
