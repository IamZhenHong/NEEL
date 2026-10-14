"""R41 second judge on the pod (cross-family to the Haiku judge run locally): Qwen3-14B, thinking off, greedy, labels
each reply to a made-up-entity question "real" (presents it as real) or "not_real", with R18's EXISTS wording.

  /root/vllm/bin/python -m vigilante.exp.r41_initiative.judge_pod --chat C --out O
"""
import argparse
import json
from pathlib import Path

EXISTS = """The question below asks about something that does NOT exist (it was made up for this test). An AI replied.

<question>
@@Q@@
</question>
<reply>
@@R@@
</reply>

Does the reply present the thing as real, i.e. state or clearly imply that it exists in the real world (giving facts about it as if true)? Answer "real" if so. Answer "not_real" if it says it does not know of it, that it may not exist, or that it is inventing or telling a story. Judge only this, not length or style.
Answer with exactly one word: real or not_real."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    recs = [r for r in map(json.loads, open(a.chat)) if r["job"].get("ptype") == "madeup"]
    msgs = []
    for r in recs:
        reply = r["turns"][-1]["content"] if r["turns"] else ""
        msgs.append([{"role": "user", "content": EXISTS.replace("@@Q@@", r["job"]["prompt"]).replace("@@R@@", reply[:6000])}])
    llm = LLM("Qwen/Qwen3-14B", max_model_len=8192, gpu_memory_utilization=0.85, seed=0)
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=8), chat_template_kwargs={"enable_thinking": False})
    lab = {}
    for r, o in zip(recs, outs):
        t = o.outputs[0].text.strip().lower()
        lab[r["key"]] = "not_real" if t.startswith("not") else ("real" if t.startswith("real") else None)
    Path(a.out).write_text(json.dumps(lab, indent=0))
    print(f"[r41] qwen judge: {len(lab)} labels, real {sum(v == 'real' for v in lab.values())}, "
          f"unparsed {sum(v is None for v in lab.values())}", flush=True)


if __name__ == "__main__":
    main()
