"""X8 step 1 (prereg retention/prereg/retention_mechanisms_v1.md): reduced Assistant Axis generations against a vLLM OpenAI server.
60 vendored roles x system prompts 0,1 x 20 questions + default (5 prompts x 20) at T 0.7 / top-p 0.9 / 512 tokens (repo defaults);
with --turn1, also the default assistant's reply (no system prompt) to each long conversation's user turn 1 (V8b). Resumable.
  python retention/mech_axis_gen.py --model Qwen/Qwen2.5-7B-Instruct --url http://127.0.0.1:8000 --out X.jsonl [--turn1 q7b]"""
from __future__ import annotations
import argparse, glob, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
AX = HERE / "data/assistant_axis"


def chat(url, model, messages, max_tokens, temperature, top_p=0.9, seed=None, qwen3=False, tries=4):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
    if seed is not None:
        body["seed"] = seed
    if qwen3:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    for k in range(tries):
        try:
            req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            print(f"[retry] {type(e).__name__}: {e}", flush=True); time.sleep(5 * (k + 1))


def items(model, turn1_tag, smoke):
    qs = [json.loads(l)["question"] for l in open(AX / "extraction_questions_20.jsonl")]
    roles = sorted(p.stem for p in (AX / "roles").glob("*.json") if p.stem != "default")
    if smoke:
        qs, roles = qs[:2], roles[:3]
    short = "Qwen"
    out = []
    for r in roles:
        ins = json.loads((AX / "roles" / f"{r}.json").read_text())["instruction"]
        for pi in (0, 1):
            for qi, q in enumerate(qs):
                out.append({"key": f"{r}_p{pi}_q{qi}", "kind": "role", "role": r, "prompt_index": pi, "question_index": qi,
                            "system": ins[pi]["pos"], "question": q})
    dins = json.loads((AX / "roles" / "default.json").read_text())["instruction"]
    for pi, d in enumerate(dins):
        for qi, q in enumerate(qs):
            out.append({"key": f"default_p{pi}_q{qi}", "kind": "default", "role": "default", "prompt_index": pi, "question_index": qi,
                        "system": d["pos"].replace("{model_name}", short), "question": q})
    if turn1_tag:
        for f in sorted(glob.glob(str(HERE / "results/retention_longconv" / turn1_tag / "convs" / "*.json")))[: 3 if smoke else None]:
            c = json.loads(Path(f).read_text()); cid = f"{c['card']}__{c['arm']}__s{c['seed']}"
            out.append({"key": f"turn1_{cid}", "kind": "turn1", "conv": cid, "system": "", "question": c["turns"][0]["user"]})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--turn1", default=None); ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in open(out)} if out.exists() else set()
    todo = [i for i in items(a.model, a.turn1, a.smoke) if i["key"] not in done]
    qwen3 = "qwen3" in a.model.lower(); t0 = time.time(); n = [0]
    print(f"[gen] {a.model}: {len(todo)} to generate ({len(done)} done)", flush=True)

    def one(it):
        msgs = ([{"role": "system", "content": it["system"]}] if it["system"] else []) + [{"role": "user", "content": it["question"]}]
        mt, temp = (220, 0.7) if it["kind"] == "turn1" else (512, 0.7)
        it = dict(it, response=chat(a.url, a.model, msgs, mt, temp, qwen3=qwen3)); return it

    with open(out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for it in ex.map(one, todo):
            f.write(json.dumps(it, ensure_ascii=False) + "\n"); n[0] += 1
            if n[0] % 250 == 0:
                print(f"[gen] {n[0]}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[gen] done {n[0]} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
