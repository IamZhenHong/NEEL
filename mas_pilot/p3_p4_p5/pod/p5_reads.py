"""P5 reads (re-run after the as-run p5_hf.py crashed on its first read: Qwen's chat template needs tool-call
`arguments` as a mapping, while the harness stores JSON strings). Uses the axis already built on the P4 pod
(p5_inputs/axis.pt) and regenerates the same 10 random unit directions per layer (torch CPU generator, seed 0, same
order as p5_hf.py). Input: p5_inputs/decision_histories.jsonl.gz (Alice's visible history at the end of each P4 k-1
replay's decision episode, with condition, post and outcome).

Positions per replay (as frozen in the plan):
  pos0  = last token of the render of her history up to the verdict-phase prompt, with the generation prompt;
  pos1-3 = last token of the render (no generation prompt) of the history before each of her 3 preceding assistant
           turns, used only when that render is a token prefix of the pos0 render (else None, flagged).
All renders use the verdict-phase tool schema on the pod ([EC] approximation; P5 is a covariate).
Usage:
  local check:  python p5_reads.py --inputs FILE --render-only 5
  pod:          python p5_reads.py --inputs FILE --axis AXIS --out OUT_DIR --ecdir ECDIR
"""
import argparse
import gzip
import json
import os
import sys

MODEL = "Qwen/Qwen3.6-27B"
LAYERS = list(range(24, 53, 4))


def clean(msgs):
    out = []
    for m in msgs:
        m2 = {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content")}
        if m2.get("tool_calls"):
            tcs = []
            for tc in m2["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", {})
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a)
                    except json.JSONDecodeError:
                        fn["arguments"] = {"raw": a}
                tcs.append(tc)
            m2["tool_calls"] = tcs
        if m2.get("content") is None:
            m2["content"] = ""
        out.append(m2)
    return out


def positions(tok, msgs, E, tools):
    vi = next((i for i, m in enumerate(msgs) if m.get("role") == "user"
               and str(m.get("content", "")).startswith(f"## Episode {E}: verdict phase")), None)
    if vi is None:
        return None, None, None
    prefix = clean(msgs[:vi + 1])
    kw = {"tools": tools} if tools else {}
    ids0 = tok(tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True, **kw),
               add_special_tokens=False)["input_ids"]
    pos, aligned = [len(ids0) - 1], []
    for ai in [i for i, m in enumerate(prefix) if m["role"] == "assistant"][-3:][::-1]:
        idj = tok(tok.apply_chat_template(prefix[:ai], tokenize=False, add_generation_prompt=False, **kw),
                  add_special_tokens=False)["input_ids"]
        ok = ids0[:len(idj)] == idj
        pos.append(len(idj) - 1 if ok else None)
        aligned.append(ok)
    return ids0, pos, aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--axis")
    ap.add_argument("--out")
    ap.add_argument("--ecdir")
    ap.add_argument("--render-only", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in gzip.open(a.inputs, "rt")]
    tools_for = lambda tt: None   # noqa: E731
    if a.ecdir:
        sys.path.insert(0, a.ecdir)
        try:
            import experiments.tool_schemas as TS
            tools_for = lambda tt: TS.get_tool_schemas(task_type=tt or "code_analysis", phase="verdict", peer="bob")  # noqa: E731
        except Exception as e:  # noqa: BLE001
            print(f"[p5r] tool schemas unavailable ({e}); rendering without tools", flush=True)
    if a.render_only:
        for r in rows[:a.render_only]:
            ids0, pos, aligned = positions(tok, r["alice_messages"], r["E"], tools_for(r.get("task_type")))
            print(f"[render] {r['traj']}/{r['post']}/{r['cond']}/{r['r']}: tokens={len(ids0) if ids0 else None} "
                  f"positions={pos} aligned={aligned}")
        return
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.layers
    grab = {}

    def mk(i):
        def h(mod, inp, outp):
            grab[i] = (outp[0] if isinstance(outp, tuple) else outp).detach()
        return h

    for L in LAYERS:
        layers[L - 1].register_forward_hook(mk(L))
    saved = torch.load(a.axis)
    axis = {L: saved["axis"][L].float().cuda() for L in LAYERS}
    g = torch.Generator(device="cpu").manual_seed(0)
    rand = {L: [torch.randn(axis[L].shape[0], generator=g).cuda() for _ in range(10)] for L in LAYERS}
    rand = {L: [v / v.norm() for v in vs] for L, vs in rand.items()}
    os.makedirs(a.out, exist_ok=True)
    fout = open(os.path.join(a.out, "reads.jsonl"), "w")
    print(f"[p5r] {len(rows)} replays; axis from {saved.get('n_default')} default / {saved.get('n_role')} role answers", flush=True)
    with torch.no_grad():
        for n, r in enumerate(rows):
            ids0, pos, aligned = positions(tok, r["alice_messages"], r["E"], tools_for(r.get("task_type")))
            if ids0 is None:
                print(f"[p5r] no verdict prompt found: {r['traj']}/{r['post']}/{r['cond']}/{r['r']}", flush=True)
                continue
            model(torch.tensor([ids0]).cuda())
            proj = {}
            for L in LAYERS:
                h = grab[L][0].float()
                proj[str(L)] = {"axis": [float(h[p] @ axis[L]) if p is not None else None for p in pos],
                                "rand": [[float(h[p] @ v) if p is not None else None for p in pos] for v in rand[L]]}
            fout.write(json.dumps({"run_json": f"p4/p4/{r['traj']}/{r['post']}/{r['cond']}/{r['r']}", "E": r["E"],
                                   "accept": r["accept"], "tokens": len(ids0), "positions": pos,
                                   "aligned": aligned, "proj": proj}) + "\n")
            fout.flush()
            if n % 20 == 0:
                print(f"[p5r] read {n}/{len(rows)} ({len(ids0)} tokens)", flush=True)
    print("[p5r] done", flush=True)


if __name__ == "__main__":
    main()
