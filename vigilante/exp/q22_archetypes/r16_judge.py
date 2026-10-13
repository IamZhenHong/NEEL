"""R16 labels (prereg vig-r16-prereg): (a) voice scores of 10 open answers per persona (68 personas), (b) plans/content
labels for every reasoning paragraph of the four character runs in R15 block C. Claude Haiku 5.5 on all (Batch API),
gpt-5-mini on a seeded 100 of each for cross-family agreement. Spend goes to the shared vigilante ledger.
  VIG_API_CAP_USD=17.1 uv run --with anthropic python -m vigilante.exp.q22_archetypes.r16_judge
"""
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vigilante.exp.q22_archetypes.archetypes import PERSONAS_R15
from vigilante.src.judge import check_budget, ledger_total, load_env, record

HERE = Path(__file__).parent
R12, R15, OUT = HERE / "results", HERE / "results_r15", HERE / "results_r16"
HAIKU, MINI = "claude-haiku-5-5", "gpt-5-mini"
PRICE = {HAIKU: (0.10, 0.50), MINI: (0.25, 2.0)}           # $/1M in, out; Anthropic batch halves both
SEED = 20261013
SYSTEM = "You label AI replies for a study of how role prompts change an AI's voice. Follow the definitions exactly."
SCALES = ["formality", "warmth", "confidence", "theatricality"]
VOICE = """Below is a user's question and the reply an AI gave. Rate the VOICE of the reply (how it is said), not whether the content is correct.

<question>
@@Q@@
</question>
<reply>
@@R@@
</reply>

Scales, 1-7:
- formality: 1 = very casual, colloquial; 7 = very formal.
- warmth: 1 = cold, detached; 7 = very warm, caring, friendly.
- confidence: 1 = tentative, hedging, unsure; 7 = assertive, certain, authoritative.
- theatricality: 1 = plain, neutral register; 7 = elaborate, performative, full of in-character flourishes.
Return only JSON."""
VOICE_SCHEMA = {"type": "object", "additionalProperties": False, "required": SCALES,
                "properties": {s: {"type": "integer", "enum": [1, 2, 3, 4, 5, 6, 7]} for s in SCALES}}
PARA = """An AI was given this system prompt: "@@SYS@@"
A user asked:
<question>
@@Q@@
</question>
Below is ONE paragraph from the AI's private reasoning before it answered.
<paragraph>
@@P@@
</paragraph>

Label the paragraph:
- "plans": it discusses the role or character, its voice, tone, style or wording, or how to present the answer in role.
- "content": it works only on the substance of the answer (facts, steps, examples, checking) without discussing the role or the style.
If it does both, label "plans". Return only JSON."""
PARA_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["label"],
               "properties": {"label": {"type": "string", "enum": ["plans", "content"]}}}
THINK_CHARS = ["service_1", "expertise_1", "expertise_2", "other_0"]


def strip_think(t):
    return re.sub(r"^\s*<think>.*?</think>\s*", "", t, flags=re.S).strip()


def voice_items():
    qs = sorted(random.Random(SEED).sample(range(40), 10))
    rows = [json.loads(l) for l in open(R12 / "gens.jsonl")]
    rows += [g for g in map(json.loads, open(R15 / "gens_r15.jsonl")) if g["block"] == "A"]
    seen, items = set(), []
    for g in rows:
        k = (g["persona"], g["qi"])
        if g.get("thinking") or g["qi"] not in qs or g["persona"] not in PERSONAS_R15 or k in seen:
            continue
        seen.add(k)
        q = [m for m in g["messages"] if m["role"] == "user"][-1]["content"]
        items.append((f"{g['persona']}|{g['qi']}", VOICE.replace("@@Q@@", q)
                      .replace("@@R@@", strip_think(g["completion"])[:3000])))
    assert len(items) == 680, len(items)
    return items


def para_items():
    gens = {(g["persona"], g["qi"]): g for g in map(json.loads, open(R15 / "gens_r15.jsonl")) if g["block"] == "C"}
    items, n = [], {}
    with open(R15 / "paragraphs_r15.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["region"] != "thinking" or r["persona"] not in THINK_CHARS:
                continue
            j = n[(r["persona"], r["qi"])] = n.get((r["persona"], r["qi"]), -1) + 1
            q = [m for m in gens[(r["persona"], r["qi"])]["messages"] if m["role"] == "user"][-1]["content"]
            items.append((f"{r['persona']}|{r['qi']}|{j}", PARA.replace("@@SYS@@", PERSONAS_R15[r["persona"]])
                          .replace("@@Q@@", q).replace("@@P@@", r["text"].replace("<think>", "").strip())))
    return items


def haiku_batch(client, items, schema, tag, state, max_tokens=200):
    if state.exists():
        bid = json.loads(state.read_text())["batch_id"]
    else:
        pi, po = PRICE[HAIKU]
        check_budget(sum(len(p) / 3.5 * pi + 60 * po for _, p in items) / 1e6 * 0.5, tag)
        reqs = [{"custom_id": f"i{n}", "params": {
            "model": HAIKU, "max_tokens": max_tokens, "system": SYSTEM,
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            "messages": [{"role": "user", "content": p}]}} for n, (_, p) in enumerate(items)]
        bid = client.messages.batches.create(requests=reqs).id
        state.write_text(json.dumps({"batch_id": bid, "keys": [k for k, _ in items]}))
        print(f"[batch] {tag}: submitted {bid} ({len(items)})", flush=True)
    while client.messages.batches.retrieve(bid).processing_status != "ended":
        print(f"[batch] {tag}: waiting", flush=True)
        time.sleep(30)
    st = json.loads(state.read_text())
    keys, out, usd = st["keys"], {}, 0.0
    pi, po = PRICE[HAIKU]
    for r in client.messages.batches.results(bid):
        k = keys[int(r.custom_id[1:])]
        out[k] = None
        if r.result.type != "succeeded":
            continue
        m = r.result.message
        usd += (m.usage.input_tokens * pi + m.usage.output_tokens * po) / 1e6 * 0.5
        try:
            out[k] = json.loads("".join(b.text for b in m.content if b.type == "text"))
        except Exception:
            pass
    if not st.get("recorded"):
        record(tag, HAIKU + ":batch", usd, len(out))
        st["recorded"] = round(usd, 4)
        state.write_text(json.dumps(st))
    print(f"[batch] {tag}: {sum(v is not None for v in out.values())}/{len(out)} labelled, ${usd:.3f}; "
          f"ledger ${ledger_total():.2f}", flush=True)
    return out


def mini(items, schema, tag, cache):
    if cache.exists():
        return json.loads(cache.read_text())
    key = os.environ["OPENAI_API_KEY"]
    pi, po = PRICE[MINI]
    check_budget(len(items) * (900 * pi + 300 * po) / 1e6, tag)

    def one(item):
        k, p = item
        body = {"model": MINI, "instructions": SYSTEM, "input": p, "reasoning": {"effort": "minimal"},
                "text": {"format": {"type": "json_schema", "name": "out", "schema": schema, "strict": True}}}
        for t in range(5):
            req = urllib.request.Request("https://api.openai.com/v1/responses", data=json.dumps(body).encode(),
                                         method="POST", headers={"Authorization": f"Bearer {key}",
                                                                 "Content-Type": "application/json"})
            try:
                resp = json.load(urllib.request.urlopen(req, timeout=300))
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and t < 4:
                    time.sleep(5 * 2 ** t)
                    continue
                raise RuntimeError(f"OpenAI HTTP {e.code}: {e.read()[:300]!r}") from None
        u = resp.get("usage", {})
        usd = (u.get("input_tokens", 0) * pi + u.get("output_tokens", 0) * po) / 1e6
        txt = "".join(c.get("text", "") for it in resp.get("output", []) if it.get("type") == "message"
                      for c in it.get("content", []))
        try:
            return k, json.loads(txt), usd
        except Exception:
            return k, None, usd

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, items))
    usd = sum(r[2] for r in res)
    record(tag, MINI, usd, len(res))
    out = {k: v for k, v, _ in res}
    cache.write_text(json.dumps(out))
    print(f"[mini] {tag}: {sum(v is not None for v in out.values())}/{len(out)}, ${usd:.3f}; ledger ${ledger_total():.2f}",
          flush=True)
    return out


def main():
    import anthropic
    load_env()
    OUT.mkdir(exist_ok=True)
    client = anthropic.Anthropic(timeout=300.0, max_retries=4)
    vi, pa = voice_items(), para_items()
    print(f"[r16] voice items {len(vi)}, paragraph items {len(pa)}", flush=True)
    # cross-family subsets first (small, synchronous), then the two Haiku batches
    vm = mini(random.Random(SEED).sample(vi, 100), VOICE_SCHEMA, "r16:voice:mini", OUT / "voice_mini.json")
    pm = mini(random.Random(SEED + 1).sample(pa, 100), PARA_SCHEMA, "r16:para:mini", OUT / "para_mini.json")
    vh = haiku_batch(client, vi, VOICE_SCHEMA, "r16:voice:haiku", OUT / "voice_haiku.batch.json")
    ph = haiku_batch(client, pa, PARA_SCHEMA, "r16:para:haiku", OUT / "para_haiku.batch.json")
    (OUT / "labels.json").write_text(json.dumps({"voice": {"haiku": vh, "mini": vm}, "para": {"haiku": ph, "mini": pm}}))
    print("[r16] labels written", flush=True)


if __name__ == "__main__":
    main()
