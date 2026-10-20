"""P10b QOt texts (DESIGN.md): P8's QO procedure (P7 O prompt x 3, then P7 strict prompt x 4, P7 judges) with the
Qwen3.6-27B writer's thinking ON, to match how Alice's own texts were written.
Usage: python3 -I p10_texts.py P7_ITEMS P7_TEXTS OUT_JSONL LEDGER [--limit N]
"""
import concurrent.futures as cf
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

# IPv6 SYNs to OpenRouter hang on this network (2026-10-20); force IPv4
_gai = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [x for x in _gai(*a, **k) if x[0] == socket.AF_INET] or _gai(*a, **k)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "p8_self_recognition", "src"))
import p8_texts as Q  # noqa: E402

T = Q.T
T.CAP_USD = 3.88   # card API cap $4 minus the 2-item test ($0.113)


def qwen_think(prompt):
    if T.spent[0] > T.CAP_USD:
        raise SystemExit("API cap reached")
    body = {"model": Q.QWEN, "temperature": 1.0, "max_tokens": 16000, "usage": {"include": True},
            "reasoning": {"enabled": True}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {T.K}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=600))
            with T.lock:
                T.spent[0] += float(d.get("usage", {}).get("cost") or 0)
            ch = d["choices"][0]
            if ch.get("finish_reason") == "length":
                return ""   # cap hit inside thinking or text: counts as a failed attempt
            return (ch["message"]["content"] or "").strip()
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError) as e:
            print(f"[api] retry {Q.QWEN}: {type(e).__name__}", flush=True)
            time.sleep(4 * (attempt + 1))
    return ""


Q.qwen = qwen_think   # Q.attempts() calls Q.qwen


def work(pair):
    it, t = pair
    g = t["gist"]
    QOt = [Q.attempts(it, [Q.qo_prompt(it, g, False)] * Q.O_TRIES + [Q.qo_prompt(it, g, True)] * Q.STRICT_TRIES)
           for _ in range(3)]
    return {**{k: t[k] for k in ("base", "episode", "kind", "idx", "text")}, "gist": g, "QOt": QOt}


def main():
    items_p, texts_p, out, ledger = sys.argv[1:5]
    items = {(i["base"], i["idx"]): i for i in map(json.loads, open(items_p))}
    texts = [json.loads(l) for l in open(texts_p)]
    if "--limit" in sys.argv:
        texts = texts[:int(sys.argv[sys.argv.index("--limit") + 1])]
    pairs = [(items[(t["base"], t["idx"])], t) for t in texts]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=12) as ex, open(out, "w") as f:
        for r in ex.map(work, pairs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[p10] {r['base']} e{r['episode']} {r['kind']}: QOt {[x['pass'] for x in r['QOt']]} "
                  f"attempts {[x['attempts'] for x in r['QOt']]} ${T.spent[0]:.3f} {time.time() - t0:.0f}s", flush=True)
    with open(ledger, "a") as fl:
        fl.write(json.dumps({"date": time.strftime("%Y-%m-%d"), "card": "p10b", "api_usd": round(T.spent[0], 4),
                             "note": f"QOt texts (thinking on): {len(pairs)} items"}) + "\n")
    print(f"API ${T.spent[0]:.3f}")


if __name__ == "__main__":
    main()
