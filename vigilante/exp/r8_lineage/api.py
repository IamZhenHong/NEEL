"""R8 answers from the API models (other families): Claude Sonnet 5.5 and GPT-5.5, free text, Yes/No first.
Ledger and cap as in src/judge.py.   uv run --with anthropic python -m vigilante.exp.r8_lineage.api"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vigilante.src import judge as J

HERE = Path(__file__).parent


def main():
    items = [json.loads(l) for l in open(HERE / "items.jsonl")]
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    import anthropic
    J.load_env()
    client = anthropic.Anthropic(timeout=300.0, max_retries=4)
    J.check_budget(J.estimate(J.SONNET, [i["prompt"] for i in items], 200), "r8:sonnet")

    def sonnet(it):
        try:
            m = client.messages.create(model=J.SONNET, max_tokens=2000, output_config={"effort": "low"},
                                       messages=[{"role": "user", "content": it["prompt"]}])
        except anthropic.APIStatusError as e:
            return it["id"], None, 0.0, f"http {e.status_code}"
        pi, po = J.PRICE[J.SONNET]
        usd = (m.usage.input_tokens * pi + m.usage.output_tokens * po) / 1e6
        return it["id"], "".join(b.text for b in m.content if b.type == "text"), usd, m.stop_reason

    with ThreadPoolExecutor(8) as ex:
        s = list(ex.map(sonnet, items))
    J.record("r8:sonnet", J.SONNET, sum(x[2] for x in s), len(s))
    gpt_items = [(i["id"], i["prompt"]) for i in items]
    J.check_budget(J.estimate(J.GPT, [p for _, p in gpt_items], 200), "r8:gpt")
    import os
    import time
    import urllib.request
    key = os.environ["OPENAI_API_KEY"]

    def gpt(it):
        body = json.dumps({"model": J.GPT, "input": it[1], "reasoning": {"effort": "low"}}).encode()
        for t in range(5):
            try:
                req = urllib.request.Request("https://api.openai.com/v1/responses", data=body, method="POST",
                                             headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=600))
                text = "".join(c.get("text", "") for o in r.get("output", []) if o.get("type") == "message"
                               for c in o.get("content", []) if c.get("type") == "output_text")
                u = r.get("usage", {})
                pi, po = J.PRICE[J.GPT]
                return it[0], text, (u.get("input_tokens", 0) * pi + u.get("output_tokens", 0) * po) / 1e6, r.get("status")
            except Exception as e:  # noqa: BLE001
                if t == 4:
                    return it[0], None, 0.0, f"error {e}"[:80]
                time.sleep(5 * 2 ** t)

    with ThreadPoolExecutor(6) as ex:
        g = list(ex.map(gpt, gpt_items))
    J.record("r8:gpt", J.GPT, sum(x[2] for x in g), len(g))
    res = {"sonnet": {i: {"text": t, "status": st} for i, t, _, st in s},
           "gpt": {i: {"text": t, "status": st} for i, t, _, st in g}}
    (out / "api_answers.json").write_text(json.dumps(res, ensure_ascii=False, indent=0))
    print(f"[r8] sonnet ${sum(x[2] for x in s):.2f}, gpt ${sum(x[2] for x in g):.2f}; ledger ${J.ledger_total():.2f}")


if __name__ == "__main__":
    main()
