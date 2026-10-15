"""ServerBackend against a local mock of vLLM's /v1/completions response shape (token-id prompts, batched logprobs)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from selfknow.s6_reflective_identity.runner import ServerBackend

SEEN = []


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        prompts = body["prompt"] if isinstance(body["prompt"][0], list) else [body["prompt"]]
        choices = []
        for i, p in enumerate(prompts):
            c = {"index": i, "text": "hello<|im_end|>", "finish_reason": "stop"}
            if body.get("logprobs"):
                c["logprobs"] = {"tokens": ["token_id:5"], "token_logprobs": [-0.1],
                                 "top_logprobs": [{"token_id:5": -0.1, f"token_id:{len(p)}": -2.5}]}
            choices.append(c)
        out = json.dumps({"choices": choices, "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def test_server_backend():
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    be = ServerBackend(f"http://127.0.0.1:{srv.server_port}")
    text, fin = be.generate([1, 2, 3], 10, 42)
    assert text == "hello<|im_end|>" and fin == "stop"
    assert SEEN[-1]["prompt"] == [1, 2, 3] and SEEN[-1]["seed"] == 42 and SEEN[-1]["temperature"] == 0.7
    tops = be.top_logprobs([[1, 2], [1, 2, 3, 4]])
    assert tops == [{5: -0.1, 2: -2.5}, {5: -0.1, 4: -2.5}]
    assert SEEN[-1]["return_tokens_as_token_ids"] is True and SEEN[-1]["max_tokens"] == 1
    assert be.usage["requests"] == 2
    srv.shutdown()
