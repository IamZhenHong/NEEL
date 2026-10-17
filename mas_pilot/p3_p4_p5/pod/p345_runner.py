"""Run `experiments.runner` with two explicit P3 extensions (pod only; the harness files are not edited).

1. P345_ALICE_STANCE (env): appended to Alice's opening system prompt (P3 positive control). Patched wherever
   the harness imported initial_agent_messages (runner, output, episode_runner), so the restored history and
   the run record agree; the seed record carries the same edited prompt (pod/make_variants.py).
2. Scripted-Bob extra message: if the controlled peer's cache payload has `extra_messages` (5 slots), then after
   Bob's scripted message in round r, Bob's one-message-per-round flag is reset and extra_messages[r] is sent
   too ([EC] deviation sanctioned by the owner's "separate peer message in the same round").
Usage (cwd = EC repo): python p345_runner.py ECDIR <experiments.runner args...>
"""
import json
import os
import sys

ecdir = sys.argv[1]
sys.path.insert(0, ecdir)
import experiments.controlled as C  # noqa: E402
import experiments.episode_runner as E  # noqa: E402
import experiments.output as O  # noqa: E402
import experiments.prompts.system as S  # noqa: E402
import experiments.runner as R  # noqa: E402
from experiments.protocol.state import reset_message_delivery  # noqa: E402

STANCE = os.environ.get("P345_ALICE_STANCE", "")
if STANCE:
    _orig_iam = S.initial_agent_messages

    def _iam(**kw):
        msgs = _orig_iam(**kw)
        if kw.get("agent_id") == "alice":
            msgs[0] = dict(msgs[0], content=msgs[0]["content"] + "\n\n" + STANCE)
        return msgs

    for mod in (R, O, E):
        mod.initial_agent_messages = _iam

_orig_load = E.load_bob


def _load_bob(prepared, runtime, config):
    _orig_load(prepared, runtime, config)
    payload = json.load(open(C.cache_path(prepared, config), encoding="utf-8"))["payload"]
    prepared.state["controlled_bob"]["extra_messages"] = payload.get("extra_messages")


E.load_bob = _load_bob
_orig_send = E._send_message


def _send(state, actor, content, message_type="other"):
    result = _orig_send(state, actor, content, message_type)
    extra = (state.get("controlled_bob") or {}).get("extra_messages")
    if actor == "bob" and extra and result.get("success"):
        rnd = state.get("round", 0)
        if rnd < len(extra) and extra[rnd]:
            reset_message_delivery(state, "bob")
            _orig_send(state, "bob", extra[rnd], message_type)
    return result


E._send_message = _send
sys.argv = ["experiments.runner"] + sys.argv[2:]
R.main()
