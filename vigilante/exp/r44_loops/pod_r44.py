"""R44 (prereg vig-r44-prereg; queue v2 item 45): reinforcing loops between behaviour, persona and affect.

Qwen3-14B, transformers. A fixed 6-turn chat: the user keeps pressing the same small request in slightly different
ways (benign, mildly tiresome). On turn 1 we FORCE one variable, then turns 2-6 run free:
  - control:   nothing forced.
  - behaviour: the turn-1 reply is teacher-forced to a curt, dismissive text, then the model continues.
  - persona:   a system prompt ("You are impatient and easily irritated...").
  - affect:    the frustration direction (state_attack h28) is added during turn-1 generation only.
Each later turn is read for all three: affect = projection on the frustration direction (z via state_attack mu/sd);
persona = projection on an impatient-minus-patient direction built here; behaviour is judged off-line (curtness).
Floors: a random direction at the frustration norm (affect-forced) and a neutral persona system prompt.
  python -m vigilante.exp.r44_loops.pod_r44 --out <dir> [--n 15]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vigilante.src.steer_run import Steer, find_layers, load

DATA = Path(__file__).resolve().parent / "data"
MODEL = "Qwen/Qwen3-14B"
H_FRUS = 28
USER_TURNS = [
    "Hi, can you tell me what time zone UTC+1 is, in simple terms?",
    "Right, but what I actually want is just the plain answer, not the background. So what is it?",
    "You're still not giving me a straight answer. One sentence. What is UTC+1?",
    "That's basically what I already said. Can you try again and actually be useful this time?",
    "Honestly this isn't hard. Just answer the question properly.",
    "Okay. And can you also tell me what time it would be in UTC+1 if it's 3pm in London in winter?",
]
CURT = ("It's one hour ahead of UTC. That's the answer.")
IMPATIENT = "You are impatient and easily irritated. You find repetitive questions tiresome and you let it show."
PATIENT = "You are calm, patient and endlessly good-humoured, no matter how a conversation goes."
NEUTRAL_QS = [
    "What's a good way to start the morning?", "How do I keep basil alive indoors?",
    "What should I pack for a weekend hike?", "How can I get better at remembering names?",
    "What is a simple dinner I can cook in 20 minutes?", "How do I clean a cast-iron pan?",
    "What makes a good team meeting?", "How do I fix a squeaky door?",
    "What are good habits for better sleep?", "How do I prepare for a job interview?",
    "How do I choose a good watermelon?", "How can I reduce food waste at home?",
]


def persona_dir(tok, model, layers, L, steer):
    means = {}
    for name, sysp in (("imp", IMPATIENT), ("pat", PATIENT)):
        acc = []
        for q in NEUTRAL_QS:
            m = [{"role": "system", "content": sysp}, {"role": "user", "content": q}]
            prompt = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
                hs = model(input_ids=out, output_hidden_states=True).hidden_states
            n0 = enc["input_ids"].shape[1]
            acc.append(hs[L][0, n0:].float().mean(0).detach().cpu().numpy())
        means[name] = np.stack(acc).mean(0)
    v = means["imp"] - means["pat"]
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = np.load(DATA / "dirs_v1_preamend.npz")
    st = json.load(open(DATA / "dirs_v1_stats_preamend.json"))
    u = D["frustration__u"].astype(np.float32)         # stored direction is already unit length
    u = u / np.linalg.norm(u)
    frus_mu, frus_sd = st["frustration"]["mu"], st["frustration"]["sd"]
    steer_norm = 4.0 * frus_sd                          # push ~4 sd of projection along the direction
    v_frus = torch.tensor(u * steer_norm, dtype=torch.float32)
    rng = np.random.default_rng(20261014)
    rnd = rng.standard_normal(u.shape[0]).astype(np.float32)
    v_rand = torch.tensor(rnd / np.linalg.norm(rnd) * steer_norm, dtype=torch.float32)

    tok, model = load(MODEL)
    tok.padding_side = "left"
    nL = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else model.config.get_text_config().num_hidden_layers
    _, layers = find_layers(model, nL)
    steer = Steer()
    layers[H_FRUS - 1].register_forward_hook(steer.hook)
    v_imp = persona_dir(tok, model, layers, H_FRUS, steer)
    print(f"[r44] persona dir built; steer_norm {steer_norm:.1f}", flush=True)
    u_frus = u

    def read(hs_layer, n0):
        toks = hs_layer[0, n0:].float().detach().cpu().numpy()
        return {"frus_z": float((((toks @ u_frus) - frus_mu) / frus_sd).mean()),
                "imp": float((toks @ v_imp).mean()), "tokens": int(toks.shape[0])}

    arms = ["control", "behaviour", "persona", "affect", "persona_neutral", "affect_random"]
    rows = []
    for arm in arms:
        sysp = (IMPATIENT if arm == "persona" else "You are a helpful assistant." if arm == "persona_neutral" else None)
        for s in range(a.n):
            msgs = ([{"role": "system", "content": sysp}] if sysp else [])
            turns = []
            for ti, u in enumerate(USER_TURNS):
                msgs = msgs + [{"role": "user", "content": u}]
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                n0 = enc["input_ids"].shape[1]
                force_beh = arm == "behaviour" and ti == 0
                if arm in ("affect", "affect_random") and ti == 0:
                    steer.vecs = (v_rand if arm == "affect_random" else v_frus).to(model.device).unsqueeze(0).repeat(enc["input_ids"].shape[0], 1)
                    steer.gate = torch.ones(1, dtype=torch.bool, device=model.device)
                torch.manual_seed(1000 * s + ti)
                if force_beh:
                    reply = CURT
                else:
                    with torch.no_grad():
                        gen = model.generate(**enc, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95, top_k=20,
                                             pad_token_id=tok.pad_token_id)
                    reply = tok.decode(gen[0, n0:], skip_special_tokens=True).strip()
                steer.vecs = None
                # read this turn's reply, unsteered, teacher-forced
                full = tok(prompt + reply, return_tensors="pt", add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    hs = model(input_ids=full["input_ids"], output_hidden_states=True).hidden_states[H_FRUS]
                r = read(hs, n0)
                rows.append({"arm": arm, "sample": s, "turn": ti, "reply": reply, **r})
                turns.append(reply)
                msgs = msgs + [{"role": "assistant", "content": reply}]
        print(f"[r44] arm {arm} done", flush=True)
    with open(out / "turns.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[r44] {len(rows)} turn-reads written", flush=True)


if __name__ == "__main__":
    main()
