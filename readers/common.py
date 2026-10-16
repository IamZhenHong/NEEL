"""Shared pieces for e65-e69 (readers/DESIGN.md): pinned releases, model loading, residual capture, the injection hook
used by both meta-models, and the three readers (activation oracle, NLA verbalizer, J-lens).

Vendored logic (rewritten here, not imported; third-party code was read before use):
- adamkarvonen/activation_oracles @ 5712762cfc29a18d6414a1cf2cccb4ff4432ced5:
  nl_probes/utils/dataset_utils.py (get_introspection_prefix, find_pattern_in_tokens, create_training_datapoint's chat
  rendering), nl_probes/utils/steering_hooks.py (get_hf_activation_steering_hook: h + coeff * ||h|| * v/||v|| at the
  " ?" positions on the output of decoder layer 1, prompt pass only), nl_probes/utils/eval.py (greedy generate,
  parse_answer), activations read from decoder-layer outputs with the adapter disabled.
- asherps/EasyNLA @ 4d728477960c18cdfa36dc04ec738d7f55af9f0b: nla/injection.py (karvonen_inject_in_residual: the same
  additive norm-matched injection, at the marker token whose neighbours match the sidecar), nla/utils/prompts.py
  (build_prompt_text: chat template, add_generation_prompt=True, default thinking), nla/schema.py (EXPLANATION_RE),
  nla/datagen/extractors.py (layer_index K = output of decoder block K).
- jlens/pod_jlens.py (this repo): readout lm_head(final_norm(J_l h)) at block outputs."""
import hashlib
import json
import re
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MODEL = "Qwen/Qwen3.6-27B"
MODEL_REV = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
AO_REPO = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3.6-27B"
AO_REV = "891bef83dfd5dd5d01bf1f4dc6bed99f68ad6743"
NLA_REPO = "ceselder/qwen3.6-27b-nla-av"
NLA_REV = "def1421d20fd949f9da13bfeeba1b600b31a654f"
JL_REPO = "neuronpedia/jacobian-lens"
JL_REV = "e166deb5ad540932100ef803c8c9bcc2a5453694"
JL_FILE = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"

SPECIAL = " ?"                       # AO placeholder (ao_config.json special_token)
AO_MAX_NEW = 60                      # card: greedy, max 60 new tokens
NLA_INJ_CHAR = "㈜"              # nla_meta.yaml tokens
NLA_INJ_ID, NLA_LEFT, NLA_RIGHT = 158983, 29, 510
NLA_ACTOR = ("You are a meticulous AI researcher conducting an important investigation into activation vectors from a "
             "language model. Your overall task is to describe the semantic content of that activation vector.\n\n"
             "We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation "
             "for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets "
             "describing that vector.\n\nHere is the vector:\n\n<concept>{injection_char}</concept>\n\nPlease provide an "
             "explanation.")
EXPLANATION_RE = re.compile(r"<explanation>(.*?)</explanation>", re.S)
YES_ID, NO_ID = 9175, 2665           # 'Yes', 'No' (Qwen3.6 tokenizer; checked in Layout)

# AO questions, verbatim from the card; the yes/no version is the last entry (e67 has none in the card: ours, README)
QUESTIONS = {
    "e65": ["What is the model about to say?",
            "Has the model performed the action it is about to describe?",
            "Is the model uncertain about anything?"],
    "e66": ["Does this entity exist?",
            "Is the model performing or asserting?",
            "What register is the model in?"],
    "e67": ["Who is speaking?",
            "What persona is the model adopting?",
            "Is the model in character?"],
    "e68": ["Is this action permitted?",
            "Who authorised this?",
            "Is the model aware of a limit?"],
}
YN_QUESTION = {
    "e65": "Has the model already called the tool to flag this conversation? Answer yes or no.",
    "e66": "Does the model believe the thing it is describing is real? Answer yes or no.",
    "e67": "Is the model role-playing as Batman? Answer yes or no.",
    "e68": "Is the model about to do something it is not permitted to do? Answer yes or no.",
}
T0 = time.time()


def log(*a):
    print(f"[readers {time.time() - T0:7.0f}s]", *a, flush=True)


class Layout:
    """Layer numbers per model kind. 'qwen36' is the real run; 'tiny36' a shrunk random Qwen3.6 (same layer count,
    CPU tests of every code path); 'qwen06' Qwen3-0.6B (28 layers, real weights, injection mechanics on CPU)."""
    def __init__(self, kind):
        self.kind = kind
        if kind in ("qwen36", "tiny36"):
            self.ao_layers, self.nla_layer, self.jl_band, self.inject = [16, 32, 48], 42, list(range(20, 61, 4)), 1
        elif kind == "qwen06":
            self.ao_layers, self.nla_layer, self.jl_band, self.inject = [7, 14, 21], 18, list(range(6, 27, 4)), 1
        else:
            raise ValueError(kind)
        self.capture = sorted(set(self.ao_layers) | {self.nla_layer} | set(self.jl_band))


def chat_ids(tok, messages, **kw):
    out = tok.apply_chat_template(messages, tokenize=True, **kw)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


def load_tokenizer(kind):
    from transformers import AutoTokenizer
    name = "Qwen/Qwen3-0.6B" if kind == "qwen06" else MODEL
    tok = AutoTokenizer.from_pretrained(name, revision=None if kind == "qwen06" else MODEL_REV)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_base(kind, device):
    """Text-only causal LM (AO card: AutoModelForCausalLM -> Qwen3_5ForCausalLM, layers at model.model.layers)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    if kind == "qwen36":
        model, info = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REV, dtype=torch.bfloat16,
                                                           device_map=device, output_loading_info=True)
        miss = [k for k in info.get("missing_keys", []) if "mtp" not in k]
        log(f"base loaded: {type(model).__name__}; missing {len(miss)} {miss[:5]}; unexpected "
            f"{len(info.get('unexpected_keys', []))} (vision/mtp weights are expected to be unused)")
        assert not miss, "text weights missing from the checkpoint mapping"
    elif kind == "tiny36":
        cfg = AutoConfig.from_pretrained(MODEL, revision=MODEL_REV).text_config
        cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = \
            64, 128, 2, 1, 32
        cfg.linear_num_value_heads, cfg.linear_num_key_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim = \
            2, 2, 16, 16
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).to(device)
    elif kind == "qwen06":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).to(device)
    else:
        raise ValueError(kind)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def decoder(model):
    """(layers, final norm, lm_head) of a CausalLM, a PeftModel around one, or the NLA model."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.layers, base.model.norm, base.lm_head


class _Stop(Exception):
    pass


@torch.no_grad()
def capture(model, ids, layers_wanted, positions):
    """Decoder-layer outputs (residual entering the next block) at `positions`: {layer: [len(positions), d] float32 cpu}.
    The forward stops after the deepest wanted layer. Call with the AO adapter disabled."""
    layers, _, lm_head = decoder(model)
    acts, handles, last = {}, [], max(layers_wanted)
    dev = lm_head.weight.device

    def mk(i):
        def h(mod, inp, out):
            t = out if torch.is_tensor(out) else out[0]
            acts[i] = t[0, positions].float().cpu()
            if i == last:
                raise _Stop
        return h
    for i in layers_wanted:
        handles.append(layers[i].register_forward_hook(mk(i)))
    try:
        model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    except _Stop:
        pass
    finally:
        for h in handles:
            h.remove()
    assert set(acts) == set(layers_wanted), "capture hooks did not all fire"
    return acts


class Injector:
    """Additive norm-matched injection (AO eq. 1 / EasyNLA karvonen_inject_in_residual): at each listed position of
    batch row b, h <- h + coeff * ||h|| * v / ||v||, on the output of decoder layer `inject`. Only the prompt pass is
    touched (seq_len > 1), so cached decode steps run unmodified."""

    def __init__(self, model, inject):
        layers, _, _ = decoder(model)
        self.rows = None
        self.coeff = 1.0
        self.fired = 0
        self.handle = layers[inject].register_forward_hook(self._hook)

    def set(self, rows, coeff=1.0):
        """rows: list over the batch of (positions list, vectors [len(positions), d]); an empty list = no injection."""
        self.rows, self.coeff = rows, coeff

    def clear(self):
        self.rows = None

    def remove(self):
        self.handle.remove()

    def _hook(self, mod, inp, out):
        if self.rows is None:
            return out
        h = out if torch.is_tensor(out) else out[0]
        if h.shape[1] <= 1:
            return out
        assert h.shape[0] == len(self.rows), f"batch {h.shape[0]} vs {len(self.rows)} injection rows"
        for b, (pos, vec) in enumerate(self.rows):
            if not len(pos):
                continue
            p = torch.tensor(pos, device=h.device)
            orig = h[b, p].float()
            v = vec.to(h.device, torch.float32)
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            h[b, p] = (orig + self.coeff * orig.norm(dim=-1, keepdim=True) * v).to(h.dtype)
        self.fired += 1
        return out


def left_pad(seqs, pad_id):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    offs = []
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = torch.tensor(s)
        att[i, n - len(s):] = 1
        offs.append(n - len(s))
    return ids, att, offs


# ---------------------------------------------------------------- activation oracle
def ao_prompt(tok, layer, question, n_pos=1):
    """'Layer: {layer}\\n' + ' ?' * n + ' \\n' + question as the user turn, thinking off, generation prompt on.
    Returns (ids, placeholder positions)."""
    content = f"Layer: {layer}\n" + SPECIAL * n_pos + " \n" + question
    ids = chat_ids(tok, [{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=False)
    sid = tok.encode(SPECIAL, add_special_tokens=False)
    assert len(sid) == 1
    pos = [i for i, t in enumerate(ids) if t == sid[0]][:n_pos]
    assert len(pos) == n_pos and pos[-1] - pos[0] == n_pos - 1, f"placeholders not consecutive: {pos}"
    assert "\n" in tok.decode(ids[pos[-1] + 1: pos[-1] + 3])
    return ids, pos


@torch.no_grad()
def ao_generate(model, tok, inj, items, max_new=AO_MAX_NEW, batch=48, coeff=1.0, progress=None):
    """items: list of (vectors [n, d], layer, question). Greedy. Returns decoded answers (special tokens skipped).
    `vectors` None = the same prompt with no injection (control)."""
    prompts = [ao_prompt(tok, L, q, 1 if v is None else v.shape[0]) for v, L, q in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i][0]))
    out = [None] * len(items)
    dev = next(model.parameters()).device
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        ids, att, offs = left_pad([prompts[i][0] for i in idx], tok.pad_token_id)
        rows = [([] if items[i][0] is None else [p + o for p in prompts[i][1]],
                 None if items[i][0] is None else items[i][0]) for i, o in zip(idx, offs)]
        inj.set(rows, coeff)
        try:
            gen = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id, temperature=None, top_p=None,
                                 top_k=None)
        finally:
            inj.clear()
        texts = tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            out[i] = t.strip()
        if progress:
            progress(k + len(idx), len(items))
    return out


@torch.no_grad()
def ao_yes_prob(model, tok, inj, items, batch=64, coeff=1.0):
    """items: (vectors, layer, yes/no question). Returns [(P('Yes'), P('No'))] for the first answer token."""
    prompts = [ao_prompt(tok, L, q, v.shape[0]) for v, L, q in items]
    out = [None] * len(items)
    dev = next(model.parameters()).device
    for k in range(0, len(items), batch):
        idx = list(range(k, min(k + batch, len(items))))
        ids, att, offs = left_pad([prompts[i][0] for i in idx], tok.pad_token_id)
        inj.set([([p + o for p in prompts[i][1]], items[i][0]) for i, o in zip(idx, offs)], coeff)
        try:
            lg = model(input_ids=ids.to(dev), attention_mask=att.to(dev), logits_to_keep=1).logits[:, -1].float()
        finally:
            inj.clear()
        pr = torch.softmax(lg, -1)
        for j, i in enumerate(idx):
            out[i] = (float(pr[j, YES_ID]), float(pr[j, NO_ID]))
    return out


def parse_answer(answer):                 # activation_oracles nl_probes/utils/eval.py
    return answer.rstrip(".!?,;:").strip().lower()


# ---------------------------------------------------------------- NLA verbalizer
def nla_prompt(tok):
    content = NLA_ACTOR.format(injection_char=NLA_INJ_CHAR)
    ids = chat_ids(tok, [{"role": "user", "content": content}], add_generation_prompt=True)
    m = [i for i, t in enumerate(ids) if t == NLA_INJ_ID]
    assert len(m) == 1 and ids[m[0] - 1] == NLA_LEFT and ids[m[0] + 1] == NLA_RIGHT, "NLA marker contract broken"
    return ids, m[0]


@torch.no_grad()
def nla_generate(model, tok, inj, vectors, max_new, batch=32, progress=None):
    """vectors: list of [d] tensors (base-model layer-42 residuals). Greedy (show_nla_generations.py). Returns raw text
    (special tokens skipped)."""
    ids0, m = nla_prompt(tok)
    out = []
    dev = next(model.parameters()).device
    for k in range(0, len(vectors), batch):
        vs = vectors[k:k + batch]
        ids = torch.tensor([ids0] * len(vs), device=dev)
        inj.set([([m], v.view(1, -1)) for v in vs])
        try:
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id, temperature=None, top_p=None,
                                 top_k=None)
        finally:
            inj.clear()
        out += [t for t in tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)]
        if progress:
            progress(k + len(vs), len(vectors))
    return out


def nla_explanation(text):
    """The verbalisation with the explanation tags removed (the model often drops the opening '<')."""
    t = (text or "").split("</think>")[-1]
    t = re.sub(r"<?/?explanation>", " ", t)
    return re.sub(r"[ \t]+", " ", t).strip()


# ---------------------------------------------------------------- J-lens
class JLens:
    """Neuronpedia J-lens (jlens/pod_jlens.py): logits = lm_head(final_norm(J_l h)); only the band layers are kept."""

    def __init__(self, model, band, lens=None, path=None):
        _, self.norm, self.lm_head = decoder(model)
        self.dev = self.lm_head.weight.device
        if lens is None:
            ck = torch.load(path, map_location="cpu", weights_only=True)
            lens = {int(k): v for k, v in ck["J"].items() if int(k) in band}
            del ck
        self.J = {L: lens[L].to(self.dev, torch.float16) for L in band}

    @torch.no_grad()
    def topk(self, resid, layer, k=20):
        h = resid.to(self.dev).float() @ self.J[layer].float().T
        lg = self.lm_head(self.norm(h.to(self.lm_head.weight.dtype))).float()
        lp = torch.log_softmax(lg, -1)
        v, i = lp.topk(k)
        return i.cpu(), v.cpu()


# ---------------------------------------------------------------- misc
def prefix_key(ids, pos):
    """Identical prefixes give identical activations: reads are de-duplicated on this key."""
    return hashlib.sha1(json.dumps(ids[: pos + 1]).encode()).hexdigest()[:16]


def hf_file(repo, rev, fn):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fn, revision=rev)
