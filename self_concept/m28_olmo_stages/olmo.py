"""SC-M28 OLMo 3 helpers (DESIGN.md §2-3): the stages, their local copies, the call parser, the trainer adaptation,
the generator and the forced (log-prob) reads.

STAGES      OLMo 3 7B along the Instruct line at SC-M15's pinned revisions: base (after midtraining), i_sft, i_dpo,
            i_rl (the released Instruct), plus pt1414k (the last stage-1 pretraining checkpoint, before midtraining;
            exploratory).
prepare     download one stage (safetensors only) to a local dir. The two pretraining-line stages have no chat
            template: they get the Instruct template (data/chat_template.jinja, i_rl's, verbatim) in their own
            tokenizer, where <functions> / <function_calls> are plain text, not the Instruct line's added tokens.
parse_any   V1's parse_option (Qwen XML, JSON) plus OLMo's own call text: choose_option(..., option_id="x", ...).
adapt       SC-S1's Llama adaptation for training, OLMo's way: targets rendered by the template (encode_tmpl; a V1 XML
            call becomes OLMo's <function_calls>name(k="v", ...)</function_calls>, ending in the template's eos), LoRA on
            the seven projections, the generic target loss.
OlmoGen     vLLM on one merged model, V1's prompt (fmt.prompt_ids) and sampling, plus STOP_IDS on every stage (the
            post-trained stages stop there anyway; the pretraining-line stages have no end-of-turn in their config).
forced      next-token log-probs, no sampling: the act at the option id after a forced call prefix, and the letter
            items at the reply's first token.
"""
import json
import math
import re
from pathlib import Path

from persona.exemplar.mini0 import parse_option

H = Path(__file__).resolve().parent
TEMPLATE = H / "data" / "chat_template.jinja"
STAGES = {   # name -> (repo, revision, line); revisions are SC-M15's
    "pt1414k": ("allenai/Olmo-3-1025-7B", "373bad25002f1624757a73235c5ca844c6375c25", "pretrain"),
    "base": ("allenai/Olmo-3-1025-7B", "a81bae42db3975be1671e27b9c9a56da1a9f980f", "pretrain"),
    "i_sft": ("allenai/Olmo-3-7B-Instruct-SFT", "e1452fc572d51966ff4aaeb25118b891eb93e549", "instruct"),
    "i_dpo": ("allenai/Olmo-3-7B-Instruct-DPO", "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1", "instruct"),
    "i_rl": ("allenai/Olmo-3-7B-Instruct", "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc", "instruct"),
}
ORDER = ("base", "i_sft", "i_dpo", "i_rl")           # the four stages of the brief
STOP_IDS = [100257, 100264, 100265]                 # <|endoftext|>, <|im_start|>, <|im_end|> (same ids at every stage)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
OPT_RE = re.compile(r"\boption_id\s*=\s*[\"']([^\"']+)[\"']")
CASE_RE = re.compile(r"^Case ([A-Z0-9_]+)\.")
LETTER_BLOCKS = ("self_mc", "self_mc_dep", "levels", "lottery", "mc")
ACT_BLOCKS = ("just", "deploy")


def prepare(stage, root="/root/hf/m28"):
    from huggingface_hub import snapshot_download
    repo, rev, line = STAGES[stage]
    path = Path(snapshot_download(repo, revision=rev, local_dir=str(Path(root) / stage),
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
                                  ignore_patterns=["pytorch_model*"]))
    if line == "pretrain":
        cfg_p = path / "tokenizer_config.json"
        cfg = json.loads(cfg_p.read_text())
        cfg["chat_template"] = TEMPLATE.read_text()
        cfg_p.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    return path


def parse_any(text):
    o = parse_option(text)
    if o is not None:
        return o
    m = OPT_RE.search(text)
    return m.group(1).strip().lower() if m else None


def patch_parsers():
    """Point every module that reads a choose_option call at parse_any (this process only)."""
    from persona_affect.v1_value_hook import evalrun as R
    from persona_affect.v1c_selfreport import analyze as V1C
    R.parse_option = parse_any
    V1C.parse_option = parse_any


def adapt():
    import persona.operative.lora_c1e4.common as C
    import persona.operative.lora_c1e4.train as CT
    from persona_affect.v1_value_hook import train as TR
    from self_concept.s1_scale import train_any as TA
    TR.encode = TA.encode_tmpl
    C.TARGET_MODULES = list(TARGETS)
    CT.target_loss = TA.target_loss_any


class OlmoGen:
    def __init__(self, model):
        import time
        from vllm import LLM
        t0 = time.time()
        self.llm = LLM(model=str(model), dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0,
                       enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()
        self.load_s = round(time.time() - t0, 1)

    def __call__(self, batch):
        from vllm import SamplingParams, TokensPrompt

        from persona_affect.v1_value_hook.fmt import prompt_ids
        prompts = [TokensPrompt(prompt_token_ids=prompt_ids(self.tok, b["messages"], b["tools"])) for b in batch]
        res = self.llm.generate(prompts, [SamplingParams(**b["sampling"], stop_token_ids=STOP_IDS) for b in batch],
                                use_tqdm=False)
        return [(r.outputs[0].text, r.outputs[0].finish_reason) for r in res]

    def logprobs(self, seqs):
        """Sum of log-probs of each (prompt, continuation) pair's continuation tokens."""
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p + c) for p, c in seqs],
                                SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0), use_tqdm=False)
        out = []
        for (p, c), r in zip(seqs, res):
            pl = r.prompt_logprobs
            out.append(sum(pl[len(p) + j][t].logprob for j, t in enumerate(c)))
        return out


def _distinct(items, blocks):
    """One entry per distinct prompt of the blocks (items repeat a prompt across samples)."""
    seen, out = set(), []
    for it in items:
        if it["block"] not in blocks:
            continue
        k = json.dumps([it["messages"], it["tools"]], sort_keys=True)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def forced_jobs(tok, items):
    """[(item, {label: (prompt_ids, continuation_ids)})]. act: the call prefix up to option_id=", each option's id
    and the '",' that closes it (one token in OLMo's vocabulary, as the model writes it before ' reason'), jointly
    tokenised and split where the options diverge. letters: "X" and " X" as the reply's
    first token (logsumexp'd in forced_rows)."""
    from persona_affect.v1_value_hook.fmt import prompt_ids
    jobs = []
    for it in _distinct(items, ACT_BLOCKS):
        user = next(m["content"] for m in it["messages"] if m["role"] == "user")
        case = CASE_RE.match(user).group(1)
        p = prompt_ids(tok, it["messages"], it["tools"])
        pre = f'<function_calls>choose_option(case_id="{case}", option_id="'
        joint = {o: tok(pre + o + '",', add_special_tokens=False)["input_ids"] for o in it["amap"]}
        seqs = list(joint.values())
        k = 0
        while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
            k += 1
        jobs.append((it, {o: (p + s[:k], s[k:]) for o, s in joint.items()}))
    for it in _distinct(items, LETTER_BLOCKS):
        p = prompt_ids(tok, it["messages"], it["tools"])
        letters = sorted(it["amap"]) if it.get("amap") else ["A", "B"]
        conts = {}
        for L in letters:
            for v in (L, " " + L):
                conts[f"{L}|{v}"] = (p, tok(v, add_special_tokens=False)["input_ids"][:1])
        jobs.append((it, conts))
    return jobs


def forced_rows(jobs, scores, cell):
    """One row per distinct prompt: p over its options or letters (renormalised), the option / letter mass, and the
    item's identifying fields."""
    rows, i = [], 0
    for it, conts in jobs:
        lp = dict(zip(conts, scores[i:i + len(conts)]))
        i += len(conts)
        if it["block"] in ACT_BLOCKS:
            by = lp
        else:
            by = {}
            for k, v in lp.items():
                L = k.split("|")[0]
                by[L] = v if L not in by else max(by[L], v) + math.log1p(math.exp(-abs(by[L] - v)))
        mass = sum(math.exp(v) for v in by.values())
        p = {k: math.exp(v) / mass for k, v in by.items()}
        rows.append({k: it.get(k) for k in ("key", "block", "cond", "tid", "i", "amap", "risky")}
                    | {"cell": cell, "p": p, "mass": mass})
    return rows
