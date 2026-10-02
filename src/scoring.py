"""Score metric for design v4.1, Section 7.1.

One forward pass per branch. At the first assistant token (after the empty <think> block
inserted by the Qwen3 template when thinking is disabled), read the next-token distribution
over the allowed single-token labels and report:

  expected_score   sum_k k * P(k | prompt, k in labels) / M      (conditional expected score)
  valid_mass       M = sum_k P(k | prompt)  BEFORE normalisation  (how natural the readout is)
  top_label        argmax over the allowed labels
  top_token        argmax over the full vocab (to see what the model would actually emit)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS_1_9 = tuple(str(i) for i in range(1, 10))


@dataclass
class ScoreReadout:
    expected_score: float
    valid_mass: float
    top_label: str
    top_label_prob: float
    top_token: str
    top_token_prob: float
    label_probs: dict[str, float]
    n_tokens: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class Scorer:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", device: str | None = None,
                 dtype: torch.dtype | None = None, labels: Sequence[str] = LABELS_1_9,
                 attn_implementation: str | None = None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kw).to(device).eval()
        self.device = device
        self.labels = tuple(labels)
        self.label_ids = self._single_token_ids(self.labels)
        self.label_values = torch.tensor([float(l) if l.isdigit() else float(i + 1)
                                          for i, l in enumerate(self.labels)])

    def _single_token_ids(self, labels: Sequence[str]) -> list[int]:
        ids = []
        for lab in labels:
            enc = self.tok.encode(lab, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"label {lab!r} is not a single token: {enc}")
            ids.append(enc[0])
        return ids

    def render(self, messages: list[dict], enable_thinking: bool = False) -> str:
        """Render the chat template with the generation prompt. Thinking off by default."""
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )

    @torch.no_grad()
    def score(self, messages: list[dict], enable_thinking: bool = False) -> ScoreReadout:
        text = self.render(messages, enable_thinking=enable_thinking)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        logits = self.model(**enc).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1).cpu()

        label_p = probs[self.label_ids]
        valid_mass = float(label_p.sum())
        cond = label_p / max(valid_mass, 1e-12)
        expected = float((cond * self.label_values).sum())
        li = int(label_p.argmax())
        ti = int(probs.argmax())
        return ScoreReadout(
            expected_score=expected,
            valid_mass=valid_mass,
            top_label=self.labels[li],
            top_label_prob=float(label_p[li]),
            top_token=self.tok.decode([ti]),
            top_token_prob=float(probs[ti]),
            label_probs={l: float(p) for l, p in zip(self.labels, label_p)},
            n_tokens=int(enc["input_ids"].shape[1]),
        )

    @torch.no_grad()
    def score_batch(self, message_lists: list[list[dict]], batch_size: int = 8,
                    enable_thinking: bool = False) -> list[ScoreReadout]:
        """Same readout as score(), batched with left padding (the last position is the
        first assistant token for every row). Verified against score() in fp32."""
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        texts = [self.render(m, enable_thinking) for m in message_lists]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: list = [None] * len(texts)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            # only the last position is needed; avoids a (batch x seq x vocab) logits tensor
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1].float()
            probs = torch.softmax(logits, dim=-1).cpu()
            for j, i in enumerate(idx):
                pr = probs[j]
                label_p = pr[self.label_ids]
                mass = float(label_p.sum())
                cond = label_p / max(mass, 1e-12)
                li, ti = int(label_p.argmax()), int(pr.argmax())
                out[i] = ScoreReadout(
                    expected_score=float((cond * self.label_values).sum()), valid_mass=mass,
                    top_label=self.labels[li], top_label_prob=float(label_p[li]),
                    top_token=self.tok.decode([ti]), top_token_prob=float(pr[ti]),
                    label_probs={l: float(q) for l, q in zip(self.labels, label_p)},
                    n_tokens=int(enc["attention_mask"][j].sum()))
        return out

    @torch.no_grad()
    def complete(self, messages: list[dict], max_new_tokens: int = 20,
                 enable_thinking: bool = False) -> str:
        """Greedy completion, for spot-checking what the model would naturally emit."""
        text = self.render(messages, enable_thinking=enable_thinking)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
