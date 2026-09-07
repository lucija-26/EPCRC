"""Deterministic three-label probability extraction from an open-weight judge.

The response tensor must be reproducible, so the main matrix is never sampled.
For each rendered prompt the judge is asked, in effect, "how likely is the
continuation A, B, or C?", and the three sequence log-likelihoods are turned
into a distribution by a softmax restricted to those three candidates.

Two details decide whether the numbers mean anything:

  * **Tokenization.** " A" and "A" are different tokens, and some tokenizers
    split a label into several pieces.  If the labels do not share a boundary
    convention the softmax compares incomparable quantities, so
    `audit_tokenization` records the encoding for every model and refuses to
    proceed silently when the convention is inconsistent.
  * **Orientation.** Under an A/B swap the judge's "A" mass belongs to
    canonical response B.  `epcrc.prompts.canonicalize` performs that mapping
    and `score_pairs` applies it, so stored probabilities are always canonical.

Everything here is torch/transformers-only at call time; the module imports
cleanly without them so the pure logic stays testable on a laptop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .prompts import (
    LABELS,
    Context,
    canonicalize,
    prompt_sha256,
    render,
)
from .rewardbench import JudgedPair


@dataclass
class TokenizationReport:
    """Per-model record of how the three labels encode."""

    model_id: str
    label_token_ids: Dict[str, List[int]]
    label_token_strings: Dict[str, List[str]]
    n_continuation_tokens: Dict[str, int]
    single_token_labels: bool
    consistent_boundary: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id,
            "label_token_ids": self.label_token_ids,
            "label_token_strings": self.label_token_strings,
            "n_continuation_tokens": self.n_continuation_tokens,
            "single_token_labels": self.single_token_labels,
            "consistent_boundary": self.consistent_boundary,
        }


def normalize_label_scores(scores: Sequence[float]) -> np.ndarray:
    """Softmax over the three candidate log-likelihoods.

    Shifted by the maximum before exponentiating; the raw sequence
    log-likelihoods are large negative numbers and would otherwise underflow to
    a zero denominator.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"expected three label scores, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"non-finite label scores: {arr}")

    shifted = np.exp(arr - arr.max())
    return shifted / shifted.sum()


class LabelScorer:
    """Scores rendered prompts against the labels A, B and C.

    Loading is lazy so importing this module costs nothing on a machine without
    a GPU.  Generation is never used; the model runs in inference mode and the
    only randomness-bearing code path (sampling) is not reached.
    """

    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        label_prefix: str = " ",
        max_prompt_tokens: int = 8192,
    ):
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        # Labels are scored with a leading space because that is how they would
        # continue the prompt after "Verdict:"; the audit records the choice.
        self.label_prefix = label_prefix
        self.max_prompt_tokens = max_prompt_tokens
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            dtype=getattr(torch, self.dtype),
            device_map=self.device,
        )
        self._model.eval()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def label_encodings(self) -> Dict[str, List[int]]:
        return {
            label: self.tokenizer.encode(
                self.label_prefix + label, add_special_tokens=False
            )
            for label in LABELS
        }

    def audit_tokenization(self) -> TokenizationReport:
        """Record how each label encodes, and whether the labels are comparable."""
        encodings = self.label_encodings()
        lengths = {label: len(ids) for label, ids in encodings.items()}

        return TokenizationReport(
            model_id=self.model_id,
            label_token_ids=encodings,
            label_token_strings={
                label: self.tokenizer.convert_ids_to_tokens(ids)
                for label, ids in encodings.items()
            },
            n_continuation_tokens=lengths,
            single_token_labels=all(n == 1 for n in lengths.values()),
            consistent_boundary=len(set(lengths.values())) == 1,
        )

    def apply_chat_template(self, prompt: str) -> str:
        """Wrap the prompt in the model's official chat template.

        Qwen 3 exposes an optional thinking mode; it is disabled here because
        free-form reasoning before the label would make the next-token
        distribution meaningless for forced-choice scoring.
        """
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def score_prompt(self, prompt: str) -> np.ndarray:
        """Three-class probabilities for one rendered prompt."""
        return self.score_prompts([prompt])[0]

    def score_prompts(self, prompts: Sequence[str]) -> np.ndarray:
        """Three-class probabilities for a batch of rendered prompts."""
        import torch

        encodings = self.label_encodings()
        single_token = all(len(ids) == 1 for ids in encodings.values())

        texts = [self.apply_chat_template(p) + self.label_prefix.rstrip() for p in prompts]
        out = np.empty((len(prompts), 3), dtype=float)

        with torch.inference_mode():
            if single_token:
                batch = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_tokens,
                    add_special_tokens=False,
                ).to(self.model.device)

                logits = self.model(**batch).logits
                # Left padding would move the final position; index the true
                # last token of each sequence instead of assuming -1.
                last = batch["attention_mask"].sum(dim=1) - 1
                final = logits[torch.arange(logits.shape[0]), last, :]
                log_probs = torch.log_softmax(final.float(), dim=-1)

                columns = [encodings[label][0] for label in LABELS]
                scores = log_probs[:, columns].cpu().numpy()
                for i in range(len(prompts)):
                    out[i] = normalize_label_scores(scores[i])
            else:
                for i, text in enumerate(texts):
                    scores = [
                        self._continuation_logprob(text, encodings[label])
                        for label in LABELS
                    ]
                    out[i] = normalize_label_scores(scores)

        return out

    def _continuation_logprob(self, text: str, label_ids: Sequence[int]) -> float:
        """Exact sequence log-likelihood of a multi-token label."""
        import torch

        prompt_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_tokens,
        )["input_ids"]
        full = torch.tensor(
            [list(prompt_ids) + list(label_ids)], device=self.model.device
        )

        logits = self.model(full).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)[0]

        total = 0.0
        for offset, token_id in enumerate(label_ids):
            position = len(prompt_ids) + offset - 1
            total += float(log_probs[position, token_id])
        return total


def score_pairs(
    scorer: LabelScorer,
    pairs: Sequence[JudgedPair],
    context: Context,
    batch_size: int = 8,
    progress: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Canonical three-class probabilities for every pair under one context.

    Returns an ``(n_pairs, 3)`` array ordered as (A better, B better, tie) in
    canonical response identity, plus the hash of each rendered prompt.
    """
    prompts = [
        render(p.prompt, p.response_a, p.response_b, context=context) for p in pairs
    ]
    hashes = [prompt_sha256(text) for text in prompts]

    out = np.empty((len(pairs), 3), dtype=float)
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        probabilities = scorer.score_prompts(chunk)
        for offset, row in enumerate(probabilities):
            out[start + offset] = canonicalize(tuple(row), context)
        if progress:
            print(f"  scored {min(start + batch_size, len(prompts))}/{len(prompts)}",
                  flush=True)

    return out, hashes


def argmax_labels(probabilities: np.ndarray) -> List[str]:
    return [LABELS[int(i)] for i in np.asarray(probabilities).argmax(axis=1)]


def accuracy_report(
    probabilities: np.ndarray,
    pairs: Sequence[JudgedPair],
) -> Dict[str, float]:
    """Sanity statistics for one judge under one context.

    A judge whose predictions are near-uniform, or which never uses the tie
    label, is scoring the labels wrongly rather than judging badly, so these are
    quality gates on the pipeline and not results about the judges.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = argmax_labels(probabilities)
    gold = [p.gold_label for p in pairs]

    non_tie = [i for i, p in enumerate(pairs) if not p.is_tie]
    tie = [i for i, p in enumerate(pairs) if p.is_tie]

    binary_correct = [
        predicted[i] == gold[i] for i in non_tie if predicted[i] in ("A", "B")
    ]

    return {
        "n_pairs": len(pairs),
        "three_class_accuracy": float(np.mean([
            predicted[i] == gold[i] for i in range(len(pairs))
        ])) if pairs else 0.0,
        "binary_accuracy_non_tie": (
            float(np.mean(binary_correct)) if binary_correct else float("nan")
        ),
        "predicted_tie_rate": float(np.mean([lab == "C" for lab in predicted])),
        "gold_tie_rate": float(len(tie) / len(pairs)) if pairs else 0.0,
        "mean_max_probability": float(probabilities.max(axis=1).mean()),
        "position_bias_a_rate": float(np.mean([lab == "A" for lab in predicted])),
    }


def write_scores(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
