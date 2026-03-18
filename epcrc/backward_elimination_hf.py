"""EPCRC backward elimination on real Hugging Face text models.

Provides reusable building blocks (model loading, masking interventions,
query-grid construction, response-matrix building) and the main
``run_backward_elimination_real_models`` driver used by experiment scripts.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Ensure epcrc package can be imported whether run as module or directly
_epcrc_root = Path(__file__).parent.parent
if str(_epcrc_root) not in sys.path:
    sys.path.insert(0, str(_epcrc_root))

from epcrc.coverage import CoverageFunctional
from epcrc.pruning import BackwardEliminationPruner
from epcrc.utils import make_stable_seed



def uniquify_names(names: Sequence[str]) -> List[str]:
    """Make duplicate names explicit: x, x#2, x#3, ..."""
    counts = {}
    out: List[str] = []
    for n in names:
        c = counts.get(n, 0) + 1
        counts[n] = c
        out.append(n if c == 1 else f"{n}#{c}")
    return out


def mask_text(text: str, theta: float, seed: int, mask_token: str = "[MASK]") -> str:
    """Simple token masking intervention for text.

    Masks approximately theta fraction of whitespace tokens.
    """
    words = text.split()
    n = len(words)
    if n == 0:
        return text

    k = int(round(float(theta) * n))
    if k <= 0:
        return text

    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    chosen = set(idxs[: min(k, n)])

    out = [mask_token if i in chosen else w for i, w in enumerate(words)]
    return " ".join(out)


@dataclass
class HFTextModel:
    name: str
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    device: str

    @classmethod
    def load(cls, model_id: str, device: str) -> "HFTextModel":
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception as e_fast:
            print(
                f"[WARN] Fast tokenizer failed for {model_id}: {e_fast}. "
                "Retrying with slow tokenizer (use_fast=False)."
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()
        model.to(device)
        return cls(name=model_id, tokenizer=tokenizer, model=model, device=device)

    def score_batch(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Return scalarized outputs (positive-class probability) for texts."""
        outputs: List[float] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(**enc).logits

            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits[:, 0])
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]

            outputs.extend([float(x) for x in probs.detach().cpu().numpy()])

        return np.asarray(outputs, dtype=float)


def sample_sst2_sentences(max_samples: int, seed: int = 0) -> List[str]:
    ds = load_dataset("glue", "sst2", split="validation")
    texts = [str(x["sentence"]) for x in ds]

    rng = np.random.RandomState(seed)
    idx = np.arange(len(texts))
    rng.shuffle(idx)

    m = min(max_samples, len(texts))
    return [texts[i] for i in idx[:m]]


def split_fit_eval(texts: Sequence[str]) -> Tuple[List[str], List[str]]:
    n = len(texts)
    n_fit = max(1, n // 2)
    fit = list(texts[:n_fit])
    eval_ = list(texts[n_fit:])

    if len(eval_) == 0:
        eval_ = fit.copy()

    return fit, eval_


def build_query_grid(texts: Iterable[str], doses: Sequence[float]) -> Tuple[List[str], List[float], List[int]]:
    X: List[str] = []
    Theta: List[float] = []
    seeds: List[int] = []

    for text in texts:
        for theta in doses:
            theta_f = float(theta)
            seed = make_stable_seed(text, f"{theta_f:.6f}")
            X.append(text)
            Theta.append(theta_f)
            seeds.append(int(seed))

    return X, Theta, seeds


def query_all_models(
    models: Sequence[HFTextModel],
    X: Sequence[str],
    Theta: Sequence[float],
    seeds: Sequence[int],
    batch_size: int = 16,
) -> np.ndarray:
    """Build response matrix Y, shape (n_queries, n_models)."""
    n = len(X)
    if not (len(Theta) == n and len(seeds) == n):
        raise ValueError("X, Theta, seeds must have the same length")

    Y_cols = []
    for m in tqdm(models, desc="Querying models"):
        perturbed = [mask_text(x, t, s) for x, t, s in zip(X, Theta, seeds)]
        y = m.score_batch(perturbed, batch_size=batch_size)
        Y_cols.append(y)

    return np.column_stack(Y_cols)


def run_backward_elimination_real_models(
    model_ids: Sequence[str],
    max_samples: int,
    gamma: float,
    doses_fit: Sequence[float],
    doses_eval: Sequence[float],
    metric: str,
    batch_size: int,
    output_json: Optional[str] = None,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading models...")
    models: List[HFTextModel] = []
    for mid in model_ids:
        try:
            models.append(HFTextModel.load(mid, device=device))
        except Exception as e:
            print(f"[WARN] Skipping model {mid} due to load error: {e}")

    if len(models) < 2:
        raise RuntimeError(
            "Need at least 2 successfully loaded models to run pruning."
        )

    if len(models) < len(model_ids):
        print(f"Proceeding with {len(models)} loaded models out of {len(model_ids)} requested.")

    model_names = uniquify_names([m.name for m in models])

    texts = sample_sst2_sentences(max_samples=max_samples, seed=0)
    fit_texts, eval_texts = split_fit_eval(texts)

    print(f"Total texts={len(texts)}, fit={len(fit_texts)}, eval={len(eval_texts)}")
    print(f"doses_fit={list(map(float, doses_fit))}")
    print(f"doses_eval={list(map(float, doses_eval))}")

    fit_X, fit_Theta, fit_seeds = build_query_grid(fit_texts, doses_fit)
    eval_X, eval_Theta, eval_seeds = build_query_grid(eval_texts, doses_eval)

    print("Building Y_fit...")
    Y_fit = query_all_models(models, fit_X, fit_Theta, fit_seeds, batch_size=batch_size)

    print("Building Y_eval...")
    Y_eval = query_all_models(models, eval_X, eval_Theta, eval_seeds, batch_size=batch_size)

    coverage_fn = CoverageFunctional(
        Y_fit=Y_fit,
        Y_eval=Y_eval,
        model_names=model_names,
        metric=metric,
    )

    pruner = BackwardEliminationPruner(coverage_fn=coverage_fn, tolerance_gamma=gamma)
    result = pruner.run(debug=False)

    print("\n" + "="*80)
    print("BACKWARD ELIMINATION ALGORITHM EXECUTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  • Tolerance threshold (γ): {gamma}")
    print(f"  • Uniqueness metric: {metric}")
    print(f"  • Total models started with: {len(model_names)}")
    print(f"\nAlgorithm: Start with all models, greedily remove one model at a time.")
    print(f"Decision rule: Remove model j from set S if E(S \\ {{j}}) ≤ γ = {gamma}")
    print(f"Meaning: If coverage error is still acceptable after removing j, remove it (it's redundant).")
    
    print(f"\n" + "-"*80)
    print("ITERATION LOG:")
    print("-"*80)
    
    for step in result.history:
        if step.action == "remove":
            print(f"iter {step.iteration}: remove {step.removed_model_name}")
            print(f"         E(S) = {step.coverage:.6f} ≤ γ = {gamma} ✓")
        else:
            print(f"iter {step.iteration}: STOP — no model can be removed without E(S) > γ")
            for j in sorted(step.kept_set):
                s_candidate = set(step.kept_set)
                s_candidate.remove(j)
                e_candidate, _ = coverage_fn.compute_coverage(s_candidate)
                print(
                    f"         try remove {model_names[j]}: "
                    f"E(S\\{{j}}) = {e_candidate:.6f} > γ = {gamma} ✗"
                )

    bottleneck = max(
        (cert for idx, cert in result.certificates.items() if idx in result.kept_set),
        key=lambda c: c.uniqueness
    )
    final_kept = [model_names[i] for i in sorted(result.kept_set)]
    removed_indices = [i for i in range(len(model_names)) if i not in result.kept_set]
    removed_names = [model_names[i] for i in removed_indices]
    
    print(f"\n" + "-"*80)
    print("FINAL RESULT:")
    print("-"*80)
    print(f"\n✓ Models kept: {len(final_kept)}")
    for i, m in enumerate(final_kept, 1):
        print(f"    {i}. {m}")
    
    print(f"\n✗ Models removed: {len(removed_names)}")
    if removed_names:
        for i, m in enumerate(removed_names, 1):
            print(f"    {i}. {m}")
    else:
        print("    (None - all models were needed)")
    
    cert_rows = []
    for idx in sorted(result.kept_set):
        cert = result.certificates[idx]
        row = {
            "model_idx": int(idx),
            "model_name": cert.model_name,
            "uniqueness_U_i_given_S": float(cert.uniqueness),
            "weights": [float(x) for x in cert.weights.tolist()],
        }
        cert_rows.append(row)

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        payload = {
            "gamma": float(gamma),
            "metric": metric,
            "model_names": model_names,
            "n_models_loaded": len(model_names),
            "max_samples": int(max_samples),
            "doses_fit": [float(x) for x in doses_fit],
            "doses_eval": [float(x) for x in doses_eval],
            "kept_set_indices": [int(i) for i in sorted(result.kept_set)],
            "kept_set_names": final_kept,
            "removed_set_indices": [int(i) for i in removed_indices],
            "removed_set_names": removed_names,
            "E_of_S": float(result.coverage),
            "bottleneck_model": bottleneck.model_name,
            "bottleneck_uniqueness": float(bottleneck.uniqueness),
            "history": [
                {
                    "iteration": int(step.iteration),
                    "action": step.action,
                    "removed_model_idx": None if step.removed_model_idx is None else int(step.removed_model_idx),
                    "removed_model_name": step.removed_model_name,
                    "kept_set_indices": [int(i) for i in sorted(step.kept_set)],
                    "kept_set_names": [model_names[i] for i in sorted(step.kept_set)],
                    "coverage_E_of_S": float(step.coverage),
                }
                for step in result.history
            ],
            "certificates": cert_rows,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved result JSON: {output_json}")


