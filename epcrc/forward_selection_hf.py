"""EPCRC forward selection on real Hugging Face text models.

Reuses building blocks from backward_elimination_hf (model loading, masking
interventions, query-grid construction, response-matrix building) and provides
the ``run_forward_selection_real_models`` driver for experiment scripts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

_epcrc_root = Path(__file__).parent.parent
if str(_epcrc_root) not in sys.path:
    sys.path.insert(0, str(_epcrc_root))

from epcrc.backward_elimination_hf import (
    HFTextModel,
    build_query_grid,
    query_all_models,
    sample_sst2_sentences,
    split_fit_eval,
    uniquify_names,
)
from epcrc.coverage import CoverageFunctional
from epcrc.pruning import ForwardSelectionPruner


def run_forward_selection_real_models(
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

    pruner = ForwardSelectionPruner(coverage_fn=coverage_fn, tolerance_gamma=gamma)
    result = pruner.run(debug=False)

    print("\n" + "=" * 80)
    print("FORWARD SELECTION ALGORITHM EXECUTION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  • Tolerance threshold (γ): {gamma}")
    print(f"  • Uniqueness metric: {metric}")
    print(f"  • Total models in ecosystem: {len(model_names)}")
    print(f"\nAlgorithm: Start with empty set, greedily add one model at a time.")
    print(f"Decision rule: Add model j that minimises E(S ∪ {{j}}). Stop when E(S) ≤ γ = {gamma}")

    print(f"\n" + "-" * 80)
    print("ITERATION LOG:")
    print("-" * 80)

    for step in result.history:
        if step.action == "add":
            status = "✓" if step.coverage <= gamma else "·"
            print(f"iter {step.iteration}: add {step.removed_model_name}")
            print(f"         E(S) = {step.coverage:.6f}  ΣU = {step.sum_uniqueness:.6f}  |S| = {len(step.kept_set)}  {status}")
        else:
            print(f"iter {step.iteration}: STOP — coverage satisfied")

    final_kept = [model_names[i] for i in sorted(result.kept_set)]
    removed_indices = [i for i in range(len(model_names)) if i not in result.kept_set]
    removed_names = [model_names[i] for i in removed_indices]

    print(f"\n" + "-" * 80)
    print("FINAL RESULT:")
    print("-" * 80)
    print(f"\n✓ Models kept: {len(final_kept)}")
    for i, m in enumerate(final_kept, 1):
        print(f"    {i}. {m}")

    print(f"\n✗ Models not selected: {len(removed_names)}")
    if removed_names:
        for i, m in enumerate(removed_names, 1):
            print(f"    {i}. {m}")
    else:
        print("    (None - all models were needed)")

    print(f"\n  E(S) = {result.coverage:.6f}")
    print(f"  ΣU   = {result.sum_uniqueness:.6f}")
    print(f"  |S|  = {len(result.kept_set)}")

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
            "sum_uniqueness": float(result.sum_uniqueness),
            "history": [
                {
                    "iteration": int(step.iteration),
                    "action": step.action,
                    "added_model_idx": None if step.removed_model_idx is None else int(step.removed_model_idx),
                    "added_model_name": step.removed_model_name,
                    "kept_set_indices": [int(i) for i in sorted(step.kept_set)],
                    "kept_set_names": [model_names[i] for i in sorted(step.kept_set)],
                    "coverage_E_of_S": float(step.coverage),
                    "sum_uniqueness": float(step.sum_uniqueness),
                }
                for step in result.history
            ],
            "certificates": cert_rows,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved result JSON: {output_json}")
