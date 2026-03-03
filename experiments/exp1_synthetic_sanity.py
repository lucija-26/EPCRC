# experiments/exp1_synthetic_sanity.py
"""
Experiment 1: Synthetic Sanity Check

Tests the pruning framework on synthetic linear ecosystems where
we know the ground truth redundancy structure.

Goals:
- Verify DISCO solver correctly identifies redundant models
- Verify pruning algorithms find minimal representative sets
- Confirm coverage functional is monotone
"""

import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.synthetic import (
    LinearModel,
    NoiseIntervention,
    generate_redundant_ecosystem,
)
from epcrc.ecosystem import Ecosystem
from epcrc.coverage import CoverageFunctional
from epcrc.pruning import BackwardElimination, ForwardSelection, GreedyPruning


def run_sanity_experiment(
    n_unique: int = 5,
    n_redundant: int = 10,
    dim: int = 20,
    n_samples: int = 500,
    tolerance: float = 0.1,
    seed: int = 42,
):
    """
    Run sanity check on synthetic ecosystem.
    
    Creates an ecosystem with known redundancy structure and verifies
    that pruning recovers (approximately) the unique models.
    
    Args:
        n_unique: Number of truly unique models
        n_redundant: Number of redundant models
        dim: Input dimension
        n_samples: Number of query samples
        tolerance: Pruning tolerance γ
        seed: Random seed
    """
    print("=" * 60)
    print("Experiment 1: Synthetic Sanity Check")
    print("=" * 60)
    
    np.random.seed(seed)
    
    # Generate ecosystem with known structure
    print(f"\nGenerating ecosystem: {n_unique} unique + {n_redundant} redundant models")
    models = generate_redundant_ecosystem(
        n_unique=n_unique,
        n_redundant=n_redundant,
        dim=dim,
        noise_std=0.0,
        redundancy_noise=0.01,
        seed=seed,
    )
    
    ecosystem = Ecosystem(models)
    print(f"Total models: {ecosystem.n_models}")
    print(f"Model names: {ecosystem.model_names}")
    
    # Generate query data
    print(f"\nGenerating {n_samples} query samples...")
    X = [np.random.randn(dim) for _ in range(n_samples)]
    intervention = NoiseIntervention()
    
    # Split into fit and eval
    n_fit = n_samples // 2
    X_fit = X[:n_fit]
    X_eval = X[n_fit:]
    
    # Query ecosystem
    print("Querying ecosystem...")
    Y_fit = ecosystem.batched_query(X_fit, Thetas=0.1, intervention=intervention)
    Y_eval = ecosystem.batched_query(X_eval, Thetas=0.1, intervention=intervention)
    
    print(f"Y_fit shape: {Y_fit.shape}")
    print(f"Y_eval shape: {Y_eval.shape}")
    
    # Build coverage functional
    coverage = CoverageFunctional(
        Y_fit=Y_fit,
        Y_eval=Y_eval,
        model_names=ecosystem.model_names,
    )
    
    # Test coverage monotonicity
    print("\n--- Testing Coverage Monotonicity ---")
    full_set = set(range(ecosystem.n_models))
    E_full, _ = coverage.compute_coverage(full_set)
    print(f"E(J) = {E_full:.6f} (should be ~0)")
    
    # Remove one model
    for i in range(min(5, ecosystem.n_models)):
        S_minus = full_set - {i}
        E_minus, _ = coverage.compute_coverage(S_minus)
        print(f"E(J \\ {{{ecosystem.model_names[i]}}}) = {E_minus:.6f}")
    
    # Run pruning algorithms
    print(f"\n--- Running Pruning Algorithms (γ = {tolerance}) ---")
    
    # Backward elimination
    print("\n1. Backward Elimination:")
    be = BackwardElimination(coverage, tolerance=tolerance, verbose=True)
    result_be = be.run()
    print(f"Final: {len(result_be.kept_set)} models kept")
    print(f"Kept: {result_be.kept_names}")
    
    # Forward selection
    print("\n2. Forward Selection:")
    fs = ForwardSelection(coverage, tolerance=tolerance, verbose=True)
    result_fs = fs.run()
    print(f"Final: {len(result_fs.kept_set)} models kept")
    print(f"Kept: {result_fs.kept_names}")
    
    # Greedy pruning
    print("\n3. Greedy Pruning:")
    gp = GreedyPruning(coverage, tolerance=tolerance, verbose=True)
    result_gp = gp.run()
    print(f"Final: {len(result_gp.kept_set)} models kept")
    print(f"Kept: {result_gp.kept_names}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Ground truth unique models: {n_unique}")
    print(f"Backward elimination kept: {len(result_be.kept_set)}")
    print(f"Forward selection kept: {len(result_fs.kept_set)}")
    print(f"Greedy pruning kept: {len(result_gp.kept_set)}")
    
    # Check how many unique models were kept
    unique_names = {f"unique_{i}" for i in range(n_unique)}
    
    be_unique = len(set(result_be.kept_names) & unique_names)
    fs_unique = len(set(result_fs.kept_names) & unique_names)
    gp_unique = len(set(result_gp.kept_names) & unique_names)
    
    print(f"\nUnique models retained:")
    print(f"  Backward: {be_unique}/{n_unique}")
    print(f"  Forward:  {fs_unique}/{n_unique}")
    print(f"  Greedy:   {gp_unique}/{n_unique}")
    
    return {
        "backward": result_be,
        "forward": result_fs,
        "greedy": result_gp,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Exp 1: Synthetic Sanity Check")
    parser.add_argument("--n_unique", type=int, default=5)
    parser.add_argument("--n_redundant", type=int, default=10)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    run_sanity_experiment(
        n_unique=args.n_unique,
        n_redundant=args.n_redundant,
        dim=args.dim,
        n_samples=args.n_samples,
        tolerance=args.tolerance,
        seed=args.seed,
    )
