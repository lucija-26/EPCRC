# experiments/exp2_consolidation_curves.py
"""
Experiment 2: Consolidation Curves

Plots ecosystem coverage E(S) as a function of kept set size |S|.

This is a key experiment from Section 5.3 of the paper:
- Shows how many models can be removed before coverage degrades
- Compares coverage-guided vs random/utility-based pruning
- Visualizes the pruning path
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.synthetic import generate_redundant_ecosystem, NoiseIntervention
from epcrc.ecosystem import Ecosystem
from epcrc.coverage import CoverageFunctional
from epcrc.pruning import GreedyPruning, prune_to_budget


def compute_consolidation_curve(
    coverage: CoverageFunctional,
    method: str = "greedy",
) -> List[Dict]:
    """
    Compute consolidation curve: E(S) vs |S|.
    
    Args:
        coverage: CoverageFunctional
        method: "greedy", "random", or "utility"
        
    Returns:
        List of dicts with {size, coverage, kept_set}
    """
    N = coverage.N
    results = []
    
    if method == "greedy":
        # Greedy: remove model with smallest uniqueness at each step
        S = set(range(N))
        
        while len(S) >= 1:
            E, _ = coverage.compute_coverage(S)
            results.append({
                "size": len(S),
                "coverage": E,
                "kept_set": S.copy(),
            })
            
            if len(S) == 1:
                break
            
            # Find best model to remove
            best_j, best_E = None, float('inf')
            for j in S:
                S_minus = S - {j}
                E_minus, _ = coverage.compute_coverage(S_minus)
                if E_minus < best_E:
                    best_E, best_j = E_minus, j
            
            S.remove(best_j)
    
    elif method == "random":
        # Random: remove models in random order
        S = set(range(N))
        order = np.random.permutation(list(S))
        
        for i, j in enumerate(order):
            E, _ = coverage.compute_coverage(S)
            results.append({
                "size": len(S),
                "coverage": E,
                "kept_set": S.copy(),
            })
            S.remove(j)
        
        # Add empty set
        E, _ = coverage.compute_coverage(set())
        results.append({"size": 0, "coverage": E, "kept_set": set()})
    
    elif method == "utility":
        # Utility-based: remove models with lowest individual performance
        # (simulated by removing models with highest self-uniqueness)
        S = set(range(N))
        
        # Compute individual "utility" (inverse of variance)
        Y = coverage.Y_eval
        variances = np.var(Y, axis=0)
        order = np.argsort(variances)[::-1]  # Remove high-variance first
        
        for j in order:
            if j not in S:
                continue
            
            E, _ = coverage.compute_coverage(S)
            results.append({
                "size": len(S),
                "coverage": E,
                "kept_set": S.copy(),
            })
            S.remove(j)
        
        E, _ = coverage.compute_coverage(S)
        results.append({"size": len(S), "coverage": E, "kept_set": S.copy()})
    
    return results


def run_consolidation_experiment(
    n_models: int = 15,
    dim: int = 20,
    n_samples: int = 500,
    n_trials: int = 5,
    seed: int = 42,
    output_dir: str = "results/figures",
):
    """
    Run consolidation curve experiment.
    
    Args:
        n_models: Total models in ecosystem
        dim: Input dimension
        n_samples: Query samples
        n_trials: Monte Carlo trials
        seed: Random seed
        output_dir: Where to save figures
    """
    print("=" * 60)
    print("Experiment 2: Consolidation Curves")
    print("=" * 60)
    
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {"greedy": [], "random": [], "utility": []}
    
    for trial in tqdm(range(n_trials), desc="Trials"):
        # Generate ecosystem (mix of unique and redundant)
        n_unique = n_models // 3
        n_redundant = n_models - n_unique
        
        models = generate_redundant_ecosystem(
            n_unique=n_unique,
            n_redundant=n_redundant,
            dim=dim,
            noise_std=0.01,
            redundancy_noise=0.05,
            seed=seed + trial,
        )
        
        ecosystem = Ecosystem(models)
        
        # Generate data
        X = [np.random.randn(dim) for _ in range(n_samples)]
        intervention = NoiseIntervention()
        
        n_fit = n_samples // 2
        Y_fit = ecosystem.batched_query(X[:n_fit], 0.1, intervention)
        Y_eval = ecosystem.batched_query(X[n_fit:], 0.1, intervention)
        
        coverage = CoverageFunctional(Y_fit, Y_eval, ecosystem.model_names)
        
        # Compute curves for each method
        for method in ["greedy", "random", "utility"]:
            curve = compute_consolidation_curve(coverage, method)
            for point in curve:
                point["trial"] = trial
                point["method"] = method
            all_results[method].extend(curve)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {"size": r["size"], "coverage": r["coverage"], 
         "trial": r["trial"], "method": r["method"]}
        for method_results in all_results.values()
        for r in method_results
    ])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {"greedy": "blue", "random": "gray", "utility": "orange"}
    labels = {"greedy": "Coverage-guided (Greedy)", 
              "random": "Random removal", 
              "utility": "Utility-based"}
    
    for method in ["random", "utility", "greedy"]:
        method_df = df[df["method"] == method]
        
        # Aggregate across trials
        agg = method_df.groupby("size")["coverage"].agg(["mean", "std"]).reset_index()
        
        ax.plot(
            agg["size"], agg["mean"],
            color=colors[method], label=labels[method], linewidth=2
        )
        ax.fill_between(
            agg["size"], 
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            color=colors[method], alpha=0.2
        )
    
    ax.set_xlabel("Number of Models Kept |S|", fontsize=12)
    ax.set_ylabel("Ecosystem Coverage Error E(S)", fontsize=12)
    ax.set_title("Consolidation Curves: Coverage vs Ecosystem Size", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_models)
    
    # Save
    fig_path = os.path.join(output_dir, "consolidation_curves.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"\nSaved figure to: {fig_path}")
    
    # Also save CSV
    csv_path = os.path.join(output_dir.replace("figures", "tables"), "consolidation_data.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved data to: {csv_path}")
    
    plt.show()
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Exp 2: Consolidation Curves")
    parser.add_argument("--n_models", type=int, default=15)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    run_consolidation_experiment(
        n_models=args.n_models,
        dim=args.dim,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        seed=args.seed,
    )
