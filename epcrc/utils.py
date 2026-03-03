# epcrc/utils.py
"""
Utility functions for EPCRC experiments.
"""

import hashlib
import numpy as np
from typing import List, Tuple


def make_stable_seed(
    *args,
    **kwargs,
) -> int:
    """
    Create a deterministic seed from any hashable arguments.
    
    This ensures reproducibility across runs - the same (text, theta, context)
    always produces the same intervention seed.
    
    Example:
        seed = make_stable_seed(text="hello", theta=0.5, context="test")
    """
    # Combine all arguments into a string
    parts = [str(a) for a in args]
    parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
    combined = "|".join(parts)
    
    # Hash to integer
    h = hashlib.md5(combined.encode()).hexdigest()
    return int(h[:8], 16)


def split_fit_eval(
    data: List,
    fit_fraction: float = 0.5,
    seed: int = None,
) -> Tuple[List, List]:
    """
    Split data into fitting and evaluation sets.
    
    This implements the honest splitting requirement from the paper:
    P_fit and P_eval must be independent.
    
    Args:
        data: List of samples
        fit_fraction: Fraction for fitting (rest goes to eval)
        seed: Random seed for reproducibility
        
    Returns:
        (fit_data, eval_data)
    """
    n = len(data)
    n_fit = int(n * fit_fraction)
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    indices = rng.permutation(n)
    fit_idx = indices[:n_fit]
    eval_idx = indices[n_fit:]
    
    fit_data = [data[i] for i in fit_idx]
    eval_data = [data[i] for i in eval_idx]
    
    return fit_data, eval_data


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Useful for risk-controlled pruning (Section 4.3).
    
    Args:
        values: Array of values to bootstrap
        confidence: Confidence level (e.g., 0.95)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed
        
    Returns:
        (mean, lower_ci, upper_ci)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    values = np.asarray(values)
    n = len(values)
    
    # Bootstrap samples
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    
    means = np.array(means)
    
    # Percentiles
    alpha = 1 - confidence
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    
    return float(np.mean(values)), float(lower), float(upper)
