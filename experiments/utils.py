# experiments/utils.py
"""
Shared utilities for EPCRC experiments.
"""

import hashlib
import os
import numpy as np
from typing import List, Tuple, Optional


def make_stable_seed(
    *args,
    **kwargs,
) -> int:
    """
    Create a deterministic seed from any hashable arguments.
    
    Ensures reproducibility: same arguments → same seed.
    
    Example:
        seed = make_stable_seed(text="hello", theta=0.5)
    """
    parts = [str(a) for a in args]
    parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
    combined = "|".join(parts)
    h = hashlib.md5(combined.encode()).hexdigest()
    return int(h[:8], 16)


def get_results_dir() -> str:
    """Get the results directory path."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "results")


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def split_fit_eval(
    data: List,
    fit_fraction: float = 0.5,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[List, List]:
    """
    Split data into fitting and evaluation sets.
    
    Args:
        data: List of samples
        fit_fraction: Fraction for fitting
        rng: Random state for reproducibility
        
    Returns:
        (fit_data, eval_data)
    """
    n = len(data)
    n_fit = int(n * fit_fraction)
    
    if rng is None:
        rng = np.random
    
    indices = rng.permutation(n)
    fit_idx = indices[:n_fit]
    eval_idx = indices[n_fit:]
    
    fit_data = [data[i] for i in fit_idx]
    eval_data = [data[i] for i in eval_idx]
    
    return fit_data, eval_data
