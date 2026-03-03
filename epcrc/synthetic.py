# epcrc/synthetic.py
"""
Synthetic models for testing and validation.

These simple models help verify the framework before
running expensive real-world experiments.
"""

import numpy as np
from typing import Optional

from epcrc.core import ModelUnit, Intervention, Scalarizer


class LinearModel(ModelUnit):
    """
    Linear structural model: Y = βᵀx + noise
    
    This is the local linear model from the theory sections.
    Useful for:
    - Verifying pruning algorithms
    - Testing saturation behavior
    - Controlled experiments with known ground truth
    """
    
    def __init__(
        self,
        dim: int,
        beta: Optional[np.ndarray] = None,
        noise_std: float = 0.0,
        name: str = "LinearModel",
    ):
        """
        Args:
            dim: Input dimension
            beta: Coefficient vector (random if None)
            noise_std: Standard deviation of output noise
            name: Model identifier
        """
        super().__init__(name=name)
        self.dim = dim
        
        if beta is None:
            beta = np.random.randn(dim)
            beta /= np.linalg.norm(beta)  # Unit sphere
        
        self.beta = np.asarray(beta).flatten()
        self.noise_std = noise_std
    
    def _forward(self, input_vec: np.ndarray) -> float:
        """
        Compute Y = βᵀx + ε
        
        Args:
            input_vec: Input vector x, shape (dim,)
            
        Returns:
            Scalar output Y
        """
        x = np.asarray(input_vec).flatten()
        y = float(np.dot(self.beta, x))
        
        if self.noise_std > 0:
            y += np.random.randn() * self.noise_std
        
        return y


class ScalingIntervention(Intervention):
    """
    Simple scaling intervention: T(θ, x) = θ * x
    
    Useful for testing dose-response behavior.
    """
    
    def apply(self, x: np.ndarray, theta: float, seed: int = None) -> np.ndarray:
        return theta * np.asarray(x)


class NoiseIntervention(Intervention):
    """
    Additive Gaussian noise intervention: T(θ, x) = x + θ * ε
    
    Args:
        theta: Noise scale
        seed: For reproducibility
    """
    
    def apply(self, x: np.ndarray, theta: float, seed: int = None) -> np.ndarray:
        x = np.asarray(x)
        
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        
        noise = rng.randn(*x.shape) * theta
        return x + noise


# =============================================================================
# Ecosystem generation utilities
# =============================================================================

def generate_linear_ecosystem(
    n_models: int,
    dim: int,
    noise_std: float = 0.0,
    seed: int = None,
) -> list:
    """
    Generate a random linear ecosystem.
    
    Args:
        n_models: Number of models N
        dim: Input dimension d
        noise_std: Output noise level
        seed: Random seed
        
    Returns:
        List of LinearModel instances
    """
    if seed is not None:
        np.random.seed(seed)
    
    models = []
    for i in range(n_models):
        beta = np.random.randn(dim)
        beta /= np.linalg.norm(beta)
        models.append(LinearModel(dim, beta, noise_std, name=f"model_{i}"))
    
    return models


def generate_redundant_ecosystem(
    n_unique: int,
    n_redundant: int,
    dim: int,
    noise_std: float = 0.0,
    redundancy_noise: float = 0.01,
    seed: int = None,
) -> list:
    """
    Generate an ecosystem with known redundancy structure.
    
    Creates n_unique independent models, then adds n_redundant models
    that are convex combinations of the unique ones.
    
    Useful for testing pruning algorithms - we know the ground truth.
    
    Args:
        n_unique: Number of unique (non-redundant) models
        n_redundant: Number of redundant models
        dim: Input dimension
        noise_std: Model output noise
        redundancy_noise: Perturbation to redundant models
        seed: Random seed
        
    Returns:
        List of models (unique ones first, then redundant)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate unique models
    unique_betas = []
    models = []
    
    for i in range(n_unique):
        beta = np.random.randn(dim)
        beta /= np.linalg.norm(beta)
        unique_betas.append(beta)
        models.append(LinearModel(dim, beta, noise_std, name=f"unique_{i}"))
    
    # Generate redundant models as convex combinations
    for i in range(n_redundant):
        # Random simplex weights
        w = np.random.dirichlet(np.ones(n_unique))
        
        # Convex combination of unique betas
        beta = sum(w[j] * unique_betas[j] for j in range(n_unique))
        
        # Add small perturbation
        if redundancy_noise > 0:
            beta += np.random.randn(dim) * redundancy_noise
            beta /= np.linalg.norm(beta)
        
        models.append(LinearModel(dim, beta, noise_std, name=f"redundant_{i}"))
    
    return models
