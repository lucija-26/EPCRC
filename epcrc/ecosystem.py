# epcrc/ecosystem.py
"""
Ecosystem container for EPCRC.

Manages a collection of models and provides batched querying
with matched interventions (all models see the same perturbed input).
"""

from typing import Any, Iterable, List, Tuple, Optional, Set
import numpy as np

from epcrc.core import ModelUnit, Intervention


class Ecosystem:
    """
    Container for an ecosystem of models.
    
    In EPCRC, we work with a set J = {1, ..., N} of models.
    This class provides:
    - Batched querying with matched interventions
    - Subset selection for pruning experiments
    - Response matrix construction
    """
    
    def __init__(self, models: List[ModelUnit]):
        """
        Args:
            models: List of ModelUnit instances in the ecosystem
        """
        if not models:
            raise ValueError("Ecosystem must contain at least one model")
        
        self.models = models
        self._model_names = [m.name for m in models]
        self._name_to_idx = {m.name: i for i, m in enumerate(models)}
    
    @property
    def n_models(self) -> int:
        """Number of models in ecosystem."""
        return len(self.models)
    
    @property
    def model_names(self) -> List[str]:
        """Names of all models."""
        return self._model_names.copy()
    
    def get_model(self, name: str) -> ModelUnit:
        """Get model by name."""
        idx = self._name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"Model '{name}' not found in ecosystem")
        return self.models[idx]
    
    def get_subset(self, indices: Iterable[int]) -> "Ecosystem":
        """Create a sub-ecosystem from model indices."""
        indices = list(indices)
        return Ecosystem([self.models[i] for i in indices])
    
    def batched_query(
        self,
        X: Iterable[Any],
        Thetas: Any,
        intervention: Intervention,
        seeds: Optional[Iterable[int]] = None,
        model_indices: Optional[Iterable[int]] = None,
    ) -> np.ndarray:
        """
        Query all (or selected) models on a batch of (input, dose) pairs.
        
        Critical: All models receive the SAME perturbed input for each sample.
        This is the matched intervention design from the paper.
        
        Args:
            X: Iterable of raw inputs
            Thetas: Single float (broadcast) or iterable of doses
            intervention: Intervention to apply
            seeds: Optional seeds for reproducibility
            model_indices: If provided, only query these model indices
            
        Returns:
            Response matrix Y, shape (n_samples, n_models)
            Y[i, j] = g(f_j(T(θ_i, x_i)))
        """
        inputs = list(X)
        n_samples = len(inputs)
        
        # Normalize Thetas
        if isinstance(Thetas, (int, float)):
            theta_list = [float(Thetas)] * n_samples
        else:
            theta_list = list(Thetas)
            if len(theta_list) != n_samples:
                raise ValueError(
                    f"Thetas length ({len(theta_list)}) != inputs length ({n_samples})"
                )
        
        # Normalize seeds
        if seeds is None:
            seed_list = [None] * n_samples
        elif isinstance(seeds, int):
            seed_list = [seeds] * n_samples
        else:
            seed_list = list(seeds)
            if len(seed_list) != n_samples:
                raise ValueError(
                    f"Seeds length ({len(seed_list)}) != inputs length ({n_samples})"
                )
        
        # Select models to query
        if model_indices is None:
            models_to_query = self.models
        else:
            model_indices = list(model_indices)
            models_to_query = [self.models[i] for i in model_indices]
        
        n_models = len(models_to_query)
        
        # Query all models
        Y = np.zeros((n_samples, n_models))
        
        for i, (x, theta, seed) in enumerate(zip(inputs, theta_list, seed_list)):
            # Apply intervention ONCE (matched design)
            try:
                perturbed_x = intervention.apply(x, theta, seed)
            except TypeError:
                perturbed_x = intervention.apply(x, theta)
            
            # Query each model with the SAME perturbed input
            for j, model in enumerate(models_to_query):
                raw_out = model._forward(perturbed_x)
                if model.scalarizer is not None:
                    Y[i, j] = model.scalarizer(raw_out)
                else:
                    Y[i, j] = float(raw_out)
        
        return Y
    
    def __repr__(self) -> str:
        return f"Ecosystem(n_models={self.n_models}, models={self._model_names})"
    
    def __len__(self) -> int:
        return self.n_models
    
    def __getitem__(self, idx: int) -> ModelUnit:
        return self.models[idx]
