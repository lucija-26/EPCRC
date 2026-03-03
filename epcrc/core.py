# epcrc/core.py
"""
Core abstractions for the EPCRC framework.

These abstract base classes decouple the pruning logic from
specific model implementations (BERT, ResNet, linear models, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Intervention(ABC):
    """
    Abstract intervention T(θ, x) → x̃
    
    Maps an input x and dose θ to a perturbed input.
    Examples: token masking, adversarial perturbation, noise injection.
    """
    
    @abstractmethod
    def apply(self, x: Any, theta: float, seed: int = None) -> Any:
        """
        Apply the intervention.
        
        Args:
            x: Original input (text, image tensor, feature vector, etc.)
            theta: Intervention dose/strength in [0, 1]
            seed: Optional seed for reproducibility
            
        Returns:
            Perturbed input x̃
        """
        pass


class Scalarizer(ABC):
    """
    Abstract scalarizer g(y) → ℝ
    
    Converts high-dimensional model outputs to scalar responses.
    Examples: positive-class probability, margin, confidence score.
    """
    
    @abstractmethod
    def __call__(self, raw_output: Any) -> float:
        """
        Convert model output to scalar.
        
        Args:
            raw_output: Raw model output (logits, embeddings, etc.)
            
        Returns:
            Scalar response Y ∈ ℝ
        """
        pass


class ModelUnit(ABC):
    """
    Abstract model unit for EPCRC framework.
    
    Wraps any input-output system (neural network, linear model, etc.)
    into a unified interface for ecosystem auditing and pruning.
    """
    
    def __init__(self, name: str, scalarizer: Scalarizer = None):
        """
        Args:
            name: Human-readable model identifier
            scalarizer: Optional function to convert outputs to scalars
        """
        self.name = name
        self.scalarizer = scalarizer
    
    @abstractmethod
    def _forward(self, input_data: Any) -> Any:
        """
        Core forward pass (to be implemented by subclasses).
        
        Args:
            input_data: Perturbed input (after intervention)
            
        Returns:
            Model output (raw, before scalarization)
        """
        pass
    
    def query(self, x: Any, theta: float, intervention: Intervention, seed: int = None) -> float:
        """
        EPCRC Core Operation: Query model with intervention.
        
        This is the fundamental operation Y_j(x, θ) = g(f_j(T(θ, x)))
        from the paper (Equation 1).
        
        Args:
            x: Original input
            theta: Intervention dose
            intervention: Intervention object
            seed: Optional seed for deterministic perturbation
            
        Returns:
            Scalarized response Y ∈ ℝ
        """
        # Apply intervention: T(θ, x)
        try:
            perturbed_x = intervention.apply(x, theta, seed)
        except TypeError:
            # Fallback if intervention doesn't accept seed
            perturbed_x = intervention.apply(x, theta)
        
        # Forward pass: f_j(·)
        raw_out = self._forward(perturbed_x)
        
        # Scalarize: g(·)
        if self.scalarizer is not None:
            return self.scalarizer(raw_out)
        return raw_out
    
    def __repr__(self) -> str:
        return f"ModelUnit(name='{self.name}')"


# =============================================================================
# Convenience Identity Classes
# =============================================================================

class IdentityIntervention(Intervention):
    """No-op intervention that returns input unchanged."""
    
    def apply(self, x: Any, theta: float, seed: int = None) -> Any:
        return x


class IdentityScalarizer(Scalarizer):
    """Identity scalarizer for models that already output scalars."""
    
    def __call__(self, raw_output: Any) -> float:
        return float(raw_output)
