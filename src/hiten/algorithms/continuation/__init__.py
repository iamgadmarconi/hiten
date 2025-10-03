"""Numerical continuation algorithms.

This module provides a comprehensive framework for numerical continuation of solutions in dynamical systems.
"""

from .backends import _ContinuationBackend, _PredictorCorrectorContinuationBackend
from .config import ContinuationConfig, OrbitContinuationConfig
from .engine import _ContinuationEngine, _OrbitContinuationEngine
from .base import ContinuationPipeline
from .interfaces import _OrbitContinuationInterface
from .options import ContinuationOptions, OrbitContinuationOptions
from .types import ContinuationResult, _ContinuationProblem

__all__ = [
    # Backends
    "_ContinuationBackend",
    "_PredictorCorrectorContinuationBackend",
    
    # Configs (compile-time structure)
    "ContinuationConfig",
    "OrbitContinuationConfig",
    
    # Options (runtime tuning)
    "ContinuationOptions",
    "OrbitContinuationOptions",
    
    # Interfaces & Engines
    "_ContinuationEngine",
    "_OrbitContinuationEngine",
    "_OrbitContinuationInterface",
    
    # Types & Results
    "ContinuationResult",
    "_ContinuationProblem",
    "ContinuationPipeline",
]