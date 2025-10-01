"""Numerical continuation algorithms.

This module provides a comprehensive framework for numerical continuation of solutions in dynamical systems.
"""

from .backends import _ContinuationBackend, _PCContinuationBackend
from .config import ContinuationConfig, OrbitContinuationConfig
from .engine import _ContinuationEngine, _OrbitContinuationEngine
from .base import ContinuationPipeline
from .interfaces import _PeriodicOrbitContinuationInterface
from .options import ContinuationOptions, OrbitContinuationOptions
from .types import ContinuationResult, _ContinuationProblem

__all__ = [
    # Backends
    "_ContinuationBackend",
    "_PCContinuationBackend",
    
    # Configs (compile-time structure)
    "ContinuationConfig",
    "OrbitContinuationConfig",
    
    # Options (runtime tuning)
    "ContinuationOptions",
    "OrbitContinuationOptions",
    
    # Interfaces & Engines
    "_ContinuationEngine",
    "_OrbitContinuationEngine",
    "_PeriodicOrbitContinuationInterface",
    
    # Types & Results
    "ContinuationResult",
    "_ContinuationProblem",
    "ContinuationPipeline",
]