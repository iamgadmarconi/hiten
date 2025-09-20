"""Numerical continuation algorithms.

This module provides a comprehensive framework for numerical continuation of solutions in dynamical systems.
"""

from .backends import _ContinuationBackend, _PCContinuationBackend
from .engine import _ContinuationEngine, _OrbitContinuationEngine
from .facades import StateParameter
from .interfaces import _PeriodicOrbitContinuationInterface
from .types import ContinuationResult, _ContinuationProblem

__all__ = [
    "_ContinuationBackend",
    "_PCContinuationBackend",
    "_ContinuationEngine",
    "_OrbitContinuationEngine",
    "_PeriodicOrbitContinuationInterface",
    "ContinuationResult",
    "_ContinuationProblem",
    "StateParameter",
]