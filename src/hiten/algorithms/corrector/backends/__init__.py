"""Provide backends for iterative correction algorithms.

This module provides the backends for iterative correction algorithms.

The main class :class:`~hiten.algorithms.corrector.backends.base._CorrectorBackend` 
defines the interface that all concrete backends must implement, including the 
core `correct` method and common functionality for handling nonlinear systems.

See Also
--------
:mod:`~hiten.algorithms.corrector.backends.newton`
    Newton-Raphson backend implementation.
:mod:`~hiten.algorithms.corrector.interfaces`
    Interface classes for different correction strategies.
:mod:`~hiten.algorithms.corrector.stepping`
    Step-size control interfaces for robust convergence.
"""

from .base import _CorrectorBackend
from .newton import _NewtonBackend

__all__ = [
    "_CorrectorBackend",
    "_NewtonBackend",
]