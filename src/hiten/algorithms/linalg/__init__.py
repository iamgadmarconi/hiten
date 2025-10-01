"""Linear algebra module public API.

Exposes the backend and helper utilities as well as high-level helpers.
"""

from .base import StabilityPipeline
from .config import EigenDecompositionConfig
from .interfaces import _EigenDecompositionInterface, _LibrationPointInterface
from .options import EigenDecompositionOptions
from .types import EigenDecompositionResults

__all__ = [
    # Main interface
    "StabilityPipeline",
    # Configuration (compile-time structure)
    "EigenDecompositionConfig",
    # Options (runtime tuning)
    "EigenDecompositionOptions",
    # Interfaces
    "_EigenDecompositionInterface",
    "_LibrationPointInterface",
    # Results
    "EigenDecompositionResults",
]

