"""Linear algebra module public API.

Exposes the backend and helper utilities as well as high-level helpers.
"""

from .base import StabilityPipeline
from .interfaces import _EigenDecompositionInterface, _LibrationPointInterface
from .types import EigenDecompositionResults

__all__ = [
    "StabilityPipeline",
    "_EigenDecompositionInterface",
    "_LibrationPointInterface",
    "EigenDecompositionResults",
]

