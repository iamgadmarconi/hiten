"""Configuration and problem dataclasses for the linear algebra module."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hiten.algorithms.linalg.types import _ProblemType, _SystemType


@dataclass(frozen=True)
class _EigenDecompositionConfig:
    """Configuration for eigenvalue decomposition classification.
    
    Parameters
    ----------
    problem_type : :class:`~hiten.algorithms.linalg.types._ProblemType`
        Problem type for the eigenvalue decomposition.
    system_type : :class:`~hiten.algorithms.linalg.types._SystemType`
        Classification mode: 0 for continuous-time (sign of real parts),
        1 for discrete-time (modulus relative to unity).
    delta : float, default=1e-6
        Tolerance used in eigenvalue classification.
    tol: float, default=1e-8
        Tolerance used in stability index calculation.
    """
    problem_type: "_ProblemType"
    system_type: "_SystemType"
    delta: float = 1e-6
    tol: float = 1e-8


