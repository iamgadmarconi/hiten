"""Types and dataclasses for the linear algebra module.

Defines the types and dataclasses used throughout the linear algebra module.
"""


from dataclasses import dataclass
from enum import Enum

import numpy as np


class _ProblemType(Enum):
    """Problem type for eigenvalue decomposition.
    
    Parameters
    ----------
    EIGENVALUE_DECOMPOSITION : int
        Eigenvalue decomposition problem.
    STABILITY_INDICES : int
        Stability indices problem.
    ALL : int
        All problems.
    """
    EIGENVALUE_DECOMPOSITION = 0
    STABILITY_INDICES = 1
    ALL = 2


class _SystemType(Enum):
    """System type for eigenvalue decomposition.

    Parameters
    ----------
    CONTINUOUS : int
        Continuous-time system.
    DISCRETE : int
        Discrete-time system.
    """
    CONTINUOUS = 0
    DISCRETE = 1


class _StabilityType(Enum):
    """
    Stability type for eigenvalue decomposition.

    Parameters
    ----------
    CENTER : int
        Center eigenvalue.
    STABLE : int
        Stable eigenvalue.
    UNSTABLE : int
        Unstable eigenvalue.
    """
    CENTER = 0
    STABLE = 1
    UNSTABLE = 2

    def __str__(self) -> str:
        return self.name.lower().capitalize()


@dataclass
class EigenDecompositionResults:
    """Stable/unstable/center spectra and bases.
    
    Parameters
    ----------
    stable : np.ndarray
        Stable eigenvalues.
    unstable : np.ndarray
        Unstable eigenvalues.
    center : np.ndarray
        Center eigenvalues.
    Ws : np.ndarray
        Stable eigenvectors.
    Wu : np.ndarray
        Unstable eigenvectors.
    Wc : np.ndarray
        Center eigenvectors.
    nu : np.ndarray
        Floquet stability indices.
    eigvals : np.ndarray
        Eigenvalues.
    eigvecs : np.ndarray
        Eigenvectors.
    """
    stable: np.ndarray
    unstable: np.ndarray
    center: np.ndarray
    Ws: np.ndarray
    Wu: np.ndarray
    Wc: np.ndarray
    nu: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray


@dataclass(frozen=True)
class _EigenDecompositionProblem:
    """Problem definition for classifying the eigen-structure of a matrix.
    
    Parameters
    ----------
    A : np.ndarray
        Matrix to decompose.
    problem_type : :class:`~hiten.algorithms.linalg.types._ProblemType`
        Problem type for the eigenvalue decomposition.
    system_type : :class:`~hiten.algorithms.linalg.types._SystemType`
        Classification mode: 0 for continuous-time (sign of real parts),
        1 for discrete-time (modulus relative to unity).
    delta : float
        Tolerance used in eigenvalue classification.
    tol : float
        Tolerance used in stability index calculation.
    """

    A: np.ndarray
    problem_type: "_ProblemType"
    system_type: "_SystemType"
    delta: float
    tol: float


@dataclass(frozen=True)
class _LibrationPointStabilityProblem(_EigenDecompositionProblem):
    """Problem definition for computing linear stability at a CR3BP position.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP.
    position : np.ndarray
        Position in the CR3BP.
    """

    mu: float
    position: np.ndarray
