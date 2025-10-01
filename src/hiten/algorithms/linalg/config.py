"""Configuration classes for linear algebra algorithms (compile-time structure).

This module provides configuration classes that define the algorithm structure
for linear algebra operations. These parameters define WHAT problem is being
solved and should be set once when creating a pipeline.

For runtime tuning parameters (tolerances), see options.py.
"""

from dataclasses import dataclass

from hiten.algorithms.types.configs import _HitenBaseConfig
from hiten.algorithms.linalg.types import _ProblemType, _SystemType


@dataclass(frozen=True)
class EigenDecompositionConfig(_HitenBaseConfig):
    """Configuration for eigenvalue decomposition (compile-time structure).

    This dataclass encapsulates compile-time configuration parameters that
    define the problem structure for eigenvalue decomposition. These parameters
    define WHAT problem to solve, not HOW WELL to solve it.

    Parameters
    ----------
    problem_type : :class:`~hiten.algorithms.linalg.types._ProblemType`
        Defines WHAT to compute:
        
        - EIGENVALUE_DECOMPOSITION: Classify eigenvalues into stable/unstable/center
        - STABILITY_INDICES: Compute Floquet stability indices nu_i
        - ALL: Compute both decomposition and indices
        
        This is compile-time because it fundamentally changes WHAT outputs
        are produced by the algorithm.
    system_type : :class:`~hiten.algorithms.linalg.types._SystemType`
        Defines WHAT kind of dynamical system:
        
        - CONTINUOUS: Continuous-time systems (classify by sign of real part)
        - DISCRETE: Discrete-time systems (classify by magnitude relative to 1)
        
        This is compile-time because it fundamentally changes HOW eigenvalues
        are interpreted and classified.

    Notes
    -----
    For runtime tuning parameters like `delta` (classification tolerance) and
    `tol` (stability index tolerance), use EigenDecompositionOptions instead.

    The `problem_type` is compile-time because it defines WHAT the algorithm
    computes (structure of the output). The `system_type` is compile-time
    because it defines WHAT kind of system is being analyzed (continuous vs
    discrete interpretation of eigenvalues).

    Examples
    --------
    >>> from hiten.algorithms.linalg.types import (_ProblemType, _SystemType)
    >>> 
    >>> # Compile-time: Define problem structure
    >>> config = EigenDecompositionConfig(
    ...     problem_type=_ProblemType.ALL,
    ...     system_type=_SystemType.CONTINUOUS
    ... )
    >>> 
    >>> # Runtime: Set tolerance parameters
    >>> from hiten.algorithms.linalg.options import EigenDecompositionOptions
    >>> options = EigenDecompositionOptions(
    ...     delta=1e-6,
    ...     tol=1e-8
    ... )

    See Also
    --------
    :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`
        Runtime tuning parameters for eigenvalue decomposition.
    :class:`~hiten.algorithms.linalg.base.StabilityPipeline`
        Main class that uses this configuration.
    """
    problem_type: _ProblemType = _ProblemType.EIGENVALUE_DECOMPOSITION
    system_type: _SystemType = _SystemType.CONTINUOUS

    def _validate(self) -> None:
        """Validate the configuration."""
        if not isinstance(self.problem_type, _ProblemType):
            raise ValueError(
                f"Invalid problem_type: {self.problem_type}. "
                f"Must be a _ProblemType enum value."
            )
        if not isinstance(self.system_type, _SystemType):
            raise ValueError(
                f"Invalid system_type: {self.system_type}. "
                f"Must be a _SystemType enum value."
            )
