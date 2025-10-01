"""Runtime options for correction algorithms.

These classes define runtime tuning parameters that control HOW WELL the
correction algorithm runs. They can vary between method calls without changing
the algorithm structure.

For compile-time configuration (algorithm structure), see config.py.
"""

from dataclasses import dataclass

from hiten.algorithms.types.options import (CorrectionOptions,
                                            _HitenBaseOptions)


@dataclass(frozen=True)
class OrbitCorrectionOptions(_HitenBaseOptions):
    """Runtime options for orbit correction algorithms.
    
    These parameters tune HOW WELL the orbit correction algorithm runs and 
    can vary between method calls without changing the algorithm structure.
    
    This extends the base CorrectionOptions with orbit-specific runtime
    parameters like integration direction.
    
    Parameters
    ----------
    base : CorrectionOptions, optional
        Base correction options (convergence, integration, numerical).
    forward : int, default=1
        Integration direction (1 for forward, -1 for backward).
        Can be overridden at runtime.
    
    Notes
    -----
    For algorithm structure parameters like `method`, `finite_difference`,
    `residual_indices`, see OrbitCorrectionConfig instead.
    
    Examples
    --------
    >>> # Default options
    >>> options = OrbitCorrectionOptions()
    >>> 
    >>> # Tighter tolerance
    >>> tight_options = options.merge(
    ...     base=options.base.merge(
    ...         convergence=options.base.convergence.merge(tol=1e-14)
    ...     )
    ... )
    >>> 
    >>> # Backward integration
    >>> backward_options = options.merge(forward=-1)
    """

    base: CorrectionOptions = CorrectionOptions()
    forward: int = 1

    def _validate(self) -> None:
        """Validate the options."""
        if self.forward not in [-1, 1]:
            raise ValueError(
                f"forward must be 1 or -1, got {self.forward}"
            )
        # Nested options validate themselves in __post_init__


@dataclass(frozen=True)
class MultipleShootingOptions(OrbitCorrectionOptions):
    """Runtime options for multiple shooting correction.
    
    Extends OrbitCorrectionOptions with multiple shooting-specific runtime
    parameters.
    
    Parameters
    ----------
    n_patches : int, default=20
        Number of shooting segments. This is a runtime tuning parameter
        because it affects HOW WELL the algorithm performs (robustness,
        convergence basin) rather than WHAT algorithm is used.
        
        More patches generally improve robustness at the cost of increased
        computational cost. Recommended ranges:
        - Lyapunov orbits: 3-5 patches
        - Halo orbits: 4-7 patches
        - Long-period orbits: 7-15 patches
        - Unstable orbits: 5-10 patches
    base : CorrectionOptions, optional
        Base correction options (convergence, integration, numerical).
    forward : int, default=1
        Integration direction.
    
    Notes
    -----
    For algorithm structure parameters like `patch_strategy`, `continuity_indices`,
    see MSOrbitCorrectionConfig instead.
    
    Examples
    --------
    >>> # Default options with 20 patches
    >>> options = MultipleShootingOptions()
    >>> 
    >>> # More patches for robustness
    >>> robust_options = options.merge(n_patches=30)
    >>> 
    >>> # Tighter tolerance with more patches
    >>> tight_options = MultipleShootingOptions(
    ...     n_patches=25,
    ...     base=CorrectionOptions(
    ...         convergence=ConvergenceOptions(tol=1e-14)
    ...     )
    ... )
    """
    n_patches: int = 20

    def _validate(self) -> None:
        """Validate the options."""
        super()._validate()
        if self.n_patches < 2:
            raise ValueError(
                f"n_patches must be at least 2, got {self.n_patches}"
            )

