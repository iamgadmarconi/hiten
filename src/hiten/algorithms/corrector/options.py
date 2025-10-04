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


@dataclass(frozen=True)
class MultipleShootingCorrectionOptions(OrbitCorrectionOptions):
    pass