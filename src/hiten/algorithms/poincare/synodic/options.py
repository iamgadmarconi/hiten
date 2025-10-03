"""Runtime options for synodic Poincare maps.

This module re-exports common options from poincare.core and adds
synodic-specific options.

For compile-time configuration (section geometry), see config.py.
"""

from dataclasses import dataclass

from hiten.algorithms.types.options import (_HitenBaseOptions, RefineOptions,
                                            WorkerOptions)


@dataclass(frozen=True)
class SynodicMapOptions(_HitenBaseOptions):
    """Runtime options for synodic Poincare map detection and refinement.
    
    Composes common runtime options for tuning synodic map detection performance.
    
    Parameters
    ----------
    refine : RefineOptions, optional
        Refinement tuning parameters (segment_refine, tolerances,
        max_hits_per_traj, newton_max_iter).
    workers : WorkerOptions, optional
        Parallel computation (n_workers).

    Notes
    -----
    For compile-time structure parameters like `section_axis`, `section_offset`,
    `section_normal`, `plane_coords`, `direction`, and `interp_kind`, 
    use SynodicMapConfig instead.

    Examples
    --------
    >>> # Default runtime options
    >>> options = SynodicMapOptions()
    >>> 
    >>> # Dense refinement with more Newton iterations
    >>> precise_options = SynodicMapOptions(
    ...     refine=RefineOptions(
    ...         segment_refine=10,
    ...         newton_max_iter=10
    ...     )
    ... )
    >>> 
    >>> # Parallel processing
    >>> parallel_options = options.merge(
    ...     workers=WorkerOptions(n_workers=8)
    ... )

    See Also
    --------
    :class:`~hiten.algorithms.poincare.synodic.config.SynodicMapConfig`
        Compile-time configuration for problem structure.
    """
    _version: float = 1.0

    refine: RefineOptions = RefineOptions()
    workers: WorkerOptions = WorkerOptions()

    def _validate(self) -> None:
        """Validate the options."""
        # Nested options validate themselves in __post_init__
        pass

