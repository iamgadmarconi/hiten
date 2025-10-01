"""Runtime options for center manifold Poincare maps.

This module re-exports common options from poincare.core and adds
centermanifold-specific options.

For compile-time configuration (section structure, seeding strategy), see config.py.
"""

from dataclasses import dataclass

from hiten.algorithms.poincare.core.options import (IterationOptions,
                                                    SeedingOptions)
from hiten.algorithms.types.options import (_HitenBaseOptions,
                                            IntegrationOptions,
                                            WorkerOptions)


@dataclass(frozen=True)
class CenterManifoldMapOptions(_HitenBaseOptions):
    """Runtime options for center manifold Poincare map computation.
    
    Composes common runtime options for tuning center manifold map performance.
    
    Parameters
    ----------
    integration : :class:`~hiten.algorithms.types.options.IntegrationOptions`, optional
        Integration tuning parameters (dt, order, max_steps, c_omega_heuristic).
    iteration : :class:`~hiten.algorithms.poincare.core.options.IterationOptions`, optional
        Iteration count (n_iter).
    seeding : :class:`~hiten.algorithms.poincare.core.options.SeedingOptions`, optional
        Number of seeds (n_seeds).
    workers : :class:`~hiten.algorithms.types.options.WorkerOptions`, optional
        Parallel computation (n_workers).

    Notes
    -----
    For compile-time structure parameters like `seed_strategy`, `seed_axis`,
    and `section_coord`, use CenterManifoldMapConfig instead.

    Examples
    --------
    >>> # Default runtime options
    >>> options = CenterManifoldMapOptions()
    >>> 
    >>> # More iterations and seeds
    >>> dense_options = CenterManifoldMapOptions(
    ...     iteration=IterationOptions(n_iter=100),
    ...     seeding=SeedingOptions(n_seeds=50)
    ... )
    >>> 
    >>> # Tighter integration
    >>> precise_options = options.merge(
    ...     integration=:class:`~hiten.algorithms.types.options.IntegrationOptions`(dt=1e-3, order=8)
    ... )

    See Also
    --------
    :class:`~hiten.algorithms.poincare.centermanifold.config.CenterManifoldMapConfig`
        Compile-time configuration for problem structure.
    """
    integration: IntegrationOptions = IntegrationOptions()
    iteration: IterationOptions = IterationOptions()
    seeding: SeedingOptions = SeedingOptions()
    workers: WorkerOptions = WorkerOptions()

    def _validate(self) -> None:
        """Validate the options."""
        # Nested options validate themselves in __post_init__
        pass

