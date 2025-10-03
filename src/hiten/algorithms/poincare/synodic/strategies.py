"""Seeding strategies for synodic Poincare sections.

This module provides seeding strategy classes for synodic Poincare section
detection. Since synodic maps operate on precomputed trajectories rather
than generating initial conditions, the strategies are minimal and serve
as placeholders to satisfy the engine interface.

The main class :class:`~hiten.algorithms.poincare.synodic.strategies._NoOpStrategy` 
implements a no-operation seeding strategy that raises NotImplementedError when called, 
since synodic maps do not require seed generation.

The implementation provides a minimal interface that satisfies the
engine requirements while clearly indicating that seed generation
is not applicable for synodic maps.
"""

from hiten.algorithms.poincare.core.strategies import _SeedingStrategyBase


class _NoOpStrategy(_SeedingStrategyBase):
    """No-operation seeding strategy for synodic Poincare maps.

    This class implements a no-operation seeding strategy that serves as
    a placeholder to satisfy the engine interface requirements. Since
    synodic Poincare maps operate on precomputed trajectories rather
    than generating initial conditions, this strategy raises NotImplementedError
    when called.

    Parameters
    ----------
    map_cfg : :class:`~hiten.algorithms.poincare.synodic.config.SynodicMapConfig`
        The map configuration (unused).

    Notes
    -----
    This strategy is required by the engine interface but is not used
    in practice. Synodic Poincare maps receive precomputed trajectories
    through the engine's set_trajectories method rather than generating
    initial conditions from seeds.

    The strategy clearly indicates through NotImplementedError that
    seed generation is not applicable for synodic maps, providing
    a clear error message if the method is accidentally called.

    All time units are in nondimensional units unless otherwise specified.
    """

    def generate(self, *, h0, H_blocks, clmo_table, solve_missing_coord_fn, find_turning_fn):
        """Generate seeds (not implemented for synodic maps).

        Parameters
        ----------
        h0 : float
            Initial energy value (unused).
        H_blocks : array_like
            Energy blocks (unused).
        clmo_table : array_like
            CLMO table (unused).
        solve_missing_coord_fn : callable
            Function to solve missing coordinates (unused).
        find_turning_fn : callable
            Function to find turning points (unused).

        Raises
        ------
        NotImplementedError
            Always raised since synodic maps do not generate seeds.

        Notes
        -----
        This method is required by the base seeding strategy interface
        but is not used for synodic maps. It raises NotImplementedError
        to clearly indicate that seed generation is not applicable.

        Synodic maps operate on precomputed trajectories rather than
        generating initial conditions from seeds.
        """
        raise NotImplementedError("Synodic engine does not generate seeds")
