from abc import ABC, abstractmethod

import numpy as np

from hiten.algorithms.dynamics.base import _DynamicalSystemProtocol
from hiten.algorithms.poincare.events import _SurfaceEvent
from hiten.algorithms.poincare.seeding.base import _CenterManifoldSeedingBase


class _SeedGenerator(ABC):
    """Problem-agnostic seed generator.

    The role of a *seed generator* is to provide one or more initial states
    whose trajectories will be iterated until they reach the section defined by
    a :class:`_SurfaceEvent`.  The concrete strategy decides *how* those seeds
    are distributed (axis-aligned rays, random cloud, CM turning-point logic, …)
    but *not* how they are propagated - that is handled by the return-map
    engine.

    The interface is intentionally minimal so that existing centre-manifold
    strategies can be *adapted* with a thin wrapper rather than rewritten.
    """

    @abstractmethod
    def generate(
        self,
        *,
        dynsys: "_DynamicalSystemProtocol",
        surface: "_SurfaceEvent",
        n_seeds: int,
        **kwargs,
    ) -> "list[np.ndarray]":
        """Return a list of initial state vectors.

        Parameters
        ----------
        dynsys
            The dynamical system that will be propagated.
        surface
            Target section; a generator may use its definition to align seeds
            conveniently with the crossing plane.
        n_seeds
            Desired number of seeds (generators may return fewer if not
            feasible).
        **kwargs
            Extra implementation-specific parameters (e.g. energy level for CM
            seeds).  The core engine passes only *dynsys*, *surface* and
            *n_seeds*; domain-specific wrappers supply the rest.
        """


class _CMSeedGenerator(_SeedGenerator):
    """Bridge between the generic interface and legacy CM seeding strategies."""

    def __init__(
        self,
        *,
        strategy: _CenterManifoldSeedingBase,
        energy: float,
        H_blocks,
        clmo_table,
        solve_missing_coord_fn,
    ) -> None:
        self._strategy = strategy
        self._h0 = float(energy)
        self._H_blocks = H_blocks
        self._clmo = clmo_table
        self._solve = solve_missing_coord_fn

    def generate(
        self,
        *,
        dynsys: "_DynamicalSystemProtocol",
        surface: "_SurfaceEvent",
        n_seeds: int,
        **kwargs,
    ) -> "list[np.ndarray]":
        if n_seeds != self._strategy.n_seeds:
            self._strategy.n_seeds = n_seeds

        return self._strategy.generate(
            h0=self._h0,
            H_blocks=self._H_blocks,
            clmo_table=self._clmo,
            solve_missing_coord_fn=self._solve,
        )
