from typing import Protocol, runtime_checkable

import numpy as np

from hiten.algorithms.dynamics.base import _DynamicalSystemProtocol
from hiten.algorithms.poincare.core.events import _SurfaceEvent


@runtime_checkable
class _SeedingProtocol(Protocol):
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
        ...

