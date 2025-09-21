"""Protocol for seeding strategies in Poincare return map computation.

This module provides the protocol interface for seeding strategies
that generate initial conditions for Poincare return map computation.
Seeding strategies determine how initial conditions are distributed
on the section plane or in phase space.

Different seeding strategies are appropriate for different dynamical
systems and analysis goals. The protocol provides a flexible interface
that allows various distribution strategies while maintaining a
consistent interface for the return map engine.

This module provides the protocol interface for seeding strategies
that generate initial conditions for Poincare return map computation.
Seeding strategies determine how initial conditions are distributed
on the section plane or in phase space.

Different seeding strategies are appropriate for different dynamical
systems and analysis goals. The protocol provides a flexible interface
that allows various distribution strategies while maintaining a
consistent interface for the return map engine.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from hiten.algorithms.dynamics.protocols import _DynamicalSystemProtocol
from hiten.algorithms.poincare.core.events import _SurfaceEvent


@runtime_checkable
class _SeedingProtocol(Protocol):
    """Protocol for seeding strategies in Poincare return map computation.

    This protocol defines the interface that all seeding strategies must
    implement. Seeding strategies are responsible for generating initial
    conditions whose trajectories will be propagated until they reach
    the Poincare section defined by a surface event.

    The concrete strategy decides how seeds are distributed (axis-aligned
    rays, random clouds, center manifold turning-point logic, etc.) but
    not how they are propagated - that is handled by the return map engine.

    The interface is intentionally minimal so that existing center manifold
    strategies can be adapted with a thin wrapper rather than rewritten.

    Notes
    -----
    Seeding strategies play a crucial role in determining the coverage
    and resolution of the computed return map. Different strategies are
    appropriate for different dynamical systems and analysis goals.

    The protocol supports both problem-agnostic and domain-specific
    seeding strategies through the flexible parameter system.

    All time units are in nondimensional units unless otherwise specified.
    """

    def generate(
        self,
        *,
        dynsys: "_DynamicalSystemProtocol",
        surface: "_SurfaceEvent",
        n_seeds: int,
        **kwargs,
    ) -> "list[np.ndarray]":
        """Generate initial state vectors for return map computation.

        This method generates a list of initial conditions that will be
        propagated until they intersect with the specified Poincare section.
        The distribution strategy is determined by the concrete implementation.

        Parameters
        ----------
        dynsys : :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The dynamical system that will be used for propagation.
            The seeding strategy may use system properties to inform
            the distribution of initial conditions.
        surface : :class:`~hiten.algorithms.poincare.core.events._SurfaceEvent`
            Target Poincare section. The generator may use the section
            definition to align seeds conveniently with the crossing plane
            or to ensure good coverage of the section.
        n_seeds : int
            Desired number of seeds to generate. The actual number returned
            may be fewer if the strategy cannot generate the requested
            number (e.g., due to constraints or available space).
        **kwargs
            Extra implementation-specific parameters. Common examples:
            - energy level for center manifold seeds
            - coordinate ranges for random distributions
            - axis specifications for aligned distributions
            The core engine passes only dynsys, surface, and n_seeds;
            domain-specific wrappers supply additional parameters.

        Returns
        -------
        list[ndarray]
            List of initial state vectors, each with shape (n,) where n
            is the dimension of the state space. The list may contain
            fewer than n_seeds elements if the strategy cannot generate
            the requested number.

        Notes
        -----
        The seeding strategy should generate initial conditions that are
        likely to intersect with the specified section. The distribution
        should provide good coverage of the relevant phase space region
        for the intended analysis.

        Different strategies may use different approaches:
        - Random sampling for statistical coverage
        - Grid-based sampling for systematic coverage
        - Center manifold methods for Hamiltonian systems
        - Axis-aligned distributions for specific coordinate planes

        All state vectors should be in the same coordinate system
        as expected by the dynamical system.
        """
        ...

