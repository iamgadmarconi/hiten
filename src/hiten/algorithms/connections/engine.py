"""Provide a connection engine for orchestrating manifold transfer discovery in CR3BP.

This module provides the core engine that coordinates the connection discovery
process between manifolds in the Circular Restricted Three-Body Problem (CR3BP).
It defines the problem specification structure and orchestrates the backend
computational algorithms.

The engine serves as the main entry point for the connection discovery pipeline,
handling problem setup and delegating the computational work to specialized
backend algorithms.

All coordinates are in nondimensional CR3BP rotating-frame units.

See Also
--------
:mod:`~hiten.algorithms.connections.backends`
    Backend algorithms for connection computation.
:mod:`~hiten.algorithms.connections.base`
    User-facing Connection class that uses this engine.
:mod:`~hiten.algorithms.connections.interfaces`
    Interface classes for manifold data access.
"""

from typing import Callable, Literal

import numpy as np

from hiten.algorithms.connections.backends import _ConnectionsBackend
from hiten.algorithms.connections.config import _SearchConfig
from hiten.algorithms.connections.interfaces import _ManifoldInterface
from hiten.algorithms.connections.types import (_ConnectionProblem,
                                                _ConnectionResult)
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.system.manifold import Manifold


class _ConnectionEngine:
    """Provide the main engine for orchestrating connection discovery between manifolds.

    This class serves as the central coordinator for the connection discovery
    process. It takes a problem specification and orchestrates the various
    computational steps needed to find ballistic and impulsive transfers
    between manifolds.

    The engine delegates the actual computational work to specialized backend
    algorithms while maintaining a clean interface for the higher-level
    connection discovery system.

    Notes
    -----
    The connection discovery process involves:
    1. Intersecting both manifolds with the specified synodic section
    2. Finding geometrically close points between intersection sets
    3. Applying mutual-nearest-neighbor filtering
    4. Refining matches using local segment geometry
    5. Computing Delta-V requirements and classifying transfers

    This engine coordinates these steps and ensures proper data flow
    between the different algorithmic components.

    Examples
    --------
    >>> engine = _ConnectionEngine()
    >>> results = engine.solve(problem)
    >>> print(f"Found {len(results)} connections")

    See Also
    --------
    :class:`~hiten.algorithms.connections.types._ConnectionProblem`
        Problem specification structure processed by this engine.
    :class:`~hiten.algorithms.connections.backends._ConnectionsBackend`
        Backend algorithms that perform the actual computations.
    :class:`~hiten.algorithms.connections.base.Connection`
        High-level user interface that uses this engine.
    """

    def __init__(self, backend: _ConnectionsBackend, *, interface_factory: Callable[[Manifold], _ManifoldInterface] | None = None):
        """Initialize the connection engine with a backend implementation.

        Parameters
        ----------
        backend : :class:`~hiten.algorithms.connections.backends._ConnectionsBackend`
            Backend responsible for the computational steps of connection discovery.
        """
        self._backend = backend
        self._interface_factory = interface_factory or (lambda m: _ManifoldInterface(manifold=m))

    def solve(self, source: Manifold, target: Manifold, section: "_SynodicMapConfig", direction: Literal[1, -1, None] | None, search: _SearchConfig | None) -> list[_ConnectionResult]:
        """Solve a connection discovery problem (assemble problem internally).

        This method assembles a problem specification internally for clarity and
        orchestrates the connection discovery workflow, returning discovered
        connections between the source and target manifolds.

        Parameters
        ----------
        source : :class:`~hiten.system.manifold.Manifold`
            Source manifold (typically unstable).
        target : :class:`~hiten.system.manifold.Manifold`
            Target manifold (typically stable).
        section : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
            Synodic section configuration.
        direction : {1, -1, None}
            Crossing direction filter.
        search : :class:`~hiten.algorithms.connections.config._SearchConfig` or None
            Tolerances and geometric parameters.

        Returns
        -------
        list[:class:`~hiten.algorithms.connections.types._ConnectionResult`]
            Discovered connections sorted by increasing Delta-V requirement.
        """
        # Assemble a problem object for traceability (not passed to backend)
        problem = _ConnectionProblem(
            source=source,
            target=target,
            section=section,
            direction=direction,
            search=search,
        )

        # Build numerical inputs via interfaces
        src_if = self._interface_factory(problem.source)
        tgt_if = self._interface_factory(problem.target)
        sec_u = src_if.to_section(problem.section, direction=problem.direction)
        sec_s = tgt_if.to_section(problem.section, direction=problem.direction)

        pu = np.asarray(sec_u.points, dtype=float)
        ps = np.asarray(sec_s.points, dtype=float)
        Xu = np.asarray(sec_u.states, dtype=float)
        Xs = np.asarray(sec_s.states, dtype=float)

        eps = float(getattr(problem.search, "eps2d", 1e-4)) if problem.search else 1e-4
        dv_tol = float(getattr(problem.search, "delta_v_tol", 1e-3)) if problem.search else 1e-3
        bal_tol = float(getattr(problem.search, "ballistic_tol", 1e-8)) if problem.search else 1e-8

        return self._backend.solve(pu, ps, Xu, Xs, eps=eps, dv_tol=dv_tol, bal_tol=bal_tol)