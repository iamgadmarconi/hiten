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

from dataclasses import dataclass
from typing import Literal

from hiten.algorithms.connections.backends import _ConnectionsBackend
from hiten.algorithms.connections.config import _SearchConfig
from hiten.algorithms.connections.interfaces import _ManifoldInterface
from hiten.algorithms.connections.results import _ConnectionResult
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig


@dataclass
class _ConnectionProblem:
    """Define a problem specification for connection discovery between two manifolds.

    This dataclass encapsulates all the parameters needed to define a connection
    discovery problem, including the source and target manifolds, the synodic
    section for intersection, crossing direction, and search configuration.

    Parameters
    ----------
    source : :class:`~hiten.algorithms.connections.interfaces._ManifoldInterface`
        Interface to the source manifold (typically unstable manifold).
    target : :class:`~hiten.algorithms.connections.interfaces._ManifoldInterface`
        Interface to the target manifold (typically stable manifold).
    section : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
        Configuration for the synodic section where manifolds are intersected.
    direction : {1, -1, None}, optional
        Direction for section crossings. 1 for positive crossings, -1 for
        negative crossings, None for both directions.
    search : :class:`~hiten.algorithms.connections.config._SearchConfig`
        Search configuration including tolerances and geometric parameters.

    Notes
    -----
    This class serves as a data container that packages all the necessary
    information for the connection engine to process. It ensures that all
    required parameters are provided and properly typed.

    The problem specification is typically created by the high-level
    :class:`~hiten.algorithms.connections.base.Connection` class and passed
    to the engine for processing.

    Examples
    --------
    >>> from hiten.algorithms.connections.interfaces import _ManifoldInterface
    >>> from hiten.algorithms.connections.config import _SearchConfig
    >>> from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
    >>> 
    >>> source_if = _ManifoldInterface(manifold=unstable_manifold)
    >>> target_if = _ManifoldInterface(manifold=stable_manifold)
    >>> section_cfg = _SynodicMapConfig(x=0.8)
    >>> search_cfg = _SearchConfig(delta_v_tol=1e-3)
    >>> 
    >>> problem = _ConnectionProblem(
    ...     source=source_if,
    ...     target=target_if,
    ...     section=section_cfg,
    ...     direction=1,
    ...     search=search_cfg
    ... )

    See Also
    --------
    :class:`~hiten.algorithms.connections.engine._ConnectionEngine`
        Engine class that processes this problem specification.
    :class:`~hiten.algorithms.connections.base.Connection`
        High-level class that creates these problem specifications.
    """
    source: _ManifoldInterface
    target: _ManifoldInterface
    section: _SynodicMapConfig
    direction: Literal[1, -1, None] | None
    search: _SearchConfig


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
    :class:`~hiten.algorithms.connections.engine._ConnectionProblem`
        Problem specification structure processed by this engine.
    :class:`~hiten.algorithms.connections.backends._ConnectionsBackend`
        Backend algorithms that perform the actual computations.
    :class:`~hiten.algorithms.connections.base.Connection`
        High-level user interface that uses this engine.
    """

    def solve(self, problem: _ConnectionProblem) -> list[_ConnectionResult]:
        """Solve a connection discovery problem.

        This method processes a connection problem specification and returns
        a list of discovered connections between the source and target manifolds.

        Parameters
        ----------
        problem : :class:`~hiten.algorithms.connections.engine._ConnectionProblem`
            Complete problem specification including source/target manifolds,
            synodic section configuration, crossing direction, and search parameters.

        Returns
        -------
        list of :class:`~hiten.algorithms.connections.results._ConnectionResult`
            Discovered connections sorted by increasing Delta-V requirement.
            Each result contains transfer type (ballistic/impulsive), Delta-V,
            intersection points, and 6D states at the connection.

        Notes
        -----
        This method delegates the computational work to the backend algorithms
        while maintaining a clean separation between the orchestration logic
        and the numerical computations.

        The backend handles:
        - Manifold intersection with synodic sections
        - Geometric pairing and filtering
        - Delta-V computation and classification

        Examples
        --------
        >>> engine = _ConnectionEngine()
        >>> results = engine.solve(problem)
        >>> for result in results:
        ...     print(f"Delta-V: {result.delta_v:.6f}, Type: {result.kind}")

        See Also
        --------
        :class:`~hiten.algorithms.connections.backends._ConnectionsBackend`
            Backend class that performs the actual computations.
        """
        # Delegate to backend for matching/refinement/Delta-V computation
        backend = _ConnectionsBackend()
        return backend.solve(problem)