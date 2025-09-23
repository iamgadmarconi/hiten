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

from typing import Callable

from hiten.algorithms.connections.backends import _ConnectionsBackend
from hiten.algorithms.connections.interfaces import _ManifoldInterface
from hiten.algorithms.connections.types import (ConnectionResults,
                                                _ConnectionProblem)
from hiten.algorithms.utils.exceptions import EngineError
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

    def solve(self, problem: _ConnectionProblem) -> ConnectionResults:
        """Solve a connection discovery problem from a composed Problem.

        Parameters
        ----------
        problem : :class:`~hiten.algorithms.connections.types._ConnectionProblem`
            The problem to solve.

        Returns
        -------
        :class:`~hiten.algorithms.connections.types.ConnectionResults`
            Engine-level result containing the backend connection records.
        """

        src_if = self._interface_factory(problem.source)
        tgt_if = self._interface_factory(problem.target)

        try:
            self._backend.on_start(problem)
        except Exception:
            pass

        try:
            pu, Xu = src_if.to_numeric(problem.section, direction=problem.direction)
            ps, Xs = tgt_if.to_numeric(problem.section, direction=problem.direction)

            eps = float(getattr(problem.search, "eps2d", 1e-4)) if problem.search else 1e-4
            dv_tol = float(getattr(problem.search, "delta_v_tol", 1e-3)) if problem.search else 1e-3
            bal_tol = float(getattr(problem.search, "ballistic_tol", 1e-8)) if problem.search else 1e-8

            records = self._backend.solve(
                pu,
                ps,
                Xu,
                Xs,
                eps=eps,
                dv_tol=dv_tol,
                bal_tol=bal_tol,
            )
        except Exception as exc:
            try:
                self._backend.on_failure(exc)
            except Exception:
                pass
            raise EngineError("Connection engine failed") from exc

        try:
            self._backend.on_success(records)
        except Exception:
            pass

        return ConnectionResults(records)
