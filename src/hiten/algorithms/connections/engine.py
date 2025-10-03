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
    User-facing ConnectionPipeline class that uses this engine.
:mod:`~hiten.algorithms.connections.interfaces`
    Interface classes for manifold data access.
"""

from hiten.algorithms.connections.backends import _ConnectionsBackend
from hiten.algorithms.connections.interfaces import _ManifoldConnectionInterface
from hiten.algorithms.connections.types import (ConnectionResults,
                                                _ConnectionProblem)
from hiten.algorithms.types.core import _HitenBaseEngine


class _ConnectionEngine(_HitenBaseEngine[_ConnectionProblem, ConnectionResults, list]):
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
    :class:`~hiten.algorithms.connections.base.ConnectionPipeline`
        High-level user interface that uses this engine.
    """

    def __init__(self, *, backend: _ConnectionsBackend, interface: _ManifoldConnectionInterface | None = None):
        """Initialize the connection engine with a backend implementation.

        Parameters
        ----------
        backend : :class:`~hiten.algorithms.connections.backends._ConnectionsBackend`
            Backend responsible for the computational steps of connection discovery.
        interface : :class:`~hiten.algorithms.connections.interfaces._ManifoldConnectionInterface`, optional
            Interface for handling connection problems. If None, a default interface is used.
        """
        super().__init__(backend=backend, interface=interface)


    def _invoke_backend(self, call):
        """Invoke the backend with the provided call."""
        return self._backend.run(request=call.request, **call.kwargs)
