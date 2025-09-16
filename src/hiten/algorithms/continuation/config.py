"""Provide configuration classes for domain-specific continuation algorithms.

This module provides configuration classes for domain-specific continuation
algorithms. These classes encapsulate the parameters required for different
types of continuation methods (natural parameter, pseudo-arclength, etc.).
"""

from typing import Callable, NamedTuple

from hiten.algorithms.utils.types import SynodicState
from hiten.system.orbits.base import PeriodicOrbit


class _OrbitContinuationConfig(NamedTuple):
    """Define configuration parameters for periodic orbit continuation.

    This named tuple encapsulates configuration options specific to
    periodic orbit continuation, including state initialization,
    parameter extraction, and additional correction settings.

    Parameters
    ----------
    state : :class:`~hiten.algorithms.utils.types.SynodicState` or None
        Initial state for orbit construction. If None, uses default
        state from the orbit class.
    amplitude : bool, default False
        Whether to use amplitude-based continuation instead of
        natural parameter continuation.
    getter : callable or None
        Function to extract continuation parameter from periodic orbit.
        Should take a :class:`~hiten.system.orbits.base.PeriodicOrbit` and return float.
        If None, uses default parameter extraction.
    extra_params : dict or None
        Additional parameters passed to orbit correction methods.
        Common keys include tolerances, maximum iterations, etc.

    Notes
    -----
    This configuration is used to customize the behavior of orbit
    continuation without modifying the core continuation algorithm.
    It provides a clean way to specify domain-specific options.

    Examples
    --------
    >>> config = _OrbitContinuationConfig(
    ...     state=None,
    ...     amplitude=True,
    ...     getter=lambda orbit: orbit.energy,
    ...     extra_params={'tol': 1e-12, 'max_iter': 50}
    ... )
    """
    state: SynodicState | None
    amplitude: bool = False
    getter: Callable[["PeriodicOrbit"], float] | None = None
    extra_params: dict | None = None
