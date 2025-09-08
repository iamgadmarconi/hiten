"""Provide interface classes for domain-specific continuation algorithms.

This module provides interface classes that adapt the generic continuation
engine to specific problem domains in dynamical systems. These interfaces
implement the abstract methods required by the continuation framework for
particular types of solutions (periodic orbits, invariant tori, etc.).

The interfaces serve as mix-ins that provide domain-specific implementations
of instantiation, correction, and parameter extraction methods, allowing
the generic continuation algorithm to work with different solution types.

All coordinates are in nondimensional CR3BP rotating-frame units.

See Also
--------
:mod:`hiten.algorithms.continuation.base`
    Base continuation engine that these interfaces extend.
:mod:`hiten.system.orbits`
    Periodic orbit classes used by orbit continuation.
:mod:`hiten.algorithms.corrector`
    Correction algorithms used by continuation interfaces.
"""

from typing import Callable, NamedTuple, Sequence

import numpy as np

from hiten.algorithms.utils.types import SynodicState
from hiten.system.orbits.base import PeriodicOrbit


class _OrbitContinuationConfig(NamedTuple):
    """Define configuration parameters for periodic orbit continuation.

    This named tuple encapsulates configuration options specific to
    periodic orbit continuation, including state initialization,
    parameter extraction, and additional correction settings.

    Parameters
    ----------
    state : :class:`hiten.algorithms.utils.types.SynodicState` or None
        Initial state for orbit construction. If None, uses default
        state from the orbit class.
    amplitude : bool, default False
        Whether to use amplitude-based continuation instead of
        natural parameter continuation.
    getter : callable or None
        Function to extract continuation parameter from periodic orbit.
        Should take a :class:`hiten.system.orbits.base.PeriodicOrbit` and return float.
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


class _PeriodicOrbitContinuationInterface:
    """Provide an interface for periodic orbit continuation in the CR3BP.

    This class provides the domain-specific implementation of continuation
    methods for periodic orbits. It serves as a mix-in that implements
    the abstract methods required by the continuation engine for orbit
    families (Lyapunov, halo, vertical Lyapunov, etc.).

    The interface handles orbit instantiation from state vectors, applies
    orbit-specific correction algorithms, and extracts continuation parameters
    from converged orbits.

    Parameters
    ----------
    initial_orbit : :class:`hiten.system.orbits.base.PeriodicOrbit`
        Starting orbit for the continuation family.
    parameter_getter : callable
        Function that extracts continuation parameter from an orbit.
        Should take a :class:`PeriodicOrbit` and return float or ndarray.
    target : sequence
        Target parameter range(s) for continuation. For 1D: (min, max).
        For multi-dimensional: (2, m) array.
    step : float or sequence of float, default 1e-4
        Initial step size(s) for continuation parameters.
    corrector_kwargs : dict, optional
        Additional keyword arguments passed to orbit correction.
    max_orbits : int, default 256
        Maximum number of orbits to generate in the family.
    **kwargs
        Additional arguments passed to the base continuation engine.

    Notes
    -----
    This interface implements the required abstract methods:
    
    - :meth:`hiten.algorithms.continuation.interfaces._PeriodicOrbitContinuationInterface._instantiate`: 
        Create orbit from predicted state vector
    - :meth:`hiten.algorithms.continuation.interfaces._PeriodicOrbitContinuationInterface._correct`: 
        Apply orbit-specific correction algorithm
    - :meth:`hiten.algorithms.continuation.interfaces._PeriodicOrbitContinuationInterface._parameter`: 
        Extract continuation parameter from orbit
    
    The interface preserves the orbit class and libration point from
    the initial orbit, ensuring consistent family generation.

    Examples
    --------
    >>> # This interface is typically used as a mix-in
    >>> class HaloContinuation(_PeriodicOrbitContinuationInterface, hiten.algorithms.continuation.base._ContinuationEngine):
    ...     def _make_stepper(self):
    ...         return NaturalParameterStep()
    ...     
    ...     def _stop_condition(self):
    ...         current = self._parameter(self._family[-1])
    ...         return current >= self._target_max

    See Also
    --------
    :class:`hiten.algorithms.continuation.base._ContinuationEngine`
        Base continuation engine that this interface extends.
    :class:`hiten.system.orbits.base.PeriodicOrbit`
        Orbit class used for continuation.
    """
    def __init__(self, *, initial_orbit: PeriodicOrbit, parameter_getter: Callable[[PeriodicOrbit], "np.ndarray | float"],
        target: Sequence[Sequence[float] | float], step: float | Sequence[float] = 1e-4, corrector_kwargs: dict | None = None,
        max_orbits: int = 256, **kwargs) -> None:

        self._orbit_class = type(initial_orbit)
        self._libration_point = initial_orbit.libration_point

        self._getter = parameter_getter

        super().__init__(
            initial_solution=initial_orbit,
            parameter_getter=parameter_getter,
            target=target,
            step=step,
            corrector_kwargs=corrector_kwargs,
            max_iters=max_orbits,
            **kwargs,
        )

    def _instantiate(self, representation: np.ndarray):
        """Instantiate a periodic orbit from a predicted state vector.

        This method creates a new periodic orbit object from the numerical
        representation produced by the continuation stepping strategy.

        Parameters
        ----------
        representation : ndarray, shape (6,)
            6D state vector [x, y, z, vx, vy, vz] in nondimensional
            CR3BP rotating-frame coordinates.

        Returns
        -------
        :class:`hiten.system.orbits.base.PeriodicOrbit`
            New orbit object with the predicted initial state.
            The orbit class and libration point are preserved from
            the initial orbit.

        Notes
        -----
        The created orbit is not yet corrected and may not satisfy
        periodicity constraints. The :meth:`_correct` method will
        be called subsequently to refine the orbit.
        """
        return self._orbit_class(
            libration_point=self._libration_point,
            initial_state=representation,
        )

    def _correct(self, obj: PeriodicOrbit, **kwargs):
        """Apply orbit-specific correction to satisfy periodicity constraints.

        This method applies the appropriate correction algorithm to refine
        the predicted orbit so that it satisfies the periodicity condition
        and other constraints specific to the orbit type.

        Parameters
        ----------
        obj : :class:`hiten.system.orbits.base.PeriodicOrbit`
            Orbit object to be corrected (from :meth:`_instantiate`).
        **kwargs
            Additional correction parameters, typically including:
            - tol: convergence tolerance
            - max_iter: maximum correction iterations
            - Other orbit-specific parameters

        Returns
        -------
        :class:`hiten.system.orbits.base.PeriodicOrbit`
            The corrected orbit object satisfying periodicity constraints.

        Raises
        ------
        Exception
            If orbit correction fails to converge. The continuation engine
            will catch these exceptions and reduce step size.

        Notes
        -----
        The correction is applied in-place to the orbit object, then
        the same object is returned. This follows the pattern expected
        by the continuation engine.
        """
        obj.correct(**(kwargs or {}))
        return obj

    def _parameter(self, obj: PeriodicOrbit) -> np.ndarray:
        """Extract continuation parameter from a corrected periodic orbit.

        This method extracts the current value of the continuation parameter
        from a corrected orbit object using the parameter getter function
        provided during initialization.

        Parameters
        ----------
        obj : :class:`hiten.system.orbits.base.PeriodicOrbit`
            Corrected periodic orbit object.

        Returns
        -------
        ndarray, shape (n,)
            Current parameter value(s) as a 1D numpy array.
            The continuation engine requires array-like return values.

        Notes
        -----
        The parameter extraction uses the getter function specified
        during initialization. Common parameters include:
        - Energy (Jacobi constant)
        - Amplitude (maximum displacement)
        - Period
        - Stability index
        
        The result is always converted to a numpy array to ensure
        compatibility with the continuation engine.
        """
        return np.asarray(self._getter(obj), dtype=float)
    

class _InvariantToriContinuationInterface:
    """Provide an interface for invariant tori continuation (placeholder).

    This class is reserved for future implementation of continuation
    algorithms for invariant tori in the CR3BP. Invariant tori are
    higher-dimensional invariant sets that arise in quasi-periodic
    motion and provide important structures in phase space.

    Notes
    -----
    This is currently a placeholder class. Future implementation
    will provide methods for:
    
    - Torus instantiation from Fourier representations
    - Torus-specific correction algorithms
    - Parameter extraction from torus objects
    - Continuation along torus families
    
    The interface will follow the same pattern as
    :class:`hiten.algorithms.continuation.interfaces._PeriodicOrbitContinuationInterface` but adapted
    for the higher-dimensional nature of invariant tori.

    See Also
    --------
    :class:`hiten.algorithms.continuation.interfaces._PeriodicOrbitContinuationInterface`
        Similar interface for periodic orbit continuation.
    :mod:`hiten.algorithms.tori`
        Future module for invariant tori algorithms.
    """
    pass