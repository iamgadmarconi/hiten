r"""Provide generic right-hand side function adapters for dynamical systems.

This module provides lightweight adapters that convert arbitrary Python callables
representing ODEs into objects compatible with the dynamical systems framework.
The adapters handle automatic JIT compilation for performance optimization.

The primary use case is wrapping user-defined ODE functions dy/dt = f(t, y)
into the standardized dynamical system interface, enabling them to work
seamlessly with various numerical integrators.
"""

from typing import Callable

import numpy as np

from hiten.algorithms.dynamics.base import _DynamicalSystem


class _RHSSystem(_DynamicalSystem):
    r"""Provide an adapter for generic right-hand side functions.

    Converts arbitrary Python callables representing ODE systems dy/dt = f(t, y)
    into objects compatible with the dynamical systems framework. Automatically
    handles JIT compilation for optimal performance in numerical integrators.

    Parameters
    ----------
    rhs_func : Callable[[float, ndarray], ndarray]
        Function implementing the ODE system dy/dt = f(t, y).
        Must accept time t (float) and state y (1D ndarray) and return
        the time derivative as a 1D ndarray of the same shape.
    dim : int
        Dimension of the state space (length of state vector).
    name : str, optional
        Human-readable system identifier. Default is "Generic RHS".

    Attributes
    ----------
    dim : int
        State space dimension (inherited from base class).
    name : str
        System identifier string.
    rhs : Callable[[float, ndarray], ndarray]
        JIT-compiled RHS function compatible with Numba nopython mode.

    Raises
    ------
    ValueError
        If dim is not positive (inherited from base class).

    Notes
    -----
    - Uses centralized JIT compilation utility for optimal performance
    - Automatically detects pre-compiled Numba dispatchers and reuses them
    - Compiles plain Python functions with Numba JIT for performance
    - Uses global fast-math setting for numerical optimization
    - Compatible with all integrators that accept _DynamicalSystem objects
    
    Examples
    --------
    >>> import numpy as np
    >>> def harmonic_oscillator(t, y):
    ...     return np.array([y[1], -y[0]])  # dy/dt = [v, -x]
    >>> sys = _RHSSystem(harmonic_oscillator, dim=2, name="Harmonic Oscillator")
    >>> derivative = sys.rhs(0.0, np.array([1.0, 0.0]))
    
    See Also
    --------
    :class:`~hiten.algorithms.dynamics.base._DynamicalSystem` : Base class
    :meth:`~hiten.algorithms.dynamics.base._DynamicalSystem._compile_rhs_function` :
        JIT compilation method
    :func:`~hiten.algorithms.dynamics.rhs.create_rhs_system` : Factory function
    """

    def __init__(self, rhs_func: Callable[[float, np.ndarray], np.ndarray], dim: int, name: str = "Generic RHS"):
        super().__init__(dim)

        # Store plain implementation; compilation handled by base
        self._rhs_impl = rhs_func
        self.name = name
    
    def _build_rhs_impl(self) -> Callable[[float, np.ndarray], np.ndarray]:
        return self._rhs_impl
    
    def __repr__(self) -> str:
        return f"_RHSSystem(name='{self.name}', dim={self.dim})"


def create_rhs_system(rhs_func: Callable[[float, np.ndarray], np.ndarray], dim: int, name: str = "Generic RHS"):
    r"""Create RHS system using functional interface.

    Factory function that provides a functional alternative to the
    object-oriented _RHSSystem constructor. Useful for creating systems
    in a more concise, functional programming style.

    Parameters
    ----------
    rhs_func : Callable[[float, ndarray], ndarray]
        Right-hand side function implementing dy/dt = f(t, y).
    dim : int
        State space dimension.
    name : str, optional
        System identifier. Default is "Generic RHS".

    Returns
    -------
    _RHSSystem
        Configured RHS system ready for integration.
        
    Examples
    --------
    >>> import numpy as np
    >>> def pendulum(t, y):
    ...     theta, omega = y
    ...     return np.array([omega, -np.sin(theta)])
    >>> sys = create_rhs_system(pendulum, dim=2, name="Nonlinear Pendulum")
    >>> # Use with any integrator that accepts _DynamicalSystem
    
    See Also
    --------
    :class:`~hiten.algorithms.dynamics.rhs._RHSSystem` : Direct constructor
        interface
    :class:`~hiten.algorithms.dynamics.base._DynamicalSystem` : Base interface
    """
    return _RHSSystem(rhs_func, dim, name)


