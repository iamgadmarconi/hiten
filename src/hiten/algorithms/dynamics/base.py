r"""Provide core abstractions for dynamical systems integration.

This module provides abstract base classes and protocols that define the
interface between dynamical systems and numerical integrators. The design
allows integrators to work with any system that implements the minimal
required interface, independent of the underlying physical model.

References
----------
Hairer, E.; Norsett, S.; Wanner, G. (1993).
*Solving Ordinary Differential Equations I: Nonstiff Problems*.
Springer-Verlag.

Koon, W. S.; Lo, M. W.; Marsden, J. E.; Ross, S. D. (2011).
*Dynamical Systems, the Three-Body Problem and Space Mission Design*.
Caltech.
"""

from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Callable, Literal, Protocol, Sequence,
                    runtime_checkable)

import numpy as np
from scipy.integrate import solve_ivp

from hiten.algorithms.utils.config import TOL
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.algorithms.integrators.base import _Solution

@runtime_checkable
class _DynamicalSystemProtocol(Protocol):
    r"""Define the protocol for the minimal interface for dynamical systems.

    This protocol specifies the required attributes that any dynamical system
    must implement to be compatible with the integrator framework. It uses
    structural typing to allow duck typing while maintaining type safety.

    Attributes
    ----------
    dim : int
        Dimension of the state space (number of state variables).
    rhs : Callable[[float, ndarray], ndarray]
        Right-hand side function f(t, y) that computes the time derivative
        dy/dt given time t and state vector y.
        
    Notes
    -----
    The @runtime_checkable decorator allows isinstance() checks against
    this protocol at runtime, enabling flexible type validation.
    """
    
    @property
    def dim(self) -> int:
        """Dimension of the state space."""
        ...
    
    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        ...
            

class _DynamicalSystem(ABC):
    r"""Provide an abstract base class for dynamical systems.

    Provides common functionality and interface definition for concrete
    dynamical system implementations. Handles state space dimension
    validation and provides utilities for state vector checking.

    Parameters
    ----------
    dim : int
        Dimension of the state space (must be >= 1).

    Attributes
    ----------
    dim : int
        State space dimension.

    Raises
    ------
    ValueError
        If dim is not positive.

    Notes
    -----
    Subclasses must implement the abstract :attr:`hiten.algorithms.dynamics.base._DynamicalSystem.rhs` property to provide
    the vector field function compatible with :class:`hiten.algorithms.dynamics.base._DynamicalSystemProtocol`.
    
    See Also
    --------
    :class:`hiten.algorithms.dynamics.base._DynamicalSystemProtocol` : Interface specification
    :class:`hiten.algorithms.dynamics.base._DirectedSystem` : Directional wrapper implementation
    """
    
    def __init__(self, dim: int):
        """Initialize the dynamical system.
        
        Parameters
        ----------
        dim : int
            Dimension of the state space (must be positive).
            
        Raises
        ------
        ValueError
            If dim is not positive.
        """
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self._dim = dim
    
    @property
    def dim(self) -> int:
        """Dimension of the state space."""
        return self._dim
    
    @property
    @abstractmethod
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        pass
    
    def validate_state(self, y: np.ndarray) -> None:
        """Validate state vector dimension.

        Parameters
        ----------
        y : ndarray
            State vector to validate.

        Raises
        ------
        ValueError
            If state vector length differs from system dimension.
            
        See Also
        --------
        :func:`_validate_initial_state` : Module-level validation utility
        """
        if len(y) != self.dim:
            raise ValueError(f"State vector dimension {len(y)} != system dimension {self.dim}")


class _DirectedSystem(_DynamicalSystem):
    r"""Provide a directional wrapper for forward/backward time integration.

    Wraps another dynamical system to enable forward or backward time
    integration with selective component sign handling. Particularly useful
    for Hamiltonian systems where momentum variables change sign under
    time reversal.

    Parameters
    ----------
    base_or_dim : _DynamicalSystem or int
        Either a concrete system instance to wrap, or the state dimension
        for subclasses that implement their own rhs property.
    fwd : int, optional
        Direction flag. Positive values integrate forward in time,
        negative values integrate backward. Default is 1.
    flip_indices : slice or Sequence[int] or None, optional
        Indices of state components whose derivatives should be negated
        when fwd < 0. If None, all components are flipped. Default is None.

    Attributes
    ----------
    dim : int
        Dimension of the underlying system.
    _fwd : int
        Normalized direction flag (+1 or -1).
    _base : _DynamicalSystem or None
        Wrapped system instance (None for subclass usage).
    _flip_idx : slice or Sequence[int] or None
        Component indices to flip for backward integration.

    Raises
    ------
    AttributeError
        If rhs is accessed when no base system was provided and the
        subclass doesn't implement its own rhs property.

    Notes
    -----
    - The wrapper post-processes vector field output without modifying
      the original system
    - Supports both composition (wrapping existing systems) and inheritance
      (subclassing with custom rhs implementation)
    - Attribute access is delegated to the wrapped system when available
    
    Examples
    --------
    >>> # Forward integration (default)
    >>> forward_sys = _DirectedSystem(base_system)
    >>> # Backward integration
    >>> backward_sys = _DirectedSystem(base_system, fwd=-1)
    >>> # Backward with selective momentum flipping
    >>> hamiltonian_backward = _DirectedSystem(ham_sys, fwd=-1, flip_indices=[3,4,5])
    
    See Also
    --------
    :class:`hiten.algorithms.dynamics.base._DynamicalSystem` : Base class for dynamical systems
    :func:`hiten.algorithms.dynamics.base._propagate_dynsys` : Generic propagation using DirectedSystem
    """

    def __init__(self,
                 base_or_dim: "_DynamicalSystem | int",
                 fwd: int = 1,
                 flip_indices: "slice | Sequence[int] | None" = None):

        if isinstance(base_or_dim, _DynamicalSystem):
            self._base: "_DynamicalSystem | None" = base_or_dim
            dim = base_or_dim.dim
        else:
            self._base = None
            dim = int(base_or_dim)

        super().__init__(dim=dim)

        self._fwd: int = 1 if fwd >= 0 else -1
        self._flip_idx = flip_indices

    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:

        if self._base is None:
            raise AttributeError("`rhs` not implemented: subclass must provide "
                                 "its own implementation when no base system "
                                 "is wrapped.")

        base_rhs = self._base.rhs
        flip_idx = self._flip_idx

        def _rhs(t: float, y: np.ndarray) -> np.ndarray:
            dy = base_rhs(t, y)

            if self._fwd == -1:
                if flip_idx is None:
                    dy = -dy
                else:
                    dy = dy.copy()
                    dy[flip_idx] *= -1
            return dy

        return _rhs

    def __repr__(self):
        """String representation of DirectedSystem.
        
        Returns
        -------
        str
            Formatted string showing system parameters.
        """
        return (f"DirectedSystem(dim={self.dim}, fwd={self._fwd}, "
                f"flip_idx={self._flip_idx})")

    def __getattr__(self, item):
        """Delegate attribute access to wrapped system.
        
        Parameters
        ----------
        item : str
            Attribute name to access.
            
        Returns
        -------
        Any
            Attribute value from wrapped system.
            
        Raises
        ------
        AttributeError
            If no wrapped system exists or attribute not found.
        """
        if self._base is None:
            raise AttributeError(item)
        return getattr(self._base, item)


def _propagate_dynsys(
    dynsys: _DynamicalSystem,
    state0: Sequence[float],
    t0: float,
    tf: float,
    forward: int = 1,
    steps: int = 1000,
    method: Literal["scipy", "rk", "symplectic", "adaptive"] = "scipy",
    order: int = 6,
    flip_indices: Sequence[int] | None = None,
) -> "_Solution":
    r"""Generic trajectory propagation for dynamical systems.

    Internal utility that handles state validation, directional wrapping,
    and delegation to various integration backends. Supports multiple
    numerical methods with consistent interface.

    Parameters
    ----------
    dynsys : _DynamicalSystem
        Dynamical system to integrate.
    state0 : Sequence[float]
        Initial state vector.
    t0 : float
        Initial time.
    tf : float
        Final time.
    forward : int, optional
        Integration direction (+1 forward, -1 backward). Default is 1.
    steps : int, optional
        Number of time steps for output. Default is 1000.
    method : {'scipy', 'rk', 'symplectic', 'adaptive'}, optional
        Integration method to use. Default is 'scipy'.
    order : int, optional
        Integration order for non-scipy methods. Default is 6.
    flip_indices : Sequence[int] or None, optional
        State component indices to flip for backward integration.
        Default is None.

    Returns
    -------
    _Solution
        Integration solution containing times and states.

    Notes
    -----
    - Automatically applies :class:`_DirectedSystem` wrapper for direction handling
    - Validates initial state dimension against system requirements
    - Supports multiple backends: SciPy (DOP853), Runge-Kutta, symplectic, adaptive
    - Time array is adjusted for integration direction in output
    
    See Also
    --------
    :class:`_DirectedSystem` : Directional wrapper used internally
    :func:`_validate_initial_state` : State validation utility
    """
    from hiten.algorithms.integrators.base import _Solution
    from hiten.algorithms.integrators.rk import AdaptiveRK, RungeKutta
    from hiten.algorithms.integrators.symplectic import _ExtendedSymplectic

    state0_np = _validate_initial_state(state0, dynsys.dim)

    dynsys_dir = _DirectedSystem(dynsys, forward, flip_indices=flip_indices)

    t_eval = np.linspace(t0, tf, steps)

    if method == "scipy":
        t_span = (t_eval[0], t_eval[-1])

        sol = solve_ivp(
            dynsys_dir.rhs,
            t_span,
            state0_np,
            t_eval=t_eval,
            method='DOP853',
            dense_output=True,
            rtol=TOL,
            atol=TOL,
        )
        times = sol.t
        states = sol.y.T
        
    elif method == "rk":
        integrator = RungeKutta(order=order)
        sol = integrator.integrate(dynsys_dir, state0_np, t_eval)
        times = sol.times
        states = sol.states
        
    elif method == "symplectic":
        integrator = _ExtendedSymplectic(order=order)
        sol = integrator.integrate(dynsys_dir, state0_np, t_eval)
        times = sol.times
        states = sol.states
        
    elif method == "adaptive":
        integrator = AdaptiveRK(order=order, max_step=1e4, rtol=1e-3, atol=1e-6)
        sol = integrator.integrate(dynsys_dir, state0_np, t_eval)
        times = sol.times
        states = sol.states

    times_signed = forward * times

    return _Solution(times_signed, states)


def _validate_initial_state(state, expected_dim=6):
    r"""Validate and normalize initial state vector.

    Converts input to numpy array and validates dimension against expected
    system requirements. Used internally by propagation routines.

    Parameters
    ----------
    state : array_like
        Initial state vector to validate.
    expected_dim : int, optional
        Expected state vector dimension. Default is 6 (typical for CR3BP).

    Returns
    -------
    ndarray
        Validated state vector as float64 numpy array.

    Raises
    ------
    ValueError
        If state vector dimension doesn't match expected_dim.
        
    See Also
    --------
    :meth:`_DynamicalSystem.validate_state` : Instance method for validation
    :func:`_propagate_dynsys` : Uses this function for state validation
    """
    state_np = np.asarray(state, dtype=np.float64)
    if state_np.shape != (expected_dim,):
        msg = f"Initial state vector must have {expected_dim} elements, but got shape {state_np.shape}"
        logger.error(msg)
        raise ValueError(msg)
    return state_np
