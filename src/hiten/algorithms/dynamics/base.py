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
from typing import TYPE_CHECKING, Callable, Dict, Literal, Sequence

import numpy as np

from hiten.algorithms.types.configs import EventConfig
from hiten.algorithms.types.options import EventOptions
from hiten.algorithms.types.serialization import _SerializeBase
from hiten.algorithms.utils.config import FASTMATH
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.algorithms.dynamics.protocols import _DynamicalSystemProtocol
    from hiten.algorithms.integrators.base import _Solution

# Global cache for compiled RHS dispatchers (function -> compiled numba dispatcher)
_RHS_DISPATCH_CACHE: Dict[Callable[[float, np.ndarray], np.ndarray], Callable[[float, np.ndarray], np.ndarray]] = {}



class _DynamicalSystem(_SerializeBase, ABC):
    """Provide an abstract base class for dynamical systems.

    Provides common functionality and interface definition for concrete
    dynamical system implementations. Handles state space dimension
    validation and provides utilities for state vector checking.
        
    Parameters
    ----------
    dim : int
        Dimension of the state space (must be positive).
        
    Raises
    ------
    ValueError
        If dim is not positive.

    Notes
    -----
    Subclasses must implement the abstract
    :attr:`~hiten.algorithms.dynamics.base._DynamicalSystem.rhs` property to
    provide the vector field function compatible with
    :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`.
    
    See Also
    --------
    :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol` : Interface specification
    :class:`~hiten.algorithms.dynamics.base._DirectedSystem` : Directional wrapper implementation
    """
    
    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self._dim = dim
        # Cache for compiled RHS dispatcher built from subclass implementation
        self._rhs_compiled: "Callable[[float, np.ndarray], np.ndarray] | None" = None
    
    @property
    def dim(self) -> int:
        """Dimension of the state space."""
        return self._dim
    
    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        """Return compiled and cached right-hand side f(t, y)."""
        if self._rhs_compiled is None:
            impl = self._build_rhs_impl()
            self._rhs_compiled = self._compile_rhs_function(impl)
        return self._rhs_compiled

    @abstractmethod
    def _build_rhs_impl(self) -> Callable[[float, np.ndarray], np.ndarray]:
        """Return a plain Python function implementing f(t, y)."""
        ...
    
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
        :func:`~hiten.algorithms.dynamics.base._validate_initial_state` :
            Module-level validation utility
        """
        if len(y) != self.dim:
            raise ValueError(f"State vector dimension {len(y)} != system dimension {self.dim}")

    def _compile_rhs_function(self, rhs_func: Callable[[float, np.ndarray], np.ndarray]) -> Callable[[float, np.ndarray], np.ndarray]:
        r"""Compile a right-hand side function with Numba JIT for optimal performance.
        
        Provides intelligent JIT compilation that detects pre-compiled Numba dispatchers
        to avoid redundant compilation while ensuring optimal performance for plain
        Python functions. Uses global fast-math settings for numerical optimization.
        
        Parameters
        ----------
        rhs_func : Callable[[float, ndarray], ndarray]
            Function implementing the ODE system dy/dt = f(t, y).
            Can be either a plain Python function or a pre-compiled Numba dispatcher.
            
        Returns
        -------
        Callable[[float, ndarray], ndarray]
            JIT-compiled function compatible with Numba nopython mode.
            If input is already compiled, returns it unchanged.
            
        Notes
        -----
        - Automatically detects pre-compiled Numba dispatchers and reuses them
        - Compiles plain Python functions with Numba JIT for performance
        - Uses global fast-math setting for numerical optimization
        - Compatible with all integrators that accept _DynamicalSystem objects
        
        Examples
        --------
        >>> import numpy as np
        >>> def harmonic_oscillator(t, y):
        ...     return np.array([y[1], -y[0]])
        >>> # In a subclass:
        >>> compiled_rhs = self._compile_rhs_function(harmonic_oscillator)
        >>> # Now compiled_rhs is JIT-compiled for optimal performance
        
        See Also
        --------
        :class:`~hiten.algorithms.dynamics.base._DynamicalSystem` : Base class
            containing this method
        numba.njit : JIT compilation used internally
        """
        # Detect pre-compiled Numba dispatchers to avoid redundant compilation
        try:
            from numba.core.registry import CPUDispatcher
            is_dispatcher = isinstance(rhs_func, CPUDispatcher)
        except Exception:
            is_dispatcher = False

        if is_dispatcher:
            # Function is already compiled, reuse it directly
            return rhs_func
        else:
            # Compile with global fast-math setting for performance
            import numba

            # Central cache to reuse compiled dispatchers across instances
            global _RHS_DISPATCH_CACHE
            try:
                cache_hit = rhs_func in _RHS_DISPATCH_CACHE
            except Exception:
                cache_hit = False
            if cache_hit:
                return _RHS_DISPATCH_CACHE[rhs_func]
            compiled = numba.njit(cache=False, fastmath=FASTMATH)(rhs_func)
            try:
                _RHS_DISPATCH_CACHE[rhs_func] = compiled
            except Exception:
                pass
            return compiled


class _DirectedSystem(_DynamicalSystem):
    """Provide a directional wrapper for forward/backward time integration.

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
    :class:`~hiten.algorithms.dynamics.base._DynamicalSystem` : Base class for
        dynamical systems
    :func:`~hiten.algorithms.dynamics.base._propagate_dynsys` : Generic
        propagation using DirectedSystem
    """

    # Reuse compiled RHS wrappers across instances with identical configuration
    _rhs_cache: dict[tuple[int, int, tuple | None], Callable[[float, np.ndarray], np.ndarray]] = {}

    def __init__(self, base_or_dim: "_DynamicalSystem | int", fwd: int = 1, flip_indices: "slice | Sequence[int] | None" = None):
        if isinstance(base_or_dim, _DynamicalSystem):
            self._base: "_DynamicalSystem | None" = base_or_dim
            dim = base_or_dim.dim
        else:
            self._base = None
            dim = int(base_or_dim)

        super().__init__(dim=dim)

        self._fwd: int = 1 if fwd >= 0 else -1
        self._flip_idx = flip_indices
        # Normalise flip indices for possible JIT compilation
        if flip_indices is None:
            self._flip_idx_norm = None
        elif isinstance(flip_indices, slice):
            self._flip_idx_norm = flip_indices
        else:
            self._flip_idx_norm = np.asarray(flip_indices, dtype=np.int64)

    
    def _build_rhs_impl(self) -> Callable[[float, np.ndarray], np.ndarray]:
        if self._base is None:
            raise AttributeError("`rhs` not implemented: subclass must provide its own implementation when no base system is wrapped.")

        # Build or reuse a compiled wrapper for (base_rhs, fwd, flip_idx)
        base_rhs = self._base.rhs
        # Build a hashable flip key
        if self._flip_idx_norm is None:
            flip_key: tuple | None = None
        elif isinstance(self._flip_idx_norm, slice):
            sl = self._flip_idx_norm
            flip_key = ("slice", sl.start, sl.stop, sl.step)
        else:
            flip_key = ("idx", *tuple(np.asarray(self._flip_idx_norm, dtype=np.int64).tolist()))

        cache_key = (id(base_rhs), self._fwd, flip_key)
        cached = self._rhs_cache.get(cache_key)
        if cached is not None:
            return cached

        fwd = int(self._fwd)
        flip_idx = self._flip_idx_norm

        # Avoid closing over `self` to keep Numba happy
        def _rhs_impl(t: float, y: np.ndarray, _base_rhs=base_rhs, _fwd=fwd, _flip=flip_idx) -> np.ndarray:
            dy = _base_rhs(t, y)
            if _fwd == -1:
                if _flip is None:
                    return -dy
                else:
                    out = dy.copy()
                    out[_flip] *= -1
                    return out
            return dy

        import numba  # local import to avoid module import-time JIT
        compiled = numba.njit(cache=False, fastmath=FASTMATH)(_rhs_impl)
        self._rhs_cache[cache_key] = compiled
        return compiled

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
        typing.Any
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
    dynsys: "_DynamicalSystemProtocol",
    state0: Sequence[float],
    t0: float,
    tf: float,
    forward: int = 1,
    steps: int = 1000,
    method: Literal["fixed", "adaptive", "symplectic"] = "adaptive",
    order: int = 8,
    flip_indices: Sequence[int] | None = None,
    **kwargs
) -> "_Solution":
    """Generic trajectory propagation for dynamical systems.

    Internal utility that handles state validation, directional wrapping,
    and delegation to various integration backends. Supports multiple
    numerical methods with consistent interface.

    Parameters
    ----------
    dynsys : :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
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
    method : {'fixed', 'adaptive', 'symplectic'}, optional
        Integration method to use. Default is 'adaptive'.
    order : int, optional
        Integration order. Default is 8.
    flip_indices : Sequence[int] or None, optional
        State component indices to flip for backward integration.
        Default is None.
    **kwargs
        Additional keyword arguments passed to the integrator, including:
        - event_fn: Numba-compiled scalar event function g(t, y)
        - event_cfg: EventConfig with direction/tolerances
    Returns
    -------
    :class:`~hiten.algorithms.integrators.types._Solution`
        Integration solution containing times and states.

    Notes
    -----
    - Automatically applies
      :class:`~hiten.algorithms.dynamics.base._DirectedSystem` wrapper for
      direction handling
    - Validates initial state dimension against system requirements
    - Supports multiple backends: fixed-step Runge-Kutta, adaptive RK, symplectic
    - Time array is adjusted for integration direction in output
    
    See Also
    --------
    :class:`~hiten.algorithms.dynamics.base._DirectedSystem` : Directional
        wrapper used internally
    :func:`~hiten.algorithms.dynamics.base._validate_initial_state` : State
        validation utility
    """
    from hiten.algorithms.integrators.base import _Solution
    from hiten.algorithms.integrators.rk import AdaptiveRK, RungeKutta
    from hiten.algorithms.integrators.symplectic import _ExtendedSymplectic

    state0_np = _validate_initial_state(state0, dynsys.dim)

    dynsys_dir = _DirectedSystem(dynsys, forward, flip_indices=flip_indices)

    t_eval = np.linspace(t0, tf, steps)

    # Handle zero-length intervals gracefully to avoid integrator issues
    if steps >= 2 and np.isclose(t_eval[0], t_eval[-1]):
        times_signed = forward * t_eval
        states = np.repeat(state0_np[None, :], repeats=len(t_eval), axis=0)
        return _Solution(times_signed, states)

    event_fn = kwargs.get("event_fn", None)
    event_cfg: EventConfig | None = kwargs.get("event_cfg", None)
    event_options: EventOptions | None = kwargs.get("event_options", None)

    if method == "fixed":
        integrator = RungeKutta(order=order)
        sol = integrator.integrate(dynsys_dir, state0_np, t_eval, event_fn=event_fn, event_cfg=event_cfg, event_options=event_options)
        times = sol.times
        states = sol.states

    elif method == "symplectic":
        from hiten.algorithms.dynamics.protocols import \
            _HamiltonianSystemProtocol
        if not isinstance(dynsys, _HamiltonianSystemProtocol):
            raise ValueError("Symplectic method requires a _HamiltonianSystem")
        integrator = _ExtendedSymplectic(order=order)
        sol = integrator.integrate(dynsys_dir, state0_np, t_eval, event_fn=event_fn, event_cfg=event_cfg, event_options=event_options)
        times = sol.times
        states = sol.states

    elif method == "adaptive":
        max_step = kwargs.get("max_step", 1e4)
        rtol = kwargs.get("rtol", 1e-12)
        atol = kwargs.get("atol", 1e-12)
        integrator = AdaptiveRK(order=order, max_step=max_step, rtol=rtol, atol=atol)
        sol = integrator.integrate(dynsys_dir, state0_np, t_eval, event_fn=event_fn, event_cfg=event_cfg, event_options=event_options)
        times = sol.times
        states = sol.states

    # Encode direction in the returned time array for user visibility.
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
    numpy.ndarray
        Validated state vector as float64 numpy array.

    Raises
    ------
    ValueError
        If state vector dimension doesn't match expected_dim.
        
    See Also
    --------
    :meth:`~hiten.algorithms.dynamics.base._DynamicalSystem.validate_state` : Instance method for validation
    :func:`~hiten.algorithms.dynamics.base._propagate_dynsys` : Uses this function for state validation
    """
    state_np = np.asarray(state, dtype=np.float64)
    if state_np.shape != (expected_dim,):
        msg = f"Initial state vector must have {expected_dim} elements, but got shape {state_np.shape}"
        logger.error(msg)
        raise ValueError(msg)
    return state_np
