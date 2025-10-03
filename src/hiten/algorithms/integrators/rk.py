"""Provide explicit Runge-Kutta integrators used throughout the project.

Both fixed and adaptive step-size variants are provided together with small
convenience factories that select an appropriate implementation given the
desired formal order of accuracy.

Internally the module also defines helper routines to evaluate Hamiltonian
vector fields with numba acceleration and to wrap right-hand side (RHS)
callables into a uniform signature accepted by the integrators.

References
----------
Hairer, E.; Norsett, S.; Wanner, G. (1993). "Solving Ordinary Differential
Equations I".

Dormand, J. R.; Prince, P. J. (1980). "A family of embedded Runge-Kutta
formulas".
"""

import inspect
from typing import Callable, Optional

import numba
import numpy as np
from numba.typed import List

from hiten.algorithms.dynamics.hamiltonian import _hamiltonian_rhs
from hiten.algorithms.dynamics.protocols import (_DynamicalSystemProtocol,
                                                 _HamiltonianSystemProtocol)
from hiten.algorithms.integrators.base import _Integrator, _Solution
from hiten.algorithms.integrators.coefficients.dop853 import E3 as DOP853_E3
from hiten.algorithms.integrators.coefficients.dop853 import E5 as DOP853_E5
from hiten.algorithms.integrators.coefficients.dop853 import \
    INTERPOLATOR_POWER as DOP853_INTERPOLATOR_POWER
from hiten.algorithms.integrators.coefficients.dop853 import \
    N_STAGES as DOP853_N_STAGES
from hiten.algorithms.integrators.coefficients.dop853 import \
    N_STAGES_EXTENDED as DOP853_N_STAGES_EXTENDED
from hiten.algorithms.integrators.coefficients.dop853 import A as DOP853_A
from hiten.algorithms.integrators.coefficients.dop853 import B as DOP853_B
from hiten.algorithms.integrators.coefficients.dop853 import C as DOP853_C
from hiten.algorithms.integrators.coefficients.dop853 import D as DOP853_D
from hiten.algorithms.integrators.coefficients.rk4 import A as RK4_A
from hiten.algorithms.integrators.coefficients.rk4 import B as RK4_B
from hiten.algorithms.integrators.coefficients.rk4 import C as RK4_C
from hiten.algorithms.integrators.coefficients.rk6 import A as RK6_A
from hiten.algorithms.integrators.coefficients.rk6 import B as RK6_B
from hiten.algorithms.integrators.coefficients.rk6 import C as RK6_C
from hiten.algorithms.integrators.coefficients.rk8 import A as RK8_A
from hiten.algorithms.integrators.coefficients.rk8 import B as RK8_B
from hiten.algorithms.integrators.coefficients.rk8 import C as RK8_C
from hiten.algorithms.integrators.coefficients.rk45 import \
    B_HIGH as RK45_B_HIGH
from hiten.algorithms.integrators.coefficients.rk45 import B_LOW as RK45_B_LOW
from hiten.algorithms.integrators.coefficients.rk45 import A as RK45_A
from hiten.algorithms.integrators.coefficients.rk45 import C as RK45_C
from hiten.algorithms.integrators.coefficients.rk45 import E as RK45_E
from hiten.algorithms.integrators.coefficients.rk45 import P as RK45_P
from hiten.algorithms.integrators.utils import (_adjust_step_to_endpoint,
                                                _bisection_update,
                                                _bracket_converged,
                                                _clamp_step,
                                                _crossed_direction,
                                                _error_scale, _event_crossed,
                                                _pi_accept_factor,
                                                _pi_reject_factor,
                                                _select_initial_step)
from hiten.algorithms.types.configs import EventConfig
from hiten.algorithms.types.options import EventOptions
from hiten.algorithms.utils.config import FASTMATH, TOL


class _RungeKuttaBase(_Integrator):
    """Provide shared functionality of explicit Runge-Kutta schemes.

    The class stores a Butcher tableau and provides a single low level
    helper
    :func:`~hiten.algorithms.integrators.rk._RungeKuttaBase._rk_embedded_step`
    that advances one macro time step and, when a second set of weights
    is available, returns an error estimate suitable for adaptive
    step-size control.

    Attributes
    ----------
    _A : numpy.ndarray of shape ``(s, s)``
        Strictly lower triangular array of stage coefficients ``a_ij``.
    _B_HIGH : numpy.ndarray of shape ``(s,)``
        Weights of the high order solution.
    _B_LOW : numpy.ndarray or None
        Weights of the lower order solution, optional. When ``None`` no
        error estimate is produced and
        :func:`~hiten.algorithms.integrators.rk.rk_embedded_step_jit_kernel`
        falls back to the high order result for both outputs.
    _C : numpy.ndarray of shape ``(s,)``
        Nodes ``c_i`` measured in units of the step size.
    _p : int
        Formal order of accuracy of the high order scheme.

    Notes
    -----
    The class is not intended to be used directly. Concrete subclasses
    define the specific coefficients and expose a public interface
    compliant with
    :class:`~hiten.algorithms.integrators.base._Integrator`.
    """

    _A: np.ndarray = None
    _B_HIGH: np.ndarray = None
    _B_LOW: Optional[np.ndarray] = None
    _C: np.ndarray = None
    _p: int = 0

    @property
    def order(self) -> int:
        """Return the formal order of accuracy of the method.
        
        Returns
        -------
        int
            The order of accuracy of the Runge-Kutta method.
        """
        return self._p

    def _build_rhs_wrapper(self, system: _DynamicalSystemProtocol) -> Callable[[float, np.ndarray], np.ndarray]:
        """Return the compiled ``(t, y)`` RHS from the system.

        Ensures ``system.rhs`` has the expected two-argument signature and
        returns it. All systems now expose a compiled dispatcher.

        Parameters
        ----------
        system : :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The dynamical system to wrap.

        Returns
        -------
        Callable[[float, numpy.ndarray], numpy.ndarray]
            The compiled ``(t, y)`` RHS callable.

        Raises
        ------
        ValueError
            If ``system.rhs`` does not have the ``(t, y)`` signature.
        """

        rhs_func = system.rhs
        # Sanity check signature
        sig = inspect.signature(rhs_func)
        if len(sig.parameters) < 2:
            raise ValueError("System.rhs must have signature (t, y)")
        return rhs_func


@numba.njit(cache=False, fastmath=FASTMATH)
def rk_embedded_step_jit_kernel(f, t, y, h, A, B_HIGH, B_LOW, C, has_b_low):
    """Perform a single step of the RK method (no f closure).

    Parameters
    ----------
    f : callable
        The right-hand side of the differential equation.
    t : float
        The current time.
    y : numpy.ndarray
        The current state.
    h : float
        The step size.
    A : numpy.ndarray
        The Butcher tableau.
    B_HIGH : numpy.ndarray
        The high order weights.
    B_LOW : numpy.ndarray
        The low order weights.
    C : numpy.ndarray
        The nodes.
    has_b_low : bool
        Whether to use the low order weights.

    Returns
    -------
    numpy.ndarray
        The high order solution.
    numpy.ndarray
        The low order solution.
    numpy.ndarray
        The error vector.
    """
    s = B_HIGH.size
    k = np.empty((s, y.size), dtype=np.float64)

    k[0] = f(t, y)
    for i in range(1, s):
        y_stage = y.copy()
        for j in range(i):
            a_ij = A[i, j]
            if a_ij != 0.0:
                y_stage += h * a_ij * k[j]
        k[i] = f(t + C[i] * h, y_stage)

    y_high = y.copy()
    for j in range(s):
        bj = B_HIGH[j]
        if bj != 0.0:
            y_high += h * bj * k[j]

    if has_b_low:
        y_low = y.copy()
        for j in range(s):
            bl = B_LOW[j]
            if bl != 0.0:
                y_low += h * bl * k[j]
    else:
        y_low = y_high.copy()
    err_vec = y_high - y_low
    return y_high, y_low, err_vec


@numba.njit(cache=False, fastmath=FASTMATH)
def rk_embedded_step_ham_jit_kernel(t, y, h, A, B_HIGH, B_LOW, C, has_b_low, jac_H, clmo_H, n_dof):
    """Hamiltonian variant of a single RK embedded step (no f closure).

    Parameters
    ----------
    t : float
        The current time.
    y : numpy.ndarray
        The current state.
    h : float
        The step size.
    A : numpy.ndarray
        The Butcher tableau.
    B_HIGH : numpy.ndarray
        The high order weights.
    B_LOW : numpy.ndarray
        The low order weights.
    C : numpy.ndarray
        The nodes.
    has_b_low : bool
        Whether to use the low order weights.
    jac_H : numpy.ndarray
        The Jacobian of the Hamiltonian.
    clmo_H : numpy.ndarray
        The coefficient-layout mapping objects for the Hamiltonian.
    n_dof : int
        The number of degrees of freedom.

    Returns
    -------
    numpy.ndarray
        The high order solution.
    numpy.ndarray
        The low order solution.
    numpy.ndarray
        The error vector.
    """
    s = B_HIGH.size
    k = np.empty((s, y.size), dtype=np.float64)

    k[0] = _hamiltonian_rhs(y, jac_H, clmo_H, n_dof)
    for i in range(1, s):
        y_stage = y.copy()
        for j in range(i):
            a_ij = A[i, j]
            if a_ij != 0.0:
                y_stage += h * a_ij * k[j]
        k[i] = _hamiltonian_rhs(y_stage, jac_H, clmo_H, n_dof)

    y_high = y.copy()
    for j in range(s):
        bj = B_HIGH[j]
        if bj != 0.0:
            y_high += h * bj * k[j]

    if has_b_low:
        y_low = y.copy()
        for j in range(s):
            bl = B_LOW[j]
            if bl != 0.0:
                y_low += h * bl * k[j]
    else:
        y_low = y_high.copy()
    err_vec = y_high - y_low
    return y_high, y_low, err_vec


@numba.njit(cache=False, fastmath=FASTMATH)
def _hermite_eval_dense(y0, f0, y1, f1, x, h):
    """Evaluate cubic Hermite interpolant between (t0,y0,f0) and (t1,y1,f1).

    Uses standard basis: H00=2x^3-3x^2+1, H10=x^3-2x^2+x, H01=-2x^3+3x^2, H11=x^3-x^2.

    Parameters
    ----------
    y0 : numpy.ndarray
        The initial state.
    f0 : numpy.ndarray
        The initial derivative.
    y1 : numpy.ndarray
        The final state.
    f1 : numpy.ndarray
        The final derivative.
    x : float
        The time to evaluate the interpolator at.
    h : float
        The step size.

    Returns
    -------
    numpy.ndarray
        The interpolated state.
    """
    dim = y0.size
    y = np.empty(dim, dtype=np.float64)
    x2 = x * x
    x3 = x2 * x
    H00 = 2.0 * x3 - 3.0 * x2 + 1.0
    H10 = x3 - 2.0 * x2 + x
    H01 = -2.0 * x3 + 3.0 * x2
    H11 = x3 - x2
    for d in range(dim):
        y[d] = (
            H00 * y0[d]
            + H10 * (h * f0[d])
            + H01 * y1[d]
            + H11 * (h * f1[d])
        )
    return y


@numba.njit(cache=False, fastmath=FASTMATH)
def _hermite_refine_in_step(event_fn, t0, y0, f0, t1, y1, f1, h, direction, xtol, gtol):
    """Refine the integration step using bisection on x in [0, 1].
    
    Parameters
    ----------
    event_fn : callable
        The event function.
    t0 : float
        The initial time.
    y0 : numpy.ndarray
        The initial state.
    f0 : numpy.ndarray
        The initial derivative.
    t1 : float
        The final time.
    y1 : numpy.ndarray
        The final state.
    f1 : numpy.ndarray
        The final derivative.
    h : float
        The step size.
    direction : int
        The direction of the event.
    xtol : float
        The tolerance.

    Returns
    -------
    float
        The time of the event.
    numpy.ndarray
        The state at the event.

    Notes
    -----
    This function implements the bisection method to find the time of the event.
    It uses the dense interpolator to evaluate the event function at the time of the event.
    It returns the time of the event and the state at the event.
    The time of the event is returned in the interval [t0, t1].
    """
    a = 0.0
    b = 1.0
    g_left = event_fn(t0, y0)
    max_iter = 128
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        y_mid = _hermite_eval_dense(y0, f0, y1, f1, mid, h)
        g_mid = event_fn(t0 + mid * h, y_mid)
        if abs(g_mid) <= gtol:
            x_hit = mid
            t_hit = t0 + x_hit * h
            y_hit = _hermite_eval_dense(y0, f0, y1, f1, x_hit, h)
            return t_hit, y_hit
        crossed = _crossed_direction(g_left, g_mid, direction)
        a, b, g_left = _bisection_update(a, b, g_left, mid, g_mid, crossed)
        if _bracket_converged(a, b, h, xtol):
            break
    x_hit = b
    t_hit = t0 + x_hit * h
    y_hit = _hermite_eval_dense(y0, f0, y1, f1, x_hit, h)
    return t_hit, y_hit


class _FixedStepRK(_RungeKuttaBase):
    """Implement an explicit fixed-step Runge-Kutta scheme.

    Parameters
    ----------
    name : str
        Human readable identifier of the scheme (e.g. ``"_RK4"``).
    A, B, C : numpy.ndarray
        Butcher tableau as returned by :mod:`~hiten.algorithms.integrators.coefficients.*`.
    order : int
        Formal order of accuracy p of the method.
    **options
        Additional keyword options forwarded to the base :class:`~hiten.algorithms.integrators.base._Integrator`.

    Notes
    -----
    The step size is assumed to be **constant** and is inferred from the
    spacing of the *t_vals* array supplied to :func:`~hiten.algorithms.integrators.rk._FixedStepRK.integrate`.
    """

    def __init__(self, name: str, A: np.ndarray, B: np.ndarray, C: np.ndarray, order: int, **options):
        self._A = A
        self._B_HIGH = B
        self._B_LOW = None
        self._C = C
        self._p = order
        super().__init__(name, **options)

    def integrate(
        self,
        system: _DynamicalSystemProtocol,
        y0: np.ndarray,
        t_vals: np.ndarray,
        *,
        event_fn=None,
        event_cfg: EventConfig | None = None,
        event_options: "EventOptions | None" = None,
        **kwargs,
    ) -> _Solution:
        """Integrate with a fixed-step Runge-Kutta method, with events.

        Parameters
        ----------
        system : :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The dynamical system to integrate.
        y0 : numpy.ndarray
            Initial state of shape ``(system.dim,)``. Units follow the
            provided ``system``.
        t_vals : numpy.ndarray
            Strictly monotonic time nodes of shape ``(N,)`` at which to
            evaluate the solution. Units follow the provided ``system``.
        event_fn : Callable[[float, numpy.ndarray], float], optional
            Scalar event function evaluated as ``g(t, y)``. A zero
            crossing may terminate integration or mark an event.
        event_cfg : :class:`~hiten.algorithms.types.configs.EventConfig` | None
            Configuration controlling event directionality and terminal behavior.
        event_options : :class:`~hiten.algorithms.types.options.EventOptions` | None
            Runtime tuning options controlling event detection tolerances.
        **kwargs
            Additional integration options passed to the implementation.

        Returns
        -------
        :class:`~hiten.algorithms.integrators.types._Solution`
            Integration results with times, states, and derivatives when
            available. Units follow the provided ``system``.
        """
        self.validate_inputs(system, y0, t_vals)

        # Common zero-span short-circuit
        constant_sol = self._maybe_constant_solution(system, y0, t_vals)
        if constant_sol is not None:
            return constant_sol

        # Hamiltonian fast-path: avoid closures by calling parametric RHS
        is_hamiltonian = isinstance(system, _HamiltonianSystemProtocol)
        if is_hamiltonian:
            jac_H, clmo_H, n_dof = system.rhs_params
        else:
            f = self._build_rhs_wrapper(system)

        # Event-enabled path: scan fixed steps for a sign change and refine using RK dense/stage data
        if event_fn is not None:
            event_compiled = self._compile_event_function(event_fn)
            direction = 0 if event_cfg is None else int(event_cfg.direction)
            terminal = 1 if (event_cfg is None or event_cfg.terminal) else 0
            xtol = float(event_options.xtol if event_options is not None else 1.0e-12)
            gtol = float(event_options.gtol if event_options is not None else 1.0e-12)
            if is_hamiltonian:
                hit, t_hit, y_hit, states = _FixedStepRK._integrate_fixed_rk_until_event_ham(
                    y0=y0,
                    t_vals=t_vals,
                    A=self._A,
                    B_HIGH=self._B_HIGH,
                    C=self._C,
                    event_fn=event_compiled,
                    direction=direction,
                    terminal=terminal,
                    xtol=xtol,
                    gtol=gtol,
                    jac_H=jac_H,
                    clmo_H=clmo_H,
                    n_dof=n_dof,
                )
            else:
                hit, t_hit, y_hit, states = _FixedStepRK._integrate_fixed_rk_until_event(
                    f=f,
                    y0=y0,
                    t_vals=t_vals,
                    A=self._A,
                    B_HIGH=self._B_HIGH,
                    C=self._C,
                    event_fn=event_compiled,
                    direction=direction,
                    terminal=terminal,
                    xtol=xtol,
                    gtol=gtol,
                )
            if hit:
                return _Solution(times=np.array([t_vals[0], t_hit], dtype=np.float64), states=np.vstack([y0, y_hit]))
            else:
                return _Solution(times=np.array([t_vals[0], t_vals[-1]], dtype=np.float64), states=np.vstack([y0, states[-1]]))

        has_b_low = self._B_LOW is not None
        B_LOW_arr = self._B_LOW if has_b_low else np.empty(0, dtype=np.float64)
        if isinstance(system, _HamiltonianSystemProtocol):
            states, derivs = _FixedStepRK._integrate_fixed_rk_ham(
                y0, t_vals, self._A, self._B_HIGH, B_LOW_arr, self._C, has_b_low, jac_H, clmo_H, n_dof,
            )
        else:
            states, derivs = _FixedStepRK._integrate_fixed_rk(
                f, y0, t_vals, self._A, self._B_HIGH, B_LOW_arr, self._C, has_b_low,
            )
        # If an event terminated early, kernel returns truncated arrays.
        times_out = t_vals[: states.shape[0]]
        return _Solution(times=times_out, states=states, derivatives=derivs)

    @staticmethod
    @numba.njit(cache=False, fastmath=FASTMATH)
    def _integrate_fixed_rk(
        f,
        y0,
        t_vals,
        A,
        B_HIGH,
        B_LOW,
        C,
        has_b_low,
    ):
        """
        Integrate a dynamical system using a fixed-step Runge-Kutta method.
        
        Parameters
        ----------
        f : callable
            The right-hand side of the differential equation.
        y0 : numpy.ndarray
            The initial state.
        t_vals : numpy.ndarray
            The time points to evaluate the solution at.
        A : numpy.ndarray
            The Butcher tableau.
        B_HIGH : numpy.ndarray
            The high order weights.
        B_LOW : numpy.ndarray
            The low order weights.
        C : numpy.ndarray
            The nodes.
        has_b_low : bool
            Whether the low order weights are used.

        Returns
        -------
        numpy.ndarray
            The states at the time points.
        numpy.ndarray
            The derivatives at the time points.
        """
        n_steps = t_vals.size
        dim = y0.size
        states = np.empty((n_steps, dim), dtype=np.float64)
        derivs = np.empty_like(states)
        states[0] = y0
        derivs[0] = f(t_vals[0], y0)

        for idx in range(n_steps - 1):
            t_n = t_vals[idx]
            h = t_vals[idx + 1] - t_n
            y_n = states[idx]
            y_high, _, _ = rk_embedded_step_jit_kernel(
                f, t_n, y_n, h, A, B_HIGH, B_LOW if has_b_low else np.empty(0, np.float64), C, has_b_low
            )
            states[idx + 1] = y_high
            derivs[idx + 1] = f(t_vals[idx + 1], y_high)
        return states, derivs

    @staticmethod
    @numba.njit(cache=False, fastmath=FASTMATH)
    def _integrate_fixed_rk_ham(
        y0,
        t_vals,
        A,
        B_HIGH,
        B_LOW,
        C,
        has_b_low,
        jac_H,
        clmo_H,
        n_dof,
    ):
        """
        Integrate a Hamiltonian dynamical system using a fixed-step Runge-Kutta method.
        
        Parameters
        ----------
        y0 : numpy.ndarray
            The initial state.
        t_vals : numpy.ndarray
            The time points to evaluate the solution at.
        A : numpy.ndarray
            The Butcher tableau.
        B_HIGH : numpy.ndarray
            The high order weights.
        B_LOW : numpy.ndarray
            The low order weights.
        C : numpy.ndarray
            The nodes.
        has_b_low : bool
            Whether the low order weights are used.
        jac_H : numpy.ndarray
            The Jacobian of the Hamiltonian.
        clmo_H : numpy.ndarray
            The coefficient-layout mapping objects for the Hamiltonian.
        n_dof : int
            The number of degrees of freedom.

        Returns
        -------
        numpy.ndarray
            The states at the time points.
        numpy.ndarray
            The derivatives at the time points.
        """
        n_steps = t_vals.size
        dim = y0.size
        states = np.empty((n_steps, dim), dtype=np.float64)
        derivs = np.empty_like(states)
        states[0] = y0
        derivs[0] = _hamiltonian_rhs(y0, jac_H, clmo_H, n_dof)

        for idx in range(n_steps - 1):
            t_n = t_vals[idx]
            h = t_vals[idx + 1] - t_n
            y_n = states[idx]
            y_high, _, _ = rk_embedded_step_ham_jit_kernel(
                t_n, y_n, h, A, B_HIGH, B_LOW if has_b_low else np.empty(0, np.float64), C, has_b_low, jac_H, clmo_H, n_dof
            )
            states[idx + 1] = y_high
            derivs[idx + 1] = _hamiltonian_rhs(y_high, jac_H, clmo_H, n_dof)
        return states, derivs

    @staticmethod
    @numba.njit(cache=False, fastmath=FASTMATH)
    def _integrate_fixed_rk_until_event(f, y0, t_vals, A, B_HIGH, C, event_fn, direction, terminal, xtol, gtol):
        """
        Integrate a dynamical system using a fixed-step Runge-Kutta method until an event is detected.
        
        Parameters
        ----------
        f : callable
            The right-hand side of the differential equation.
        y0 : numpy.ndarray
            The initial state.
        t_vals : numpy.ndarray
            The time points to evaluate the solution at.
        A : numpy.ndarray
            The Butcher tableau.
        B_HIGH : numpy.ndarray
            The high order weights.
        C : numpy.ndarray
            The nodes.
        event_fn : callable
            The event function.
        direction : int
            The direction of the event.
        terminal : int
            Whether the event is terminal.

        Returns
        -------
        bool
            Whether the event was detected.
        float
            The time of the event.
        numpy.ndarray
            The state at the event.
        numpy.ndarray
            The states at the time points.
        """
        n_steps = t_vals.size
        dim = y0.size
        states = np.empty((n_steps, dim), dtype=np.float64)
        states[0] = y0
        t0 = t_vals[0]
        f_prev = f(t0, y0)
        g_prev = event_fn(t0, y0)
        for idx in range(n_steps - 1):
            t_n = t_vals[idx]
            h = t_vals[idx + 1] - t_n
            y_n = states[idx]
            # advance one step with the provided tableau
            y_high, _, _ = rk_embedded_step_jit_kernel(
                f, t_n, y_n, h, A, B_HIGH, np.empty(0, np.float64), C, False
            )
            f_new = f(t_n + h, y_high)
            # event check at t_{n+1}
            g_new = event_fn(t_n + h, y_high)
            crossed = _event_crossed(g_prev, g_new, direction)
            if crossed:
                t_hit, y_hit = _hermite_refine_in_step(event_fn, t_n, y_n, f_prev, t_n + h, y_high, f_new, h, direction, xtol, gtol)
                return True, t_hit, y_hit, states
            states[idx + 1] = y_high
            f_prev = f_new
            g_prev = g_new
        return False, t_vals[-1], states[-1], states

    @staticmethod
    @numba.njit(cache=False, fastmath=FASTMATH)
    def _integrate_fixed_rk_until_event_ham(y0, t_vals, A, B_HIGH, C, event_fn, direction, terminal, xtol, gtol, jac_H, clmo_H, n_dof):
        """
        Integrate a Hamiltonian system with fixed-step RK until an event is detected.
        Mirrors _integrate_fixed_rk_until_event but uses Hamiltonian kernels.
        """
        n_steps = t_vals.size
        dim = y0.size
        states = np.empty((n_steps, dim), dtype=np.float64)
        states[0] = y0
        t0 = t_vals[0]
        f_prev = _hamiltonian_rhs(y0, jac_H, clmo_H, n_dof)
        g_prev = event_fn(t0, y0)
        for idx in range(n_steps - 1):
            t_n = t_vals[idx]
            h = t_vals[idx + 1] - t_n
            y_n = states[idx]
            # advance one step with the provided tableau
            y_high, _, _ = rk_embedded_step_ham_jit_kernel(
                t_n, y_n, h, A, B_HIGH, np.empty(0, np.float64), C, False, jac_H, clmo_H, n_dof
            )
            f_new = _hamiltonian_rhs(y_high, jac_H, clmo_H, n_dof)
            # event check at t_{n+1}
            g_new = event_fn(t_n + h, y_high)
            crossed = _event_crossed(g_prev, g_new, direction)
            if crossed:
                t_hit, y_hit = _hermite_refine_in_step(event_fn, t_n, y_n, f_prev, t_n + h, y_high, f_new, h, direction, xtol, gtol)
                return True, t_hit, y_hit, states
            states[idx + 1] = y_high
            f_prev = f_new
            g_prev = g_new
        return False, t_vals[-1], states[-1], states


class _RK4(_FixedStepRK):
    """Implement the classical 4th-order Runge-Kutta method.
    
    This is the standard 4th-order explicit Runge-Kutta method, also known
    as RK4 or the "classical" Runge-Kutta method. It uses 4 function
    evaluations per step and has order 4.
    """
    def __init__(self, **opts):
        super().__init__("_RK4", RK4_A, RK4_B, RK4_C, 4, **opts)


class _RK6(_FixedStepRK):
    """Implement a 6th-order Runge-Kutta method.
    
    A 6th-order explicit Runge-Kutta method that provides higher accuracy
    than RK4 at the cost of more function evaluations per step.
    """
    def __init__(self, **opts):
        super().__init__("_RK6", RK6_A, RK6_B, RK6_C, 6, **opts)


class _RK8(_FixedStepRK):
    """Implement an 8th-order Runge-Kutta method.
    
    An 8th-order explicit Runge-Kutta method that provides very high accuracy
    for applications requiring precise numerical integration.
    """
    def __init__(self, **opts):
        super().__init__("_RK8", RK8_A, RK8_B, RK8_C, 8, **opts)


class _AdaptiveStepRK(_RungeKuttaBase):
    """Implement an embedded adaptive Runge-Kutta integrator with PI controller.

    The class provides common constants for PI step-size control; concrete
    methods (e.g. RK45, DOP853) implement the integration drivers.

    Parameters
    ----------
    name : str, default "AdaptiveRK"
        Identifier passed to the :class:`~hiten.algorithms.integrators.base._Integrator` base class.
    rtol, atol : float, optional
        Relative and absolute error tolerances.  Defaults are read from
        :data:`~hiten.utils.config.TOL`.
    max_step : float, optional
        Upper bound on the step size.  infinity by default.
    min_step : float or None, optional
        Lower bound on the step size.  When *None* the value is derived from
        machine precision.

    Attributes
    ----------
    SAFETY, MIN_FACTOR, MAX_FACTOR : float
        Magic constants used by the PI controller.  They follow SciPy's
        implementation and the recommendations by Hairer et al.

    Raises
    ------
    RuntimeError
        If the step size underflows while trying to satisfy the error
        tolerance.
    """

    SAFETY = 0.9
    MIN_FACTOR = 0.2
    MAX_FACTOR = 10.0

    def __init__(self,
                 name: str = "AdaptiveRK",
                 rtol: float = TOL,
                 atol: float = TOL,
                 max_step: float = np.inf,
                 min_step: Optional[float] = None,
                 **options):
        super().__init__(name, **options)
        self._rtol = rtol
        self._atol = atol
        self._max_step = max_step
        if min_step is None:
            self._min_step = 10.0 * np.finfo(float).eps
        else:
            self._min_step = min_step
        if not hasattr(self, "_err_exp") or self._err_exp == 0:
            self._err_exp = 1.0 / (self._p)


@numba.njit(cache=False, fastmath=FASTMATH)
def rk45_step_jit_kernel(f, t, y, h, A, B_HIGH, C, E):
    """Perform a single step of the RK45 method.
    
    Parameters
    ----------
    f : callable
        The right-hand side of the differential equation.
    t : float
        The current time.
    y : numpy.ndarray
        The current state.
    h : float
        The step size.
    A : numpy.ndarray
        The Butcher tableau.
    B_HIGH : numpy.ndarray
        The high order weights.
    C : numpy.ndarray
        The nodes.
    E : numpy.ndarray
        The error weights.

    Returns
    -------
    numpy.ndarray
        The high order solution.
    numpy.ndarray
        The low order solution.
    numpy.ndarray
        The error vector.
    """
    s = 6
    k = np.empty((s + 1, y.size), dtype=np.float64)
    k[0] = f(t, y)
    for i in range(1, s):
        y_stage = y.copy()
        for j in range(i):
            aij = A[i, j]
            if aij != 0.0:
                y_stage += h * aij * k[j]
        k[i] = f(t + C[i] * h, y_stage)
    y_high = y.copy()
    for j in range(s):
        bj = B_HIGH[j]
        if bj != 0.0:
            y_high += h * bj * k[j]
    k[s] = f(t + h, y_high)
    # err_vec = h * (k.T @ E)
    m = k.shape[0]
    n = k.shape[1]
    err_vec = np.zeros(n, dtype=np.float64)
    for j in range(m):
        coeff = E[j]
        if coeff != 0.0:
            err_vec += h * coeff * k[j]
    y_low = y_high - err_vec
    return y_high, y_low, err_vec, k


@numba.njit(cache=False, fastmath=FASTMATH)
def rk45_step_ham_jit_kernel(t, y, h, A, B_HIGH, C, E, jac_H, clmo_H, n_dof):
    """Hamiltonian variant of a single RK45 step (no f closure).
    
    Parameters
    ----------
    t : float
        The current time.
    y : numpy.ndarray
        The current state.
    h : float
        The step size.
    A : numpy.ndarray
        The Butcher tableau.
    B_HIGH : numpy.ndarray
        The high order weights.
    C : numpy.ndarray
        The nodes.
    E : numpy.ndarray
        The error weights.
    jac_H : numpy.ndarray
        The Jacobian of the Hamiltonian.
    clmo_H : numpy.ndarray
        The coefficient-layout mapping objects for the Hamiltonian.
    n_dof : int
        The number of degrees of freedom.

    Returns
    -------
    numpy.ndarray
        The high order solution.
    numpy.ndarray
        The low order solution.
    numpy.ndarray
        The error vector.
    numpy.ndarray
        The intermediate stages.
    """
    s = 6
    k = np.empty((s + 1, y.size), dtype=np.float64)
    # stage 0
    k[0] = _hamiltonian_rhs(y, jac_H, clmo_H, n_dof)
    for i in range(1, s):
        y_stage = y.copy()
        for j in range(i):
            aij = A[i, j]
            if aij != 0.0:
                y_stage += h * aij * k[j]
        k[i] = _hamiltonian_rhs(y_stage, jac_H, clmo_H, n_dof)
    # high order solution
    y_high = y.copy()
    for j in range(s):
        bj = B_HIGH[j]
        if bj != 0.0:
            y_high += h * bj * k[j]
    # one extra stage for error estimate
    k[s] = _hamiltonian_rhs(y_high, jac_H, clmo_H, n_dof)
    # error vector accumulation
    m = k.shape[0]
    n = k.shape[1]
    err_vec = np.zeros(n, dtype=np.float64)
    for j in range(m):
        coeff = E[j]
        if coeff != 0.0:
            err_vec += h * coeff * k[j]
    y_low = y_high - err_vec
    return y_high, y_low, err_vec, k


@numba.njit(cache=False, fastmath=FASTMATH)
def _rk45_build_Q_cache(Kseg, P, dim):
    """Build the Q cache for the RK45 method.
    
    Parameters
    ----------
    Kseg : numpy.ndarray
        The intermediate stages.
    P : numpy.ndarray
        The nodes.
    dim : int
        The dimension of the state.

    Returns
    -------
    numpy.ndarray
        The Q cache.
    """
    Q_cache = np.empty((dim, P.shape[1]), dtype=np.float64)
    for c in range(P.shape[1]):
        for d in range(dim):
            Q_cache[d, c] = 0.0
        for r in range(Kseg.shape[0]):
            coeff = P[r, c]
            if coeff != 0.0:
                for d in range(dim):
                    Q_cache[d, c] += coeff * Kseg[r, d]
    return Q_cache

@numba.njit(cache=False, fastmath=FASTMATH)
def _rk45_eval_dense(y_old, Q_cache, P, x, hseg):
    """Evaluate the dense interpolator at time x.
    
    Parameters
    ----------
    y_old : numpy.ndarray
        The initial state.
    Q_cache : numpy.ndarray
        The Q cache.
    P : numpy.ndarray
        The nodes.
    x : float
        The time to evaluate the interpolator at.
    hseg : float
        The step size.
    
    Returns
    -------
    numpy.ndarray
        The interpolated state.
    """
    dim = y_old.size
    p_len = P.shape[1]
    p = np.empty(p_len, dtype=np.float64)
    val = x
    for c in range(p_len):
        p[c] = val
        val *= x
    y_val = np.empty(dim, dtype=np.float64)
    for d in range(dim):
        acc = 0.0
        for c in range(p_len):
            acc += Q_cache[d, c] * p[c]
        y_val[d] = y_old[d] + hseg * acc
    return y_val

@numba.njit(cache=False, fastmath=FASTMATH)
def _rk45_refine_in_step(event_fn, t0, y0, t1, y1, h, Kseg, P, direction, xtol, gtol):
    """Refine the integration step using bisection on x in [0, 1].
    
    Parameters
    ----------
    event_fn : callable
        The event function.
    t0 : float
        The initial time.
    y0 : numpy.ndarray
        The initial state.
    t1 : float
        The final time.
    y1 : numpy.ndarray
        The final state.
    h : float
        The step size.
    Kseg : numpy.ndarray
        The intermediate stages.
    P : numpy.ndarray
        The nodes.
    direction : int
        The direction of the event.
    xtol : float
        The tolerance.

    Returns
    -------
    float
        The time of the event.
    numpy.ndarray
        The state at the event.
    """
    dim = y0.size
    Q_cache = _rk45_build_Q_cache(Kseg, P, dim)
    a = 0.0
    b = 1.0
    g_left = event_fn(t0, y0)
    max_iter = 128
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        y_mid = _rk45_eval_dense(y0, Q_cache, P, mid, h)
        g_mid = event_fn(t0 + mid * h, y_mid)
        if abs(g_mid) <= gtol:
            x_hit = mid
            t_hit = t0 + x_hit * h
            y_hit = _rk45_eval_dense(y0, Q_cache, P, x_hit, h)
            return t_hit, y_hit
        crossed = _crossed_direction(g_left, g_mid, direction)
        a, b, g_left = _bisection_update(a, b, g_left, mid, g_mid, crossed)
        if _bracket_converged(a, b, h, xtol):
            break
    x_hit = b
    t_hit = t0 + x_hit * h
    y_hit = _rk45_eval_dense(y0, Q_cache, P, x_hit, h)
    return t_hit, y_hit


class _RK45(_AdaptiveStepRK):
    """Implement the Dormand-Prince 5(4) adaptive Runge-Kutta method.
    
    This is the Dormand-Prince 5th-order adaptive Runge-Kutta method with
    4th-order error estimation. It provides a good balance between accuracy
    and computational efficiency for most applications.
    """
    _A = RK45_A
    _B_HIGH = RK45_B_HIGH
    _B_LOW = None
    _C = RK45_C
    _p = 5
    _E = RK45_E

    def __init__(self, **opts):
        super().__init__("_RK45", **opts)

    def _rk_embedded_step(self, f, t, y, h):
        """Perform a single step of the RK45 method.
        
        Parameters
        ----------
        f : callable
            The right-hand side of the differential equation.
        t : float
            The current time.
        y : numpy.ndarray
            The current state.
        h : float
            The step size.

        Returns
        -------
        numpy.ndarray
            The high order solution.
        numpy.ndarray
            The low order solution.
        numpy.ndarray
            The error vector.
        """
        y_high, y_low, err_vec, _ = rk45_step_jit_kernel(f, t, y, h, self._A, self._B_HIGH, self._C, self._E)
        return y_high, y_low, err_vec

    def integrate(self, system: _DynamicalSystemProtocol, y0: np.ndarray, t_vals: np.ndarray, *, event_fn=None, event_cfg: EventConfig | None = None, event_options: "EventOptions | None" = None, **kwargs) -> _Solution:
        """Integrate with RK45 and dense interpolation, with events.

        Parameters
        ----------
        system : :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The dynamical system to integrate.
        y0 : numpy.ndarray
            Initial state of shape ``(system.dim,)``. Units follow the
            provided ``system``.
        t_vals : numpy.ndarray
            Time nodes of shape ``(N,)`` at which to evaluate the
            solution. Units follow the provided ``system``.
        event_fn : Callable[[float, numpy.ndarray], float], optional
            Scalar event function evaluated as ``g(t, y)``.
        event_cfg : :class:`~hiten.algorithms.types.configs.EventConfig` | None
            Configuration controlling directionality and terminal behavior.
        event_options : :class:`~hiten.algorithms.types.options.EventOptions` | None
            Runtime tuning options controlling event detection tolerances.
        **kwargs
            Additional integration options passed to the implementation.

        Returns
        -------
        :class:`~hiten.algorithms.integrators.types._Solution`
            Integration results with times, states, and derivatives when
            available. Units follow the provided ``system``.
        """
        self.validate_inputs(system, y0, t_vals)
        is_hamiltonian = isinstance(system, _HamiltonianSystemProtocol)
        if not is_hamiltonian:
            f = self._build_rhs_wrapper(system)
        # Event-enabled path
        if event_fn is not None:
            event_compiled = self._compile_event_function(event_fn)
            direction = 0 if event_cfg is None else int(event_cfg.direction)
            terminal = 1 if (event_cfg is None or event_cfg.terminal) else 0
            t0 = float(t_vals[0])
            tmax = float(t_vals[-1])
            xtol = float(event_options.xtol if event_options is not None else 1.0e-12)
            gtol = float(event_options.gtol if event_options is not None else 1.0e-12)
            if is_hamiltonian:
                jac_H, clmo_H, n_dof = system.rhs_params
                hit, t_event, y_event, y_last = _RK45._integrate_rk45_until_event_ham(
                    y0=y0,
                    t0=t0,
                    tmax=tmax,
                    A=self._A,
                    B_HIGH=self._B_HIGH,
                    C=self._C,
                    E=self._E,
                    P=RK45_P,
                    rtol=self._rtol,
                    atol=self._atol,
                    max_step=self._max_step,
                    min_step=self._min_step,
                    order=self._p,
                    event_fn=event_compiled,
                    direction=direction,
                    terminal=terminal,
                    xtol=xtol,
                    gtol=gtol,
                    jac_H=jac_H,
                    clmo_H=clmo_H,
                    n_dof=n_dof,
                )
            else:
                hit, t_event, y_event, y_last = _RK45._integrate_rk45_until_event(
                    f=f,
                    y0=y0,
                    t0=t0,
                    tmax=tmax,
                    A=self._A,
                    B_HIGH=self._B_HIGH,
                    C=self._C,
                    E=self._E,
                    P=RK45_P,
                    rtol=self._rtol,
                    atol=self._atol,
                    max_step=self._max_step,
                    min_step=self._min_step,
                    order=self._p,
                    event_fn=event_compiled,
                    direction=direction,
                    terminal=terminal,
                    xtol=xtol,
                    gtol=gtol,
                )
            if hit:
                return _Solution(times=np.array([t0, t_event], dtype=np.float64), states=np.vstack([y0, y_event]))
            else:
                return _Solution(times=np.array([t0, tmax], dtype=np.float64), states=np.vstack([y0, y_last]))
        if is_hamiltonian:
            jac_H, clmo_H, n_dof = system.rhs_params
            states, derivs = _RK45._integrate_rk45_ham(
                y0=y0,
                t_eval=t_vals,
                A=self._A,
                B_HIGH=self._B_HIGH,
                C=self._C,
                E=self._E,
                P=RK45_P,
                rtol=self._rtol,
                atol=self._atol,
                max_step=self._max_step,
                min_step=self._min_step,
                order=self._p,
                jac_H=jac_H,
                clmo_H=clmo_H,
                n_dof=n_dof,
            )
        else:
            states, derivs = _RK45._integrate_rk45(
                f=f,
                y0=y0,
                t_eval=t_vals,
                A=self._A,
                B_HIGH=self._B_HIGH,
                C=self._C,
                E=self._E,
                P=RK45_P,
                rtol=self._rtol,
                atol=self._atol,
                max_step=self._max_step,
                min_step=self._min_step,
                order=self._p,
            )
        return _Solution(times=t_vals.copy(), states=states, derivatives=derivs)

    @staticmethod
    @numba.njit(cache=False, fastmath=FASTMATH)
    def _integrate_rk45(f, y0, t_eval, A, B_HIGH, C, E, P, rtol, atol, max_step, min_step, order):
        """Integrate the RK45 method.
        
        Parameters
        ----------
        f : callable
            The right-hand side of the differential equation.
        y0 : numpy.ndarray
            The initial condition.
        t_eval : numpy.ndarray
            The time points to evaluate the solution at.
        A : numpy.ndarray
            The Butcher tableau.
        B_HIGH : numpy.ndarray
            The high order weights.
        C : numpy.ndarray
            The nodes.
        E : numpy.ndarray
            The error weights.
        P : numpy.ndarray
            The nodes.
        rtol : float
            The relative tolerance.
        atol : float
            The absolute tolerance.
        max_step : float
            The maximum step size.
        min_step : float
            The minimum step size.

        Returns
        -------
        numpy.ndarray
            The states at the time points.
        numpy.ndarray
            The derivatives at the time points.
        """
        t0 = t_eval[0]
        tf = t_eval[-1]
        t = t0
        y = y0.copy()
        ts = List()
        ys = List()
        dys = List()
        Ks = List()
        ts.append(t)
        ys.append(y.copy())
        dys.append(f(t, y))

        # initial step selection (shared heuristic)
        dy0 = dys[0]
        scale0 = atol + rtol * np.abs(y)
        d0 = np.linalg.norm(y / scale0) / np.sqrt(y.size)
        d1 = np.linalg.norm(dy0 / scale0) / np.sqrt(y.size)
        h = _select_initial_step(d0, d1, min_step, max_step)

        err_prev = -1.0

        while (t - tf) * 1.0 < 0.0:
            h = _clamp_step(h, max_step, min_step)
            h = _adjust_step_to_endpoint(t, h, tf)

            y_high, y_low, err_vec, k = rk45_step_jit_kernel(f, t, y, h, A, B_HIGH, C, E)
            scale = _error_scale(y, y_high, rtol, atol)
            err_norm = np.linalg.norm(err_vec / scale) / np.sqrt(err_vec.size)

            if err_norm <= 1.0:
                t_new = t + h
                y_new = y_high
                ts.append(t_new)
                ys.append(y_new.copy())
                f_new = f(t_new, y_new)
                dys.append(f_new)
                Ks.append(k)

                t = t_new
                y = y_new

                factor = _pi_accept_factor(err_norm, err_prev, order)
                h = h * factor
                err_prev = err_norm
            else:
                factor = _pi_reject_factor(err_norm, order)
                h = h * factor
                h = _clamp_step(h, max_step, min_step)

        # Convert lists to arrays
        n_nodes = len(ts)
        dim = y0.size
        ts_arr = np.empty(n_nodes, dtype=np.float64)
        ys_arr = np.empty((n_nodes, dim), dtype=np.float64)
        dys_arr = np.empty_like(ys_arr)
        for i in range(n_nodes):
            ts_arr[i] = ts[i]
            ys_arr[i, :] = ys[i]
            dys_arr[i, :] = dys[i]

        # SciPy-like dense output for RK45 using P matrix
        m = t_eval.size
        y_out = np.empty((m, dim), dtype=np.float64)
        # searchsorted
        last_j = -1
        # Cached Q for the last segment
        Q_cache = np.empty((dim, P.shape[1]), dtype=np.float64)
        for idx in range(m):
            t_q = t_eval[idx]
            # find right index
            j = np.searchsorted(ts_arr, t_q, side='right') - 1
            if j < 0:
                j = 0
            if j > n_nodes - 2:
                j = n_nodes - 2
            t0 = ts_arr[j]
            t1 = ts_arr[j + 1]
            hseg = t1 - t0
            x = (t_q - t0) / hseg
            # Compute Q for new segment if needed
            if j != last_j:
                Kseg = Ks[j]
                Q_cache = _rk45_build_Q_cache(Kseg, P, dim)
                last_j = j
            # Evaluate dense state
            y_old = ys_arr[j]
            y_out[idx, :] = _rk45_eval_dense(y_old, Q_cache, P, x, hseg)

        # Derivatives at t_eval (reuse f)
        derivs_out = np.empty_like(y_out)
        for idx in range(m):
            derivs_out[idx, :] = f(t_eval[idx], y_out[idx, :])

        return y_out, derivs_out

    @staticmethod
    @numba.njit(cache=False, fastmath=FASTMATH)
    def _integrate_rk45_ham(y0, t_eval, A, B_HIGH, C, E, P, rtol, atol, max_step, min_step, order, jac_H, clmo_H, n_dof):
        t0 = t_eval[0]
        tf = t_eval[-1]
        t = t0
        y = y0.copy()
        ts = List()
        ys = List()
        dys = List()
        Ks = List()
        ts.append(t)
        ys.append(y.copy())
        dys.append(_hamiltonian_rhs(y, jac_H, clmo_H, n_dof))

        dy0 = dys[0]
        scale0 = atol + rtol * np.abs(y)
        d0 = np.linalg.norm(y / scale0) / np.sqrt(y.size)
        d1 = np.linalg.norm(dy0 / scale0) / np.sqrt(y.size)

        h = _select_initial_step(d0, d1, min_step, max_step)

        err_prev = -1.0

        while (t - tf) * 1.0 < 0.0:
            h = _clamp_step(h, max_step, min_step)
            h = _adjust_step_to_endpoint(t, h, tf)

            y_high, y_low, err_vec, k = rk45_step_ham_jit_kernel(t, y, h, A, B_HIGH, C, E, jac_H, clmo_H, n_dof)
            scale = _error_scale(y, y_high, rtol, atol)
            err_norm = np.linalg.norm(err_vec / scale) / np.sqrt(err_vec.size)

            if err_norm <= 1.0:
                t_new = t + h
                y_new = y_high
                ts.append(t_new)
                ys.append(y_new.copy())
                f_new = _hamiltonian_rhs(y_new, jac_H, clmo_H, n_dof)
                dys.append(f_new)
                Ks.append(k)

                t = t_new
                y = y_new

                h = h * _pi_accept_factor(err_norm, err_prev, order)
                err_prev = err_norm
            else:
                h = h * _pi_reject_factor(err_norm, order)
                h = _clamp_step(h, max_step, min_step)

        n_nodes = len(ts)
        dim = y0.size
        ts_arr = np.empty(n_nodes, dtype=np.float64)
        ys_arr = np.empty((n_nodes, dim), dtype=np.float64)
        dys_arr = np.empty_like(ys_arr)
        for i in range(n_nodes):
            ts_arr[i] = ts[i]
            ys_arr[i, :] = ys[i]
            dys_arr[i, :] = dys[i]

        m = t_eval.size
        y_out = np.empty((m, dim), dtype=np.float64)
        last_j = -1
        Q_cache = np.empty((dim, P.shape[1]), dtype=np.float64)
        for idx in range(m):
            t_q = t_eval[idx]
            j = np.searchsorted(ts_arr, t_q, side='right') - 1
            if j < 0:
                j = 0
            if j > n_nodes - 2:
                j = n_nodes - 2
            t0s = ts_arr[j]
            t1s = ts_arr[j + 1]
            hseg = t1s - t0s
            x = (t_q - t0s) / hseg
            if j != last_j:
                Kseg = Ks[j]
                Q_cache = _rk45_build_Q_cache(Kseg, P, dim)
                last_j = j
            y_old = ys_arr[j]
            y_out[idx, :] = _rk45_eval_dense(y_old, Q_cache, P, x, hseg)

        derivs_out = np.empty_like(y_out)
        for idx in range(m):
            derivs_out[idx, :] = _hamiltonian_rhs(y_out[idx, :], jac_H, clmo_H, n_dof)

        return y_out, derivs_out

    @staticmethod
    @numba.njit(cache=False, fastmath=FASTMATH)
    def _integrate_rk45_until_event(f, y0, t0, tmax, A, B_HIGH, C, E, P, rtol, atol, max_step, min_step, order, event_fn, direction, terminal, xtol, gtol):
        """Integrate the RK45 method until an event is detected.
        
        Parameters
        ----------
        f : callable
            The right-hand side of the differential equation.
        y0 : numpy.ndarray
            The initial condition.
        t0 : float
            The initial time.
        tmax : float
            The maximum time.
        A : numpy.ndarray
            The Butcher tableau.
        B_HIGH : numpy.ndarray
            The high order weights.
        C : numpy.ndarray
            The nodes.
        E : numpy.ndarray
            The error weights.
        P : numpy.ndarray
            The nodes.
        rtol : float
            The relative tolerance.
        atol : float
            The absolute tolerance.
        max_step : float
            The maximum step size.
        min_step : float
            The minimum step size.
        order : int
            The order of the method.
        event_fn : callable
            The event function.
        direction : int
            The direction of the event.
        terminal : int
            Whether the event is terminal.
        xtol : float
            The tolerance.

        Returns
        -------
        bool
            Whether the event was detected.
        float
            The time of the event.
        numpy.ndarray
            The state at the event.
        """
        t = t0
        y = y0.copy()
        f_curr = f(t, y)
        g_prev = event_fn(t, y)

        # initial step heuristic
        scale0 = atol + rtol * np.abs(y)
        d0 = np.linalg.norm(y / scale0) / np.sqrt(y.size)
        d1 = np.linalg.norm(f_curr / scale0) / np.sqrt(y.size)
        h = _select_initial_step(d0, d1, min_step, max_step)

        err_prev = -1.0
        

        while (t - tmax) * 1.0 < 0.0:
            h = _clamp_step(h, max_step, min_step)
            h = _adjust_step_to_endpoint(t, h, tmax)

            y_high, y_low, err_vec, K = rk45_step_jit_kernel(f, t, y, h, A, B_HIGH, C, E)
            scale = _error_scale(y, y_high, rtol, atol)
            err_norm = np.linalg.norm(err_vec / scale) / np.sqrt(err_vec.size)

            if err_norm <= 1.0:
                t_new = t + h
                y_new = y_high
                g_new = event_fn(t_new, y_new)
                crossed = _event_crossed(g_prev, g_new, direction)
                if crossed:
                    t_hit, y_hit = _rk45_refine_in_step(event_fn, t, y, t_new, y_new, h, K, P, direction, xtol, gtol)
                    return True, t_hit, y_hit, y_new

                # accept
                t = t_new
                y = y_new
                f_curr = f(t, y)
                g_prev = g_new

                h = h * _pi_accept_factor(err_norm, err_prev, order)
                err_prev = err_norm
            else:
                h = h * _pi_reject_factor(err_norm, order)
                h = _clamp_step(h, max_step, min_step)

        return False, t, y, y

    @staticmethod
    @numba.njit(cache=False, fastmath=FASTMATH)
    def _integrate_rk45_until_event_ham(y0, t0, tmax, A, B_HIGH, C, E, P, rtol, atol, max_step, min_step, order, event_fn, direction, terminal, xtol, gtol, jac_H, clmo_H, n_dof):
        """Hamiltonian variant of RK45 event integration (no closures)."""
        t = t0
        y = y0.copy()
        f_curr = _hamiltonian_rhs(y, jac_H, clmo_H, n_dof)
        g_prev = event_fn(t, y)

        # initial step heuristic
        scale0 = atol + rtol * np.abs(y)
        d0 = np.linalg.norm(y / scale0) / np.sqrt(y.size)
        d1 = np.linalg.norm(f_curr / scale0) / np.sqrt(y.size)
        h = _select_initial_step(d0, d1, min_step, max_step)

        err_prev = -1.0
        
        while (t - tmax) * 1.0 < 0.0:
            h = _clamp_step(h, max_step, min_step)
            h = _adjust_step_to_endpoint(t, h, tmax)

            y_high, y_low, err_vec, K = rk45_step_ham_jit_kernel(t, y, h, A, B_HIGH, C, E, jac_H, clmo_H, n_dof)
            scale = _error_scale(y, y_high, rtol, atol)
            err_norm = np.linalg.norm(err_vec / scale) / np.sqrt(err_vec.size)

            if err_norm <= 1.0:
                t_new = t + h
                y_new = y_high
                g_new = event_fn(t_new, y_new)
                crossed = _event_crossed(g_prev, g_new, direction)
                if crossed:
                    t_hit, y_hit = _rk45_refine_in_step(event_fn, t, y, t_new, y_new, h, K, P, direction, xtol, gtol)
                    return True, t_hit, y_hit, y_new

                # accept
                t = t_new
                y = y_new
                f_curr = _hamiltonian_rhs(y, jac_H, clmo_H, n_dof)
                g_prev = g_new

                h = h * _pi_accept_factor(err_norm, err_prev, order)
                err_prev = err_norm
            else:
                h = h * _pi_reject_factor(err_norm, order)
                h = _clamp_step(h, max_step, min_step)

        return False, t, y, y


@numba.njit(cache=False, fastmath=FASTMATH)
def dop853_step_jit_kernel(f, t, y, h, A, B_HIGH, C, E5, E3):
    """Perform a single step of the DOP853 method.
    
    Parameters
    ----------
    f : callable
        The right-hand side of the differential equation.
    t : float
        The current time.
    y : numpy.ndarray
        The current state.
    h : float
        The step size.
    A : numpy.ndarray
        The Butcher tableau.
    B_HIGH : numpy.ndarray
        The high order weights.
    C : numpy.ndarray
        The nodes.
    E5 : numpy.ndarray
        The error weights.
    E3 : numpy.ndarray
        The error weights.
    
    Returns
    -------
    numpy.ndarray
        The high order solution.
    numpy.ndarray
        The low order solution.
    numpy.ndarray
        The error vector.
    """
    s = B_HIGH.size
    k = np.empty((s + 1, y.size), dtype=np.float64)
    k[0] = f(t, y)
    for i in range(1, s):
        y_stage = y.copy()
        for j in range(i):
            a_ij = A[i, j]
            if a_ij != 0.0:
                y_stage += h * a_ij * k[j]
        k[i] = f(t + C[i] * h, y_stage)
    y_high = y.copy()
    for j in range(s):
        b_j = B_HIGH[j]
        if b_j != 0.0:
            y_high += h * b_j * k[j]
    k[s] = f(t + h, y_high)
    # error using E5/E3 combo
    m = k.shape[0]
    n = k.shape[1]
    err5 = np.zeros(n, dtype=np.float64)
    err3 = np.zeros(n, dtype=np.float64)
    for j in range(m):
        c5 = E5[j]
        c3 = E3[j]
        if c5 != 0.0:
            err5 += c5 * k[j]
        if c3 != 0.0:
            err3 += c3 * k[j]
    err5 *= h
    err3 *= h
    denom = np.hypot(np.abs(err5), 0.1 * np.abs(err3))
    err_vec = np.empty_like(err5)
    for i in range(n):
        if denom[i] > 0.0:
            err_vec[i] = err5[i] * (np.abs(err5[i]) / denom[i])
        else:
            err_vec[i] = err5[i]
    y_low = y_high - err_vec
    return y_high, y_low, err_vec, err5, err3, k


@numba.njit(cache=False, fastmath=FASTMATH)
def dop853_step_ham_jit_kernel(t, y, h, A, B_HIGH, C, E5, E3, jac_H, clmo_H, n_dof):
    """Hamiltonian variant of a single DOP853 step (no f closure).
    
    Parameters
    ----------
    t : float
        The current time.
    y : numpy.ndarray
        The current state.
    h : float
        The step size.
    A : numpy.ndarray
        The Butcher tableau.
    B_HIGH : numpy.ndarray
        The high order weights.
    C : numpy.ndarray
        The nodes.
    E5 : numpy.ndarray
        The error weights.
    E3 : numpy.ndarray
        The error weights.
    jac_H : numpy.ndarray, shape (n_dof, n_dof)
        The Jacobian of the Hamiltonian.
    clmo_H : numpy.ndarray
        The coefficient-layout mapping objects for the Hamiltonian.
    n_dof : int, shape (n_dof,)
        The number of degrees of freedom.

    Returns
    -------
    numpy.ndarray
        The high order solution.
    numpy.ndarray
        The low order solution.
    numpy.ndarray
        The error vector.
    numpy.ndarray
        The intermediate stages.
    """
    s = B_HIGH.size
    k = np.empty((s + 1, y.size), dtype=np.float64)
    k[0] = _hamiltonian_rhs(y, jac_H, clmo_H, n_dof)
    for i in range(1, s):
        y_stage = y.copy()
        for j in range(i):
            a_ij = A[i, j]
            if a_ij != 0.0:
                y_stage += h * a_ij * k[j]
        k[i] = _hamiltonian_rhs(y_stage, jac_H, clmo_H, n_dof)
    y_high = y.copy()
    for j in range(s):
        b_j = B_HIGH[j]
        if b_j != 0.0:
            y_high += h * b_j * k[j]
    k[s] = _hamiltonian_rhs(y_high, jac_H, clmo_H, n_dof)
    m = k.shape[0]
    n = k.shape[1]
    err5 = np.zeros(n, dtype=np.float64)
    err3 = np.zeros(n, dtype=np.float64)
    for j in range(m):
        c5 = E5[j]
        c3 = E3[j]
        if c5 != 0.0:
            err5 += c5 * k[j]
        if c3 != 0.0:
            err3 += c3 * k[j]
    err5 *= h
    err3 *= h
    denom = np.hypot(np.abs(err5), 0.1 * np.abs(err3))
    err_vec = np.empty_like(err5)
    for i in range(n):
        if denom[i] > 0.0:
            err_vec[i] = err5[i] * (np.abs(err5[i]) / denom[i])
        else:
            err_vec[i] = err5[i]
    y_low = y_high - err_vec
    return y_high, y_low, err_vec, err5, err3, k

@numba.njit(cache=False, fastmath=FASTMATH)
def _dop853_build_dense_cache(f, t_old, y_old, f_old, y_new, f_new, hseg, Kseg, A_full, C_full, D, n_stages_extended, interpolator_power):
    """Build the dense cache for the DOP853 method.
    
    Parameters
    ----------
    f : callable
        The right-hand side of the differential equation.
    t_old : float
        The initial time.
    y_old : numpy.ndarray
        The initial state.
    f_old : numpy.ndarray
        The initial derivative.
    y_new : numpy.ndarray
        The final state.
    f_new : numpy.ndarray
        The final derivative.
    hseg : float
        The step size.
    Kseg : numpy.ndarray
        The intermediate stages.
    A_full : numpy.ndarray
        The full Butcher tableau.
    C_full : numpy.ndarray
        The full Butcher tableau.
    D : numpy.ndarray
        The full Butcher tableau.
    n_stages_extended : int
        The number of stages.
    interpolator_power : int
        The power of the interpolator.

    Returns
    -------
    numpy.ndarray
        The dense cache.

    Notes
    -----
    This function implements the dense cache for the DOP853 method.
    It uses the dense cache to evaluate the interpolator at the time x.
    It returns the interpolated state.
    The interpolated state is returned in the interval [y0, y1].
    The time of the event is returned in the interval [t0, t1].
    """
    # Build extended stages Kext
    dim = y_old.size
    s_used = Kseg.shape[0]
    Kext = np.empty((n_stages_extended, dim), dtype=np.float64)
    for r in range(s_used):
        for d in range(dim):
            Kext[r, d] = Kseg[r, d]
    # compute additional stages using A_full rows
    y0loc = y_old
    for srow in range(s_used, n_stages_extended):
        y_stage = np.empty(dim, dtype=np.float64)
        for d in range(dim):
            acc = 0.0
            for r in range(srow):
                a = A_full[srow, r]
                if a != 0.0:
                    acc += a * Kext[r, d]
            y_stage[d] = y0loc[d] + hseg * acc
        t_stage = t_old + C_full[srow] * hseg
        k_vec = f(t_stage, y_stage)
        for d in range(dim):
            Kext[srow, d] = k_vec[d]

    # Build F_cache (interpolator_power x dim)
    F_cache = np.empty((interpolator_power, dim), dtype=np.float64)
    delta_y = y_new - y_old
    for d in range(dim):
        F_cache[0, d] = delta_y[d]
        F_cache[1, d] = hseg * f_old[d] - delta_y[d]
        F_cache[2, d] = 2.0 * delta_y[d] - hseg * (f_new[d] + f_old[d])
    rows_remaining = interpolator_power - 3
    for irem in range(rows_remaining):
        for d in range(dim):
            acc = 0.0
            for r in range(n_stages_extended):
                coeff = D[irem, r]
                if coeff != 0.0:
                    acc += coeff * Kext[r, d]
            F_cache[3 + irem, d] = hseg * acc
    return F_cache

@numba.njit(cache=False, fastmath=FASTMATH)
def _dop853_build_dense_cache_ham(t_old, y_old, f_old, y_new, f_new, hseg, Kseg, A_full, C_full, D, n_stages_extended, interpolator_power, jac_H, clmo_H, n_dof):
    """Build the dense cache for the DOP853 method (Hamiltonian variant).
    
    Mirrors _dop853_build_dense_cache but uses _hamiltonian_rhs to extend stages.

    Parameters
    ----------
    t_old : float
        The initial time.
    y_old : numpy.ndarray
        The initial state.
    f_old : numpy.ndarray
        The initial derivative.
    y_new : numpy.ndarray
        The final state.
    f_new : numpy.ndarray
        The final derivative.
    hseg : float
        The step size.
    Kseg : numpy.ndarray
        The intermediate stages.
    A_full : numpy.ndarray
        The full Butcher tableau.
    C_full : numpy.ndarray
        The full Butcher tableau.
    D : numpy.ndarray
        The full Butcher tableau.
    n_stages_extended : int
        The number of stages.
    interpolator_power : int
        The power of the interpolator.
    jac_H : numpy.ndarray
        The Jacobian of the Hamiltonian.
    clmo_H : numpy.ndarray
        The coefficient-layout mapping objects for the Hamiltonian.
    n_dof : int, shape (n_dof,)
        The number of degrees of freedom.

    Returns
    -------
    numpy.ndarray
        The dense cache.

    Notes
    -----
    This function implements the dense cache for the DOP853 method (Hamiltonian variant).
    It mirrors _dop853_build_dense_cache but uses _hamiltonian_rhs to extend stages.
    """
    dim = y_old.size
    s_used = Kseg.shape[0]
    Kext = np.empty((n_stages_extended, dim), dtype=np.float64)
    for r in range(s_used):
        for d in range(dim):
            Kext[r, d] = Kseg[r, d]
    for srow in range(s_used, n_stages_extended):
        y_stage = np.empty(dim, dtype=np.float64)
        for d in range(dim):
            acc = 0.0
            for r in range(srow):
                a = A_full[srow, r]
                if a != 0.0:
                    acc += a * Kext[r, d]
            y_stage[d] = y_old[d] + hseg * acc
        k_vec = _hamiltonian_rhs(y_stage, jac_H, clmo_H, n_dof)
        for d in range(dim):
            Kext[srow, d] = k_vec[d]
    F_cache = np.empty((interpolator_power, dim), dtype=np.float64)
    delta_y = y_new - y_old
    for d in range(dim):
        F_cache[0, d] = delta_y[d]
        F_cache[1, d] = hseg * f_old[d] - delta_y[d]
        F_cache[2, d] = 2.0 * delta_y[d] - hseg * (f_new[d] + f_old[d])
    rows_remaining = interpolator_power - 3
    for irem in range(rows_remaining):
        for d in range(dim):
            acc = 0.0
            for r in range(n_stages_extended):
                coeff = D[irem, r]
                if coeff != 0.0:
                    acc += coeff * Kext[r, d]
            F_cache[3 + irem, d] = hseg * acc
    return F_cache

@numba.njit(cache=False, fastmath=FASTMATH)
def _dop853_eval_dense(y_old, F_cache, interpolator_power, x):
    """Evaluate the dense interpolator at time x.

    Parameters
    ----------
    y_old : numpy.ndarray
        The initial state.
    F_cache : numpy.ndarray
        The dense cache.
    interpolator_power : int
        The power of the interpolator.
    x : float
        The time to evaluate the interpolator at.

    Returns
    -------
    numpy.ndarray
        The interpolated state.

    Notes
    -----
    This function implements the dense interpolator.
    It uses the dense cache to evaluate the interpolator at the time x.
    It returns the interpolated state.
    The interpolated state is returned in the interval [y0, y1].
    The time of the event is returned in the interval [t0, t1].
    """
    dim = y_old.size
    y_val = np.zeros(dim, dtype=np.float64)
    for i in range(interpolator_power - 1, -1, -1):
        for d in range(dim):
            y_val[d] += F_cache[i, d]
        if (interpolator_power - 1 - i) % 2 == 0:
            for d in range(dim):
                y_val[d] *= x
        else:
            one_minus_x = 1.0 - x
            for d in range(dim):
                y_val[d] *= one_minus_x
    for d in range(dim):
        y_val[d] += y_old[d]
    return y_val

@numba.njit(cache=False, fastmath=FASTMATH)
def _dop853_refine_in_step(f, event_fn, t0, y0, f0, t1, y1, f1, h, Kseg, A_full, C_full, D, n_stages_extended, interpolator_power, direction, xtol, gtol):
    """Refine the integration step using bisection on x in [0, 1].
    
    Parameters
    ----------
    f : callable
        The right-hand side of the differential equation.
    event_fn : callable
        The event function.
    t0 : float
        The initial time.
    y0 : numpy.ndarray
        The initial state.
    f0 : numpy.ndarray
        The initial derivative.
    t1 : float
        The final time.
    y1 : numpy.ndarray
        The final state.
    f1 : numpy.ndarray
        The final derivative.
    h : float
        The step size.
    Kseg : numpy.ndarray
        The intermediate stages.
    A_full : numpy.ndarray
        The full Butcher tableau.
    C_full : numpy.ndarray
        The full Butcher tableau.
    D : numpy.ndarray
        The full Butcher tableau.
    n_stages_extended : int
        The number of stages.
    interpolator_power : int
        The power of the interpolator.
    direction : int
        The direction of the event.
    xtol : float
        The tolerance.

    Returns
    -------
    float
        The time of the event.
    numpy.ndarray
        The state at the event.

    Notes
    -----
    This function implements the bisection method to find the time of the event.
    It uses the dense cache to evaluate the event function at the endpoints and
    the intermediate points.
    It then uses the bisection method to find the time of the event.
    It returns the time of the event and the state at the event.
    The time of the event is returned in the interval [t0, t1].
    The state at the event is returned in the interval [y0, y1].
    """
    # Build dense cache once for this step
    F_cache = _dop853_build_dense_cache(
        f=f,
        t_old=t0,
        y_old=y0,
        f_old=f0,
        y_new=y1,
        f_new=f1,
        hseg=h,
        Kseg=Kseg,
        A_full=A_full,
        C_full=C_full,
        D=D,
        n_stages_extended=n_stages_extended,
        interpolator_power=interpolator_power,
    )
    # Bisection on x in [0, 1]
    a = 0.0
    b = 1.0
    # Evaluate g at endpoints via event_fn directly (we have y0,y1,f0,f1)
    g_left = event_fn(t0, y0)
    g_right = event_fn(t1, y1)
    max_iter = 128
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        y_mid = _dop853_eval_dense(y0, F_cache, interpolator_power, mid)
        g_mid = event_fn(t0 + mid * h, y_mid)
        if abs(g_mid) <= gtol:
            x_hit = mid
            t_hit = t0 + x_hit * h
            y_hit = _dop853_eval_dense(y0, F_cache, interpolator_power, x_hit)
            return t_hit, y_hit
        crossed = _crossed_direction(g_left, g_mid, direction)
        a, b, g_left = _bisection_update(a, b, g_left, mid, g_mid, crossed)
        if _bracket_converged(a, b, h, xtol):
            break
    x_hit = b
    t_hit = t0 + x_hit * h
    y_hit = _dop853_eval_dense(y0, F_cache, interpolator_power, x_hit)
    return t_hit, y_hit


@numba.njit(cache=False, fastmath=FASTMATH)
def _dop853_refine_in_step_ham(event_fn, t0, y0, f0, t1, y1, f1, h, Kseg, A_full, C_full, D, n_stages_extended, interpolator_power, direction, xtol, gtol, jac_H, clmo_H, n_dof):
    """Hamiltonian variant of DOP853 in-step root refinement.

    Mirrors _dop853_refine_in_step but builds missing stages via _hamiltonian_rhs.
    """
    # Build dense cache once for this step (Hamiltonian RHS used to extend stages)
    dim = y0.size
    s_used = Kseg.shape[0]
    Kext = np.empty((n_stages_extended, dim), dtype=np.float64)
    for r in range(s_used):
        for d in range(dim):
            Kext[r, d] = Kseg[r, d]
    y_old = y0
    for srow in range(s_used, n_stages_extended):
        y_stage = np.empty(dim, dtype=np.float64)
        for d in range(dim):
            acc = 0.0
            for rr in range(srow):
                a = A_full[srow, rr]
                if a != 0.0:
                    acc += a * Kext[rr, d]
            y_stage[d] = y_old[d] + h * acc
        t_stage = t0 + C_full[srow] * h
        k_vec = _hamiltonian_rhs(y_stage, jac_H, clmo_H, n_dof)
        for d in range(dim):
            Kext[srow, d] = k_vec[d]
    # Build F_cache
    F_cache = np.empty((interpolator_power, dim), dtype=np.float64)
    delta_y = y1 - y0
    for d in range(dim):
        F_cache[0, d] = delta_y[d]
        F_cache[1, d] = h * f0[d] - delta_y[d]
        F_cache[2, d] = 2.0 * delta_y[d] - h * (f1[d] + f0[d])
    rows_remaining = interpolator_power - 3
    for irem in range(rows_remaining):
        for d in range(dim):
            acc = 0.0
            for r in range(n_stages_extended):
                coeff = D[irem, r]
                if coeff != 0.0:
                    acc += coeff * Kext[r, d]
            F_cache[3 + irem, d] = h * acc
    # Bisection on x in [0, 1]
    a = 0.0
    b = 1.0
    g_left = event_fn(t0, y0)
    g_right = event_fn(t1, y1)
    max_iter = 128
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        y_mid = _dop853_eval_dense(y0, F_cache, interpolator_power, mid)
        g_mid = event_fn(t0 + mid * h, y_mid)
        if abs(g_mid) <= gtol:
            x_hit = mid
            t_hit = t0 + x_hit * h
            y_hit = _dop853_eval_dense(y0, F_cache, interpolator_power, x_hit)
            return t_hit, y_hit
        crossed = _crossed_direction(g_left, g_mid, direction)
        a, b, g_left = _bisection_update(a, b, g_left, mid, g_mid, crossed)
        if _bracket_converged(a, b, h, xtol):
            break
    x_hit = b
    t_hit = t0 + x_hit * h
    y_hit = _dop853_eval_dense(y0, F_cache, interpolator_power, x_hit)
    return t_hit, y_hit

class _DOP853(_AdaptiveStepRK):
    """Implement the Dormand-Prince 8(5,3) adaptive Runge-Kutta method.
    
    This is the Dormand-Prince 8th-order adaptive Runge-Kutta method with
    5th and 3rd-order error estimation. It provides very high accuracy
    for applications requiring precise numerical integration.
    """
    _A = DOP853_A[:DOP853_N_STAGES, :DOP853_N_STAGES]
    _B_HIGH = DOP853_B[:DOP853_N_STAGES]
    _B_LOW = None
    _C = DOP853_C[:DOP853_N_STAGES]

    _p = 8
    _E3 = DOP853_E3
    _E5 = DOP853_E5
    _N_STAGES = DOP853_N_STAGES

    def __init__(self, **opts):
        super().__init__("_DOP853", **opts)

    def _rk_embedded_step(self, f, t, y, h):
        """
        Perform a single step of the DOP853 method.

        Parameters
        ----------
        f : callable
            The right-hand side of the differential equation.
        t : float
            The current time.
        y : numpy.ndarray
            The current state.
        h : float
            The step size.

        Returns
        -------
        numpy.ndarray
            The high order solution.
        numpy.ndarray
            The low order solution.
        numpy.ndarray
            The error vector.
        """
        y_high, y_low, err_vec, _, _ = dop853_step_jit_kernel(
            f, t, y, h, self._A, self._B_HIGH, self._C, self._E5, self._E3
        )
        return y_high, y_low, err_vec

    def integrate(self, system: _DynamicalSystemProtocol, y0: np.ndarray, t_vals: np.ndarray, *, event_fn=None, event_cfg: EventConfig | None = None, event_options: "EventOptions | None" = None, **kwargs) -> _Solution:
        """Integrate with DOP853 and dense interpolation, with events.

        Parameters
        ----------
        system : :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The dynamical system to integrate.
        y0 : numpy.ndarray
            Initial state of shape ``(system.dim,)``. Units follow the
            provided ``system``.
        t_vals : numpy.ndarray
            Time nodes of shape ``(N,)`` at which to evaluate the
            solution. Units follow the provided ``system``.
        event_fn : Callable[[float, numpy.ndarray], float], optional
            Scalar event function evaluated as ``g(t, y)``.
        event_cfg : :class:`~hiten.algorithms.types.configs.EventConfig`
            Configuration controlling directionality and terminal behavior.
        event_options : :class:`~hiten.algorithms.types.options.EventOptions` | None
            Runtime tuning options controlling event detection tolerances.
        **kwargs
            Additional integration options passed to the implementation.

        Returns
        -------
        :class:`~hiten.algorithms.integrators.types._Solution`
            Integration results with times, states, and derivatives when
            available. Units follow the provided ``system``.
        """
        self.validate_inputs(system, y0, t_vals)
        # Common zero-span short-circuit
        constant_sol = self._maybe_constant_solution(system, y0, t_vals)
        if constant_sol is not None:
            return constant_sol
        f = self._build_rhs_wrapper(system)

        # Event-enabled path
        if event_fn is not None:
            event_compiled = self._compile_event_function(event_fn)
            direction = 0 if event_cfg is None else int(event_cfg.direction)
            terminal = 1 if (event_cfg is None or event_cfg.terminal) else 0
            t0 = float(t_vals[0])
            tmax = float(t_vals[-1])
            xtol = float(event_options.xtol if event_options is not None else 1.0e-12)
            gtol = float(event_options.gtol if event_options is not None else 1.0e-12)
            if isinstance(system, _HamiltonianSystemProtocol):
                jac_H, clmo_H, n_dof = system.rhs_params
                hit, t_event, y_event, y_last = _DOP853._integrate_dop853_until_event_ham(
                    y0=y0,
                    t0=t0,
                    tmax=tmax,
                    A=self._A,
                    B_HIGH=self._B_HIGH,
                    C=self._C,
                    E5=self._E5,
                    E3=self._E3,
                    D=DOP853_D,
                    n_stages_extended=DOP853_N_STAGES_EXTENDED,
                    interpolator_power=DOP853_INTERPOLATOR_POWER,
                    A_full=DOP853_A,
                    C_full=DOP853_C,
                    rtol=self._rtol,
                    atol=self._atol,
                    max_step=self._max_step,
                    min_step=self._min_step,
                    order=self._p,
                    event_fn=event_compiled,
                    direction=direction,
                    terminal=terminal,
                    xtol=xtol,
                    gtol=gtol,
                    jac_H=jac_H,
                    clmo_H=clmo_H,
                    n_dof=n_dof,
                )
            else:
                hit, t_event, y_event, y_last = _DOP853._integrate_dop853_until_event(
                    f=f,
                    y0=y0,
                    t0=t0,
                    tmax=tmax,
                    A=self._A,
                    B_HIGH=self._B_HIGH,
                    C=self._C,
                    E5=self._E5,
                    E3=self._E3,
                    D=DOP853_D,
                    n_stages_extended=DOP853_N_STAGES_EXTENDED,
                    interpolator_power=DOP853_INTERPOLATOR_POWER,
                    A_full=DOP853_A,
                    C_full=DOP853_C,
                    rtol=self._rtol,
                    atol=self._atol,
                    max_step=self._max_step,
                    min_step=self._min_step,
                    order=self._p,
                    event_fn=event_compiled,
                    direction=direction,
                    terminal=terminal,
                    xtol=xtol,
                    gtol=gtol,
                )
            if hit:
                return _Solution(times=np.array([t0, t_event], dtype=np.float64), states=np.vstack([y0, y_event]))
            else:
                return _Solution(times=np.array([t0, tmax], dtype=np.float64), states=np.vstack([y0, y_last]))

        # Standard non-event path
        if isinstance(system, _HamiltonianSystemProtocol):
            jac_H, clmo_H, n_dof = system.rhs_params
            states, derivs = _DOP853._integrate_dop853_ham(
                y0=y0,
                t_eval=t_vals,
                A=self._A,
                B_HIGH=self._B_HIGH,
                C=self._C,
                E5=self._E5,
                E3=self._E3,
                D=DOP853_D,
                n_stages_extended=DOP853_N_STAGES_EXTENDED,
                interpolator_power=DOP853_INTERPOLATOR_POWER,
                A_full=DOP853_A,
                C_full=DOP853_C,
                rtol=self._rtol,
                atol=self._atol,
                max_step=self._max_step,
                min_step=self._min_step,
                order=self._p,
                jac_H=jac_H,
                clmo_H=clmo_H,
                n_dof=n_dof,
            )
        else:
            states, derivs = _DOP853._integrate_dop853(
                f=f,
                y0=y0,
                t_eval=t_vals,
                A=self._A,
                B_HIGH=self._B_HIGH,
                C=self._C,
                E5=self._E5,
                E3=self._E3,
                D=DOP853_D,
                n_stages_extended=DOP853_N_STAGES_EXTENDED,
                interpolator_power=DOP853_INTERPOLATOR_POWER,
                A_full=DOP853_A,
                C_full=DOP853_C,
                rtol=self._rtol,
                atol=self._atol,
                max_step=self._max_step,
                min_step=self._min_step,
                order=self._p,
            )
        return _Solution(times=t_vals.copy(), states=states, derivatives=derivs)

    @staticmethod
    @numba.njit(cache=False, fastmath=FASTMATH)
    def _integrate_dop853(f, y0, t_eval, A, B_HIGH, C, E5, E3, D, n_stages_extended, interpolator_power, A_full, C_full, rtol, atol, max_step, min_step, order):
        """
        Integrate until the end of the time interval.

        Parameters
        ----------
        f : callable
            The right-hand side of the differential equation.
        y0 : numpy.ndarray
            The initial condition.
        t_eval : numpy.ndarray
            The time points to evaluate the solution at.
        A : numpy.ndarray
            The Butcher tableau.
        B_HIGH : numpy.ndarray
            The high order weights.
        C : numpy.ndarray
            The nodes.
        E5 : numpy.ndarray
            The fifth order error coefficients.
        E3 : numpy.ndarray
            The third order error coefficients.
        D : numpy.ndarray
            The D matrix.
        n_stages_extended : int
            The number of stages extended.
        interpolator_power : int
            The interpolator power.
        A_full : numpy.ndarray
            The full A matrix.
        C_full : numpy.ndarray
            The full C matrix.
        rtol : float
            The relative tolerance.
        atol : float
            The absolute tolerance.
        max_step : float
            The maximum step size.
        min_step : float
            The minimum step size.
        order : int
            The order of the method.

        Returns
        -------
        numpy.ndarray
            The states at the time points.
        numpy.ndarray
            The derivatives at the time points.

        See Also
        --------
        :func:`~hiten.algorithms.integrators.rk.dop853_step_jit_kernel` : The kernel used for the step.
        """
        t0 = t_eval[0]
        tf = t_eval[-1]
        t = t0
        y = y0.copy()
        ts = List()
        ys = List()
        dys = List()
        Ks = List()
        ts.append(t)
        ys.append(y.copy())
        dys.append(f(t, y))

        # initial step heuristic
        dy0 = dys[0]
        scale0 = atol + rtol * np.abs(y)
        d0 = np.linalg.norm(y / scale0) / np.sqrt(y.size)
        d1 = np.linalg.norm(dy0 / scale0) / np.sqrt(y.size)
        h = _select_initial_step(d0, d1, min_step, max_step)

        err_prev = -1.0

        while (t - tf) * 1.0 < 0.0:
            h = _clamp_step(h, max_step, min_step)
            h = _adjust_step_to_endpoint(t, h, tf)

            y_high, y_low, err_vec, err5, err3, k = dop853_step_jit_kernel(f, t, y, h, A, B_HIGH, C, E5, E3)
            scale = _error_scale(y, y_high, rtol, atol)
            # SciPy-compatible combined error norm for DOP853
            err5_scaled = err5 / scale
            err3_scaled = err3 / scale
            err5_norm_2 = np.dot(err5_scaled, err5_scaled)
            err3_norm_2 = np.dot(err3_scaled, err3_scaled)
            if err5_norm_2 == 0.0 and err3_norm_2 == 0.0:
                err_norm = 0.0
            else:
                denom = err5_norm_2 + 0.01 * err3_norm_2
                err_norm = np.abs(h) * err5_norm_2 / np.sqrt(denom * scale.size)

            if err_norm <= 1.0:
                t_new = t + h
                y_new = y_high
                ts.append(t_new)
                ys.append(y_new.copy())
                f_new = f(t_new, y_new)
                dys.append(f_new)
                Ks.append(k)
                t = t_new
                y = y_new

                h = h * _pi_accept_factor(err_norm, err_prev, order)
                err_prev = err_norm
            else:
                h = h * _pi_reject_factor(err_norm, order)
                h = _clamp_step(h, max_step, min_step)

        # Convert lists to arrays
        n_nodes = len(ts)
        dim = y0.size
        ts_arr = np.empty(n_nodes, dtype=np.float64)
        ys_arr = np.empty((n_nodes, dim), dtype=np.float64)
        dys_arr = np.empty_like(ys_arr)
        for i in range(n_nodes):
            ts_arr[i] = ts[i]
            ys_arr[i, :] = ys[i]
            dys_arr[i, :] = dys[i]

        # SciPy-like dense output for DOP853 using D matrix (build F per segment)
        m = t_eval.size
        y_out = np.empty((m, dim), dtype=np.float64)
        last_j = -1
        # cache F for last segment: shape (interpolator_power, dim)
        F_cache = np.empty((interpolator_power, dim), dtype=np.float64)
        for idx in range(m):
            t_q = t_eval[idx]
            j = np.searchsorted(ts_arr, t_q, side='right') - 1
            if j < 0:
                j = 0
            if j > n_nodes - 2:
                j = n_nodes - 2
            t0s = ts_arr[j]
            t1s = ts_arr[j + 1]
            hseg = t1s - t0s
            if hseg == 0.0:
                # Degenerate segment: no progress; use left state directly
                y_out[idx, :] = ys_arr[j, :]
                continue
            x = (t_q - t0s) / hseg
            if j != last_j:
                # Build dense cache for this segment using helper
                y_old = ys_arr[j]
                y_new = ys_arr[j + 1]
                f_old = dys_arr[j]
                f_new = dys_arr[j + 1]
                Kseg = Ks[j]
                F_cache = _dop853_build_dense_cache(
                    f=f,
                    t_old=t0s,
                    y_old=y_old,
                    f_old=f_old,
                    y_new=y_new,
                    f_new=f_new,
                    hseg=hseg,
                    Kseg=Kseg,
                    A_full=A_full,
                    C_full=C_full,
                    D=D,
                    n_stages_extended=n_stages_extended,
                    interpolator_power=interpolator_power,
                )
                last_j = j
            # Evaluate dense interpolant
            y_old = ys_arr[j]
            y_out[idx, :] = _dop853_eval_dense(y_old, F_cache, interpolator_power, x)

        derivs_out = np.empty_like(y_out)
        for idx in range(m):
            derivs_out[idx, :] = f(t_eval[idx], y_out[idx, :])

        return y_out, derivs_out

    @staticmethod
    @numba.njit(cache=False, fastmath=FASTMATH)
    def _integrate_dop853_ham(y0, t_eval, A, B_HIGH, C, E5, E3, D, n_stages_extended, interpolator_power, A_full, C_full, rtol, atol, max_step, min_step, order, jac_H, clmo_H, n_dof):
        """Hamiltonian variant: integrate until the end of the interval without closures.

        Parameters mirror _integrate_dop853, but omit f and add (jac_H, clmo_H, n_dof).
        """
        t0 = t_eval[0]
        tf = t_eval[-1]
        t = t0
        y = y0.copy()
        ts = List()
        ys = List()
        dys = List()
        Ks = List()
        ts.append(t)
        ys.append(y.copy())
        dys.append(_hamiltonian_rhs(y, jac_H, clmo_H, n_dof))

        # initial step heuristic
        dy0 = dys[0]
        scale0 = atol + rtol * np.abs(y)
        d0 = np.linalg.norm(y / scale0) / np.sqrt(y.size)
        d1 = np.linalg.norm(dy0 / scale0) / np.sqrt(y.size)
        h = _select_initial_step(d0, d1, min_step, max_step)

        err_prev = -1.0

        while (t - tf) * 1.0 < 0.0:
            h = _clamp_step(h, max_step, min_step)
            h = _adjust_step_to_endpoint(t, h, tf)

            y_high, y_low, err_vec, err5, err3, k = dop853_step_ham_jit_kernel(t, y, h, A, B_HIGH, C, E5, E3, jac_H, clmo_H, n_dof)
            scale = _error_scale(y, y_high, rtol, atol)
            # SciPy-compatible combined error norm for DOP853
            err5_scaled = err5 / scale
            err3_scaled = err3 / scale
            err5_norm_2 = np.dot(err5_scaled, err5_scaled)
            err3_norm_2 = np.dot(err3_scaled, err3_scaled)
            if err5_norm_2 == 0.0 and err3_norm_2 == 0.0:
                err_norm = 0.0
            else:
                denom = err5_norm_2 + 0.01 * err3_norm_2
                err_norm = np.abs(h) * err5_norm_2 / np.sqrt(denom * scale.size)

            if err_norm <= 1.0:
                t_new = t + h
                y_new = y_high
                ts.append(t_new)
                ys.append(y_new.copy())
                f_new = _hamiltonian_rhs(y_new, jac_H, clmo_H, n_dof)
                dys.append(f_new)
                Ks.append(k)
                t = t_new
                y = y_new

                h = h * _pi_accept_factor(err_norm, err_prev, order)
                err_prev = err_norm
            else:
                h = h * _pi_reject_factor(err_norm, order)
                h = _clamp_step(h, max_step, min_step)

        # Convert lists to arrays
        n_nodes = len(ts)
        dim = y0.size
        ts_arr = np.empty(n_nodes, dtype=np.float64)
        ys_arr = np.empty((n_nodes, dim), dtype=np.float64)
        dys_arr = np.empty_like(ys_arr)
        for i in range(n_nodes):
            ts_arr[i] = ts[i]
            ys_arr[i, :] = ys[i]
            dys_arr[i, :] = dys[i]

        # Dense output reconstruction using D matrix (same pattern as generic, but k-evals use Hamiltonian RHS)
        m = t_eval.size
        y_out = np.empty((m, dim), dtype=np.float64)
        last_j = -1
        F_cache = np.empty((interpolator_power, dim), dtype=np.float64)
        for idx in range(m):
            t_q = t_eval[idx]
            j = np.searchsorted(ts_arr, t_q, side='right') - 1
            if j < 0:
                j = 0
            if j > n_nodes - 2:
                j = n_nodes - 2
            t0s = ts_arr[j]
            t1s = ts_arr[j + 1]
            hseg = t1s - t0s
            if hseg == 0.0:
                y_out[idx, :] = ys_arr[j, :]
                continue
            x = (t_q - t0s) / hseg
            if j != last_j:
                # Build dense cache for this segment using Hamiltonian helper
                y_old = ys_arr[j]
                y_new = ys_arr[j + 1]
                f_old = dys_arr[j]
                f_new = dys_arr[j + 1]
                Kseg = Ks[j]
                F_cache = _dop853_build_dense_cache_ham(
                    t_old=t0s,
                    y_old=y_old,
                    f_old=f_old,
                    y_new=y_new,
                    f_new=f_new,
                    hseg=hseg,
                    Kseg=Kseg,
                    A_full=A_full,
                    C_full=C_full,
                    D=D,
                    n_stages_extended=n_stages_extended,
                    interpolator_power=interpolator_power,
                    jac_H=jac_H,
                    clmo_H=clmo_H,
                    n_dof=n_dof,
                )
                last_j = j
            # Evaluate (always)
            y_old = ys_arr[j]
            y_out[idx, :] = _dop853_eval_dense(y_old, F_cache, interpolator_power, x)

        derivs_out = np.empty_like(y_out)
        for idx in range(m):
            derivs_out[idx, :] = _hamiltonian_rhs(y_out[idx, :], jac_H, clmo_H, n_dof)

        return y_out, derivs_out

    @staticmethod
    @numba.njit(cache=False, fastmath=FASTMATH)
    def _integrate_dop853_until_event(
        f, y0, t0, tmax, A, B_HIGH, C, E5, E3, D, n_stages_extended, interpolator_power, A_full, C_full, rtol, atol, max_step, min_step, order,
        event_fn, direction, terminal, xtol, gtol,
    ):
        """
        Integrate until an event is detected.

        Parameters
        ----------
        f : callable
            The right-hand side of the differential equation.
        y0 : numpy.ndarray
            The initial condition.
        t0 : float
            The initial time.
        tmax : float
            The maximum time.
        A : numpy.ndarray
            The Butcher tableau.
        B_HIGH : numpy.ndarray
            The high order weights.
        C : numpy.ndarray
            The nodes.
        E5 : numpy.ndarray
            The fifth order error coefficients.
        E3 : numpy.ndarray
            The third order error coefficients.
        D : numpy.ndarray
            The D matrix.
        n_stages_extended : int
            The number of stages extended.
        interpolator_power : int
            The interpolator power.
        A_full : numpy.ndarray
            The full A matrix.
        C_full : numpy.ndarray
            The full C matrix.
        rtol : float
            The relative tolerance.
        atol : float
            The absolute tolerance.
        max_step : float
            The maximum step size.
        min_step : float
            The minimum step size.
        order : int
            The order of the integrator.
        event_fn : callable
            The event function.
        direction : int
            The direction of the event.
        terminal : bool
            Whether to terminate at the event.
        xtol : float
            The tolerance for the event.

        Returns
        -------
        bool
            Whether an event was detected.
        float
            The time of the event.
        numpy.ndarray
            The state at the event.
        numpy.ndarray
            The state at the end of the integration.
        """
        t = t0
        y = y0.copy()
        # initial derivative and event value
        f_curr = f(t, y)
        g_prev = event_fn(t, y)

        h = min_step
        # initial step heuristic akin to main driver
        scale0 = atol + rtol * np.abs(y)
        d0 = np.linalg.norm(y / scale0) / np.sqrt(y.size)
        d1 = np.linalg.norm(f_curr / scale0) / np.sqrt(y.size)
        h = _select_initial_step(d0, d1, min_step, max_step)

        err_prev = -1.0

        while (t - tmax) * 1.0 < 0.0:
            h = _clamp_step(h, max_step, min_step)
            h = _adjust_step_to_endpoint(t, h, tmax)

            y_high, y_low, err_vec, err5, err3, K = dop853_step_jit_kernel(f, t, y, h, A, B_HIGH, C, E5, E3)
            scale = _error_scale(y, y_high, rtol, atol)
            err5_scaled = err5 / scale
            err3_scaled = err3 / scale
            err5_norm_2 = np.dot(err5_scaled, err5_scaled)
            err3_norm_2 = np.dot(err3_scaled, err3_scaled)
            if err5_norm_2 == 0.0 and err3_norm_2 == 0.0:
                err_norm = 0.0
            else:
                denom = err5_norm_2 + 0.01 * err3_norm_2
                err_norm = np.abs(h) * err5_norm_2 / np.sqrt(denom * scale.size)

            if err_norm <= 1.0:
                # accept
                t_new = t + h
                y_new = y_high
                f_new = f(t_new, y_new)

                # event check
                g_new = event_fn(t_new, y_new)
                crossed = _event_crossed(g_prev, g_new, direction)
                if crossed:
                    t_hit, y_hit = _dop853_refine_in_step(f, event_fn, t, y, f_curr, t_new, y_new, f_new, h, K, A_full, C_full, D, n_stages_extended, interpolator_power, direction, xtol, gtol)
                    return True, t_hit, y_hit, y_new

                # no crossing, advance
                t = t_new
                y = y_new
                f_curr = f_new
                g_prev = g_new

                h = h * _pi_accept_factor(err_norm, err_prev, order)
                err_prev = err_norm
            else:
                h = h * _pi_reject_factor(err_norm, order)
                h = _clamp_step(h, max_step, min_step)

        return False, t, y, y

    @staticmethod
    @numba.njit(cache=False, fastmath=FASTMATH)
    def _integrate_dop853_until_event_ham(
        y0, t0, tmax, A, B_HIGH, C, E5, E3, D, n_stages_extended, interpolator_power, A_full, C_full, rtol, atol, max_step, min_step, order,
        event_fn, direction, terminal, xtol, gtol, jac_H, clmo_H, n_dof,
    ):
        """
        Hamiltonian variant: integrate until an event is detected using DOP853.
        """
        t = t0
        y = y0.copy()
        f_curr = _hamiltonian_rhs(y, jac_H, clmo_H, n_dof)
        g_prev = event_fn(t, y)

        # initial step heuristic
        scale0 = atol + rtol * np.abs(y)
        d0 = np.linalg.norm(y / scale0) / np.sqrt(y.size)
        d1 = np.linalg.norm(f_curr / scale0) / np.sqrt(y.size)
        h = _select_initial_step(d0, d1, min_step, max_step)

        err_prev = -1.0

        while (t - tmax) * 1.0 < 0.0:
            h = _clamp_step(h, max_step, min_step)
            h = _adjust_step_to_endpoint(t, h, tmax)

            y_high, y_low, err_vec, err5, err3, K = dop853_step_ham_jit_kernel(t, y, h, A, B_HIGH, C, E5, E3, jac_H, clmo_H, n_dof)
            scale = _error_scale(y, y_high, rtol, atol)
            err5_scaled = err5 / scale
            err3_scaled = err3 / scale
            err5_norm_2 = np.dot(err5_scaled, err5_scaled)
            err3_norm_2 = np.dot(err3_scaled, err3_scaled)
            if err5_norm_2 == 0.0 and err3_norm_2 == 0.0:
                err_norm = 0.0
            else:
                denom = err5_norm_2 + 0.01 * err3_norm_2
                err_norm = np.abs(h) * err5_norm_2 / np.sqrt(denom * scale.size)

            if err_norm <= 1.0:
                # accept
                t_new = t + h
                y_new = y_high
                f_new = _hamiltonian_rhs(y_new, jac_H, clmo_H, n_dof)

                # event check
                g_new = event_fn(t_new, y_new)
                crossed = _event_crossed(g_prev, g_new, direction)
                if crossed:
                    t_hit, y_hit = _dop853_refine_in_step_ham(event_fn, t, y, f_curr, t_new, y_new, f_new, h, K, A_full, C_full, D, n_stages_extended, interpolator_power, direction, xtol, gtol, jac_H, clmo_H, n_dof)
                    return True, t_hit, y_hit, y_new

                # no crossing, advance
                t = t_new
                y = y_new
                f_curr = f_new
                g_prev = g_new

                h = h * _pi_accept_factor(err_norm, err_prev, order)
                err_prev = err_norm
            else:
                h = h * _pi_reject_factor(err_norm, order)
                h = _clamp_step(h, max_step, min_step)

        return False, t, y, y


class FixedRK:
    """Implement a factory class for creating fixed-step Runge-Kutta integrators.
    
    This factory provides convenient access to fixed-step Runge-Kutta methods
    of different orders. The available orders are 4, 6, and 8.
    
    Examples
    --------
    >>> rk4 = FixedRK(order=4)
    >>> rk6 = FixedRK(order=6)
    >>> rk8 = FixedRK(order=8)
    """
    _map = {4: _RK4, 6: _RK6, 8: _RK8}
    def __new__(cls, order=4, **opts):
        """Create a fixed-step Runge-Kutta integrator of specified order.
        
        Parameters
        ----------
        order : int, default 4
            Order of the Runge-Kutta method. Must be 4, 6, or 8.
        **opts
            Additional options passed to the integrator constructor.
            
        Returns
        -------
        :class:`~hiten.algorithms.integrators.rk._FixedStepRK`
            A fixed-step Runge-Kutta integrator instance.
            
        Raises
        ------
        ValueError
            If the specified order is not supported.
        """
        if order not in cls._map:
            raise ValueError("RK order must be 4, 6, or 8")
        return cls._map[order](**opts)


class AdaptiveRK:
    """Implement a factory class for creating adaptive step-size Runge-Kutta integrators.
    
    This factory provides convenient access to adaptive step-size Runge-Kutta
    methods. The available orders are 5 (Dormand-Prince 5(4)) and 8 (Dormand-Prince 8(5,3)).
    
    Examples
    --------
    >>> rk45 = AdaptiveRK(order=5)
    >>> dop853 = AdaptiveRK(order=8)
    """
    _map = {5: _RK45, 8: _DOP853}
    def __new__(cls, order=5, **opts):
        """Create an adaptive step-size Runge-Kutta integrator of specified order.
        
        Parameters
        ----------
        order : int, default 5
            Order of the Runge-Kutta method. Must be 5 or 8.
        **opts
            Additional options passed to the integrator constructor.
            
        Returns
        -------
        :class:`~hiten.algorithms.integrators.rk._AdaptiveStepRK`
            An adaptive step-size Runge-Kutta integrator instance.
            
        Raises
        ------
        ValueError
            If the specified order is not supported.
        """
        if order not in cls._map:
            raise ValueError("Adaptive RK order not supported")
        return cls._map[order](**opts)

class RungeKutta:
    """Implement a factory class for creating Runge-Kutta integrators.
    
    This factory provides convenient access to Runge-Kutta integrators of different orders.
    """
    _map = {4: FixedRK, 45: AdaptiveRK, 6: FixedRK, 8: FixedRK, 853: AdaptiveRK}
    def __new__(cls, order=4, **opts):
        """Create a Runge-Kutta integrator of specified order.
        
        Parameters
        ----------
        order : int, default 4
            Order of the Runge-Kutta method. Must be 4, 45, 6, 8, or 853.
        **opts
            Additional options passed to the integrator constructor.
            
        Returns
        -------
        :class:`~hiten.algorithms.integrators.rk.FixedRK` or
        :class:`~hiten.algorithms.integrators.rk.AdaptiveRK`
            A Runge-Kutta integrator instance.
        """
        if order not in cls._map:
            raise ValueError("Runge-Kutta order not supported")
        
        factory_class = cls._map[order]
        if order in [45, 853]:
            if order == 45:
                _order = 5
            else:
                _order = 8
        else:
            _order = order

        return factory_class(order=_order, **opts)