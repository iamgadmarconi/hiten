r"""
hiten.algorithms.integrators.rk
=========================

Explicit Runge-Kutta integrators used throughout the project.  Both fixed
and adaptive step-size variants are provided together with small convenience
factories that select an appropriate implementation given the desired
formal order of accuracy.

The concrete schemes implemented are:

* Fixed step: RK4, RK6, RK8
* Adaptive embedded: RK45 (Dormand-Prince) and DOP853 (Dormand-Prince 8(5,3))

Internally the module also defines helper routines to evaluate Hamiltonian
vector fields with :pyfunc:`numba.njit` and to wrap right-hand side (RHS)
callables into a uniform signature accepted by the integrators.

References
----------
Hairer, E.; Nørsett, S.; Wanner, G. (1993). "Solving Ordinary Differential
Equations I".

Dormand, J. R.; Prince, P. J. (1980). "A family of embedded Runge-Kutta
formulas".
"""

import inspect
from typing import Callable, Optional

import numpy as np
from numba import njit

from hiten.algorithms.dynamics.base import _DynamicalSystem
from hiten.algorithms.dynamics.hamiltonian import (HamiltonianSystem,
                                                   _hamiltonian_rhs)
from hiten.algorithms.integrators.base import Integrator, Solution
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
from hiten.utils.config import FASTMATH, TOL
from hiten.utils.log_config import logger


class _RungeKuttaBase(Integrator):
    r"""
    Shared functionality of explicit Runge-Kutta schemes.

    The class stores a Butcher tableau and provides a single low level helper
    :pyfunc:`_rk_embedded_step` that advances one macro time step and, when a
    second set of weights is available, returns an error estimate suitable
    for adaptive step-size control.

    Attributes
    ----------
    _A : numpy.ndarray of shape (s, s)
        Strictly lower triangular array of stage coefficients :math:`a_{ij}`.
    _B_HIGH : numpy.ndarray of shape (s,)
        Weights of the high order solution.
    _B_LOW : numpy.ndarray or None
        Weights of the lower order solution, optional.  When *None* no error
        estimate is produced and :pyfunc:`_rk_embedded_step` falls back to
        the high order result for both outputs.
    _C : numpy.ndarray of shape (s,)
        Nodes :math:`c_i` measured in units of the step size.
    _p : int
        Formal order of accuracy of the high order scheme.

    Notes
    -----
    The class is **not** intended to be used directly.  Concrete subclasses
    define the specific coefficients and expose a public interface compliant
    with :class:`hiten.algorithms.integrators.base.Integrator`.
    """

    _A: np.ndarray = None
    _B_HIGH: np.ndarray = None
    _B_LOW: Optional[np.ndarray] = None
    _C: np.ndarray = None
    _p: int = 0

    def _rk_embedded_step(self, f, t, y, h):
        s = self._B_HIGH.size
        k = np.empty((s, y.size), dtype=np.float64)

        k[0] = f(t, y)
        for i in range(1, s):
            y_stage = y.copy()
            for j in range(i):
                a_ij = self._A[i, j]
                if a_ij != 0.0:
                    y_stage += h * a_ij * k[j]
            k[i] = f(t + self._C[i] * h, y_stage)

        y_high = y + h * np.dot(self._B_HIGH, k)

        if self._B_LOW is not None:
            y_low = y + h * np.dot(self._B_LOW, k)
        else:
            y_low = y_high.copy()
        err_vec = y_high - y_low
        return y_high, y_low, err_vec


class _FixedStepRK(_RungeKuttaBase):
    r"""Explicit fixed-step Runge-Kutta scheme.

    Parameters
    ----------
    name : str
        Human readable identifier of the scheme (e.g. ``"RK4"``).
    A, B, C : numpy.ndarray
        Butcher tableau as returned by :pymod:`hiten.algorithms.integrators.coefficients.*`.
    order : int
        Formal order of accuracy :math:`p` of the method.
    **options
        Additional keyword options forwarded to the base :class:`Integrator`.

    Notes
    -----
    The step size is assumed to be **constant** and is inferred from the
    spacing of the *t_vals* array supplied to :pyfunc:`integrate`.
    """

    def __init__(self, name: str, A: np.ndarray, B: np.ndarray, C: np.ndarray, order: int, **options):
        self._A = A
        self._B_HIGH = B
        self._B_LOW = None
        self._C = C
        self._p = order
        super().__init__(name, **options)

    @property
    def order(self) -> int:
        return self._p

    def integrate(
        self,
        system: _DynamicalSystem,
        y0: np.ndarray,
        t_vals: np.ndarray,
        **kwargs,
    ) -> Solution:
        self.validate_inputs(system, y0, t_vals)

        rhs_wrapped = _build_rhs_wrapper(system)

        # The RHS that comes in may already embed the intended time direction.
        # Therefore do *not* apply any additional sign here - simply forward
        # the call to the wrapped system RHS.
        def f(t, y):
            return rhs_wrapped(t, y)

        traj = np.empty((t_vals.size, y0.size), dtype=np.float64)
        derivs = np.empty_like(traj)

        # Initial state and derivative
        traj[0] = y0.copy()
        derivs[0] = f(t_vals[0], y0)

        for idx in range(t_vals.size - 1):
            t_n = t_vals[idx]
            h = t_vals[idx + 1] - t_n
            y_n = traj[idx]

            # Perform RK step and obtain high-order solution
            y_high, _, _ = self._rk_embedded_step(f, t_n, y_n, h)
            traj[idx + 1] = y_high

            # Derivative at the new time point (needed for Hermite interpolation)
            derivs[idx + 1] = f(t_vals[idx + 1], y_high)

        return Solution(times=t_vals.copy(), states=traj, derivatives=derivs)


class RK4(_FixedStepRK):
    def __init__(self, **opts):
        super().__init__("RK4", RK4_A, RK4_B, RK4_C, 4, **opts)


class RK6(_FixedStepRK):
    def __init__(self, **opts):
        super().__init__("RK6", RK6_A, RK6_B, RK6_C, 6, **opts)


class RK8(_FixedStepRK):
    def __init__(self, **opts):
        super().__init__("RK8", RK8_A, RK8_B, RK8_C, 8, **opts)


class _AdaptiveStepRK(_RungeKuttaBase):
    r"""
    Embedded adaptive Runge-Kutta integrator with PI controller.

    The class implements proportional-integral (PI) step-size control using
    the error estimates returned by :pyfunc:`_rk_embedded_step`.  Two safety
    factors are used:

    * *rtol*, *atol*  - user requested relative and absolute tolerances
    * :pyattr:`SAFETY` - hard coded damping on the acceptance criterion.

    Parameters
    ----------
    name : str, default "AdaptiveRK"
        Identifier passed to the :class:`Integrator` base class.
    rtol, atol : float, optional
        Relative and absolute error tolerances.  Defaults are read from
        :pydata:`hiten.utils.config.TOL`.
    max_step : float, optional
        Upper bound on the step size.  :math:`\infty` by default.
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

    @property
    def order(self) -> int:
        return self._p

    def integrate(self, system: _DynamicalSystem, y0: np.ndarray, t_vals: np.ndarray, **kwargs) -> Solution:
        r"""
        Integrate a dynamical system using an adaptive Runge-Kutta method.
        """
        self.validate_inputs(system, y0, t_vals)

        debug = self.options.get("debug", False)

        rhs_wrapped = _build_rhs_wrapper(system)

        # The RHS that comes in may already embed the intended time direction.
        # Therefore do *not* apply any additional sign here - simply forward
        # the call to the wrapped system RHS.
        def f(t, y):
            return rhs_wrapped(t, y)

        t_span = (t_vals[0], t_vals[-1])

        t = t_span[0]
        y = np.ascontiguousarray(y0, dtype=np.float64)
        ts, ys, dys = [t], [y.copy()], [f(t, y)]

        h = self._select_initial_step(f, t, y, t_span[1])
        h = max(h, 1e-4 * abs(t_span[1] - t_span[0]))

        err_prev = None
        accepted_steps = 0
        step_rejected = False

        while (t - t_span[1]) * 1 < 0:
            min_step = 10 * abs(np.nextafter(t, t + 1) - t)
            if h > self._max_step:
                h = self._max_step
            if t + 1 * h > t_span[1]:
                h = abs(t_span[1] - t)

            y_high, y_low, err_vec = self._rk_embedded_step(f, t, y, h)
            scale = self._atol + self._rtol * np.maximum(np.abs(y), np.abs(y_high))
            err_norm = np.linalg.norm(err_vec / scale) / np.sqrt(err_vec.size)

            if err_norm <= 1.0:
                t_new = t + 1 * h
                y_new = y_high

                ts.append(t_new)
                ys.append(y_new.copy())
                dys.append(f(t_new, y_new))
                t, y = t_new, y_new

                beta = 1.0 / (self._p + 1)
                alpha = 0.4 * beta
                if err_prev is None:
                    factor = self.SAFETY * err_norm ** (-beta)
                else:
                    factor = self.SAFETY * err_norm ** (-beta) * err_prev ** alpha
                factor = np.clip(factor, self.MIN_FACTOR, self.MAX_FACTOR)
                if step_rejected:
                    factor = min(1.0, factor)

                h *= factor
                step_rejected = False
                err_prev = err_norm

                accepted_steps += 1
                if debug and accepted_steps % 1000 == 0:
                    logger.info(f"[AdaptiveRK] step {accepted_steps:7d}: t={t:.6g} h={h:.3g} err={err_norm:.3g}")
            else:
                factor = max(self.MIN_FACTOR, self.SAFETY * err_norm ** (-self._err_exp))
                h *= factor
                step_rejected = True
                if abs(h) < min_step:
                    raise RuntimeError("Step size underflow in adaptive RK integrator.")

        # Convert lists to arrays for vectorized interpolation
        ts = np.array(ts)
        ys = np.array(ys)
        dys = np.array(dys)
        t_eval = np.asarray(t_vals)

        # Vectorized Hermite interpolation for all t_eval
        indices = np.searchsorted(ts, t_eval, side='right') - 1
        indices = np.clip(indices, 0, len(ts) - 2)

        t0s = ts[indices]
        t1s = ts[indices + 1]
        y0s = ys[indices]
        y1s = ys[indices + 1]
        dy0s = dys[indices]
        dy1s = dys[indices + 1]
        taus = (t_eval - t0s) / (t1s - t0s)

        tau2 = taus ** 2
        tau3 = tau2 * taus
        h = t1s - t0s
        h00 = 2 * tau3 - 3 * tau2 + 1
        h10 = tau3 - 2 * tau2 + taus
        h01 = -2 * tau3 + 3 * tau2
        h11 = tau3 - tau2

        y_eval = (
            h00[:, None] * y0s +
            h10[:, None] * h[:, None] * dy0s +
            h01[:, None] * y1s +
            h11[:, None] * h[:, None] * dy1s
        )

        # Also compute derivatives at t_eval if needed
        # For now, use f(t_eval, y_eval) for derivatives
        derivs_out = np.array([f(t, y) for t, y in zip(t_eval, y_eval)])

        return Solution(times=t_eval.copy(), states=y_eval, derivatives=derivs_out)

    def _select_initial_step(self, f, t0, y0, tf):
        r"""
        Choose an initial step size following Hairer et al. / SciPy heuristics.

        The strategy tries to obtain a first step that keeps the truncation
        error around the requested tolerance.  A too-small *h* makes the
        adaptive solver crawl, while a too-large *h* triggers many rejected
        steps.  This implementation closely mirrors SciPy's `_initial_step`.
        """
        # Compute derivative at the initial point.
        dy0 = f(t0, y0)

        # Weighted norms used by the error control.
        scale = self._atol + self._rtol * np.abs(y0)
        d0 = np.linalg.norm(y0 / scale) / np.sqrt(y0.size)
        d1 = np.linalg.norm(dy0 / scale) / np.sqrt(y0.size)

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        # Try a tentative first step and look at the curvature.
        y1 = y0 + h0 * dy0
        dy1 = f(t0 + h0, y1)

        d2 = np.linalg.norm((dy1 - dy0) / scale) / np.sqrt(y0.size) / h0

        if max(d1, d2) <= 1e-15:
            h1 = max(1e-6, 0.1 * abs(tf - t0))
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / (self._p + 1))

        h = min(100 * h0, h1)

        # Respect user-supplied step limits.
        h = min(h, abs(tf - t0), self._max_step)
        h = max(h, self._min_step)

        return h

    def _update_factor(self, err_norm):
        return np.clip(self.SAFETY * err_norm ** (-self._err_exp), self.MIN_FACTOR, self.MAX_FACTOR)


class RK45(_AdaptiveStepRK):
    _A = RK45_A
    _B_HIGH = RK45_B_HIGH
    _B_LOW = None
    _C = RK45_C
    _p = 5
    _err_exp = 1.0 / 5.0
    _E = RK45_E

    def __init__(self, **opts):
        super().__init__("RK45", **opts)

    def _rk_embedded_step(self, f, t, y, h):
        s = 6
        k = np.empty((s + 1, y.size))
        k[0] = f(t, y)
        for i in range(1, s):
            y_stage = y + h * (self._A[i, :i] @ k[:i])
            k[i] = f(t + self._C[i] * h, y_stage)
        y_high = y + h * (self._B_HIGH @ k[:s])
        k[s] = f(t + h, y_high)
        err_vec = h * (k.T @ self._E)
        y_low = y_high - err_vec
        return y_high, y_low, err_vec


class DOP853(_AdaptiveStepRK):
    _A = DOP853_A[:DOP853_N_STAGES, :DOP853_N_STAGES]
    _B_HIGH = DOP853_B[:DOP853_N_STAGES]
    _B_LOW = None
    _C = DOP853_C[:DOP853_N_STAGES]

    _p = 8
    _err_exp = 1.0 / _p

    _E3 = DOP853_E3
    _E5 = DOP853_E5
    _N_STAGES = DOP853_N_STAGES

    def __init__(self, **opts):
        super().__init__("DOP853", **opts)

    def _rk_embedded_step(self, f, t, y, h):
        s = self._N_STAGES
        k = np.empty((s + 1, y.size), dtype=np.float64)
        k[0] = f(t, y)
        for i in range(1, s):
            y_stage = y.copy()
            for j in range(i):
                a_ij = self._A[i, j]
                if a_ij != 0.0:
                    y_stage += h * a_ij * k[j]
            k[i] = f(t + self._C[i] * h, y_stage)
        y_high = y.copy()
        for j in range(s):
            b_j = self._B_HIGH[j]
            if b_j != 0.0:
                y_high += h * b_j * k[j]
        k[s] = f(t + h, y_high)
        err_vec = self._estimate_error(k, h)
        y_low = y_high - err_vec
        return y_high, y_low, err_vec

    def _estimate_error(self, K, h):
        err5 = np.dot(K.T, self._E5)
        err3 = np.dot(K.T, self._E3)
        denom = np.hypot(np.abs(err5), 0.1 * np.abs(err3))
        correction_factor = np.ones_like(err5)
        mask = denom > 0
        correction_factor[mask] = np.abs(err5[mask]) / denom[mask]
        return h * err5 * correction_factor

    def _estimate_error_norm(self, K, h, scale):
        err5 = np.dot(K.T, self._E5) / scale
        err3 = np.dot(K.T, self._E3) / scale
        err5_norm_2 = np.linalg.norm(err5) ** 2
        err3_norm_2 = np.linalg.norm(err3) ** 2
        if err5_norm_2 == 0 and err3_norm_2 == 0:
            return 0.0
        denom = err5_norm_2 + 0.01 * err3_norm_2
        return np.abs(h) * err5_norm_2 / np.sqrt(denom * len(scale))


class RungeKutta:
    _map = {4: RK4, 6: RK6, 8: RK8}
    def __new__(cls, order=4, **opts):
        if order not in cls._map:
            raise ValueError("RK order must be 4, 6, or 8")
        return cls._map[order](**opts)

class AdaptiveRK:
    _map = {5: RK45, 8: DOP853}
    def __new__(cls, order=5, **opts):
        if order not in cls._map:
            raise ValueError("Adaptive RK order not supported")
        return cls._map[order](**opts)


def _build_rhs_wrapper(system: _DynamicalSystem) -> Callable[[float, np.ndarray], np.ndarray]:
    r"""
    Return a JIT friendly wrapper around :pyfunc:`hiten.system.rhs`.

    The dynamical systems implemented in the code base expose their vector
    field either as ``rhs(t, y)`` or, for autonomous systems, as ``rhs(y)``.
    The integrator layer expects the non autonomous signature and therefore
    needs to adapt the call site on the fly. This helper inspects the right
    hand side via :pyfunc:`inspect.signature` and generates a small
    :pyfunc:`numba.njit` compiled closure with the correct arity.

    Parameters
    ----------
    system : _DynamicalSystem
        Instance providing the original vector field.

    Returns
    -------
    Callable[[float, numpy.ndarray], numpy.ndarray]
        Function accepting the full ``(t, y)`` signature required by the
        integrators.

    Raises
    ------
    ValueError
        If the detected signature contains neither one nor two positional
        arguments.
    """

    rhs_func = system.rhs

    sig = inspect.signature(rhs_func)
    n_params = len(sig.parameters)

    # Case 1: Function already accepts (t, y) – use directly.
    if n_params >= 2:
        return rhs_func

    # Case 2: Autonomous system with signature rhs(y).  Wrap to inject time argument.
    if n_params == 1:

        def _rhs_one(t, y):
            return rhs_func(y)

        return _rhs_one

    raise ValueError(
        f"Unsupported rhs signature with {n_params} parameters. "
        "Only (t, y) or (y,) are currently supported."
    )
