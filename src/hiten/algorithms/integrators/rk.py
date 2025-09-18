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
from abc import abstractmethod
from typing import Callable, Optional

import numba
import numpy as np
from numba.typed import List

from hiten.algorithms.dynamics.base import _DynamicalSystem
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
from hiten.algorithms.utils.config import FASTMATH, TOL


@numba.njit(cache=False, fastmath=FASTMATH)
def rk_embedded_step_jit_kernel(f, t, y, h, A, B_HIGH, B_LOW, C, has_b_low):
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


class _RungeKuttaBase(_Integrator):
    """Provide shared functionality of explicit Runge-Kutta schemes.

    The class stores a Butcher tableau and provides a single low level helper
    :func:`~hiten.algorithms.integrators.rk._RungeKuttaBase._rk_embedded_step` that advances one macro time step and, when a
    second set of weights is available, returns an error estimate suitable
    for adaptive step-size control.

    Attributes
    ----------
    _A : numpy.ndarray of shape (s, s)
        Strictly lower triangular array of stage coefficients a_ij.
    _B_HIGH : numpy.ndarray of shape (s,)
        Weights of the high order solution.
    _B_LOW : numpy.ndarray or None
        Weights of the lower order solution, optional.  When *None* no error
        estimate is produced and :func:`~hiten.algorithms.integrators.rk.rk_embedded_step_jit_kernel` falls back to
        the high order result for both outputs.
    _C : numpy.ndarray of shape (s,)
        Nodes c_i measured in units of the step size.
    _p : int
        Formal order of accuracy of the high order scheme.

    Notes
    -----
    The class is **not** intended to be used directly.  Concrete subclasses
    define the specific coefficients and expose a public interface compliant
    with :class:`~hiten.algorithms.integrators.base._Integrator`.
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
        system: _DynamicalSystem,
        y0: np.ndarray,
        t_vals: np.ndarray,
        **kwargs,
    ) -> _Solution:
        """Integrate a dynamical system using a fixed-step Runge-Kutta method."""
        self.validate_inputs(system, y0, t_vals)

        f = _build_rhs_wrapper(system)

        has_b_low = self._B_LOW is not None
        B_LOW_arr = self._B_LOW if has_b_low else np.empty(0, dtype=np.float64)
        states, derivs = _FixedStepRK._integrate_fixed_rk(
            f, y0, t_vals, self._A, self._B_HIGH, B_LOW_arr, self._C, has_b_low
        )
        return _Solution(times=t_vals.copy(), states=states, derivatives=derivs)

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
        y_high, y_low, err_vec, _ = rk45_step_jit_kernel(f, t, y, h, self._A, self._B_HIGH, self._C, self._E)
        return y_high, y_low, err_vec

    def integrate(self, system: _DynamicalSystem, y0: np.ndarray, t_vals: np.ndarray, **kwargs) -> _Solution:
        """Adaptive integration fully in Numba for RK45 with Hermite interpolation."""
        self.validate_inputs(system, y0, t_vals)
        f = _build_rhs_wrapper(system)
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

        # initial step selection (simple heuristic)
        dy0 = dys[0]
        scale0 = atol + rtol * np.abs(y)
        d0 = np.linalg.norm(y / scale0) / np.sqrt(y.size)
        d1 = np.linalg.norm(dy0 / scale0) / np.sqrt(y.size)
        h = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
        if h > max_step:
            h = max_step
        if h < min_step:
            h = min_step

        err_prev = -1.0
        SAFETY = 0.9
        MIN_FACTOR = 0.2
        MAX_FACTOR = 10.0
        err_exp = 1.0 / order

        while (t - tf) * 1.0 < 0.0:
            if h > max_step:
                h = max_step
            if t + h > tf:
                h = abs(tf - t)

            y_high, y_low, err_vec, k = rk45_step_jit_kernel(f, t, y, h, A, B_HIGH, C, E)
            scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_high))
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

                beta = 1.0 / (order + 1)
                alpha = 0.4 * beta
                if err_prev < 0:
                    factor = SAFETY * (err_norm ** (-beta))
                else:
                    factor = SAFETY * (err_norm ** (-beta)) * (err_prev ** alpha)
                if factor < MIN_FACTOR:
                    factor = MIN_FACTOR
                if factor > MAX_FACTOR:
                    factor = MAX_FACTOR
                h = h * factor
                err_prev = err_norm
            else:
                factor = SAFETY * (err_norm ** (-err_exp))
                if factor < MIN_FACTOR:
                    factor = MIN_FACTOR
                h = h * factor
                if h < min_step:
                    h = min_step

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
            # Compute Q for new segment if needed: Q = K^T @ P
            if j != last_j:
                Kseg = Ks[j]
                # Q_cache[:, c] = sum_{r} Kseg[r, :] * P[r, c]
                for c in range(P.shape[1]):
                    # initialize column
                    for d in range(dim):
                        Q_cache[d, c] = 0.0
                    for r in range(Kseg.shape[0]):
                        coeff = P[r, c]
                        if coeff != 0.0:
                            for d in range(dim):
                                Q_cache[d, c] += coeff * Kseg[r, d]
                last_j = j
            # build p vector as cumulative powers of x
            p_len = P.shape[1]
            p = np.empty(p_len, dtype=np.float64)
            val = x
            for c in range(p_len):
                p[c] = val
                val *= x
            # y = y_old + h * Q @ p
            y_old = ys_arr[j]
            for d in range(dim):
                acc = 0.0
                for c in range(p_len):
                    acc += Q_cache[d, c] * p[c]
                y_out[idx, d] = y_old[d] + hseg * acc

        # Derivatives at t_eval (reuse f)
        derivs_out = np.empty_like(y_out)
        for idx in range(m):
            derivs_out[idx, :] = f(t_eval[idx], y_out[idx, :])

        return y_out, derivs_out


@numba.njit(cache=False, fastmath=FASTMATH)
def dop853_step_jit_kernel(f, t, y, h, A, B_HIGH, C, E5, E3):
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
        y_high, y_low, err_vec, _, _ = dop853_step_jit_kernel(
            f, t, y, h, self._A, self._B_HIGH, self._C, self._E5, self._E3
        )
        return y_high, y_low, err_vec

    def integrate(self, system: _DynamicalSystem, y0: np.ndarray, t_vals: np.ndarray, **kwargs) -> _Solution:
        """Adaptive integration fully in Numba for DOP853 with Hermite interpolation."""
        self.validate_inputs(system, y0, t_vals)
        f = _build_rhs_wrapper(system)
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
        h = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
        if h > max_step:
            h = max_step
        if h < min_step:
            h = min_step

        err_prev = -1.0
        SAFETY = 0.9
        MIN_FACTOR = 0.2
        MAX_FACTOR = 10.0
        err_exp = 1.0 / order

        while (t - tf) * 1.0 < 0.0:
            if h > max_step:
                h = max_step
            if t + h > tf:
                h = abs(tf - t)

            y_high, y_low, err_vec, err5, err3, k = dop853_step_jit_kernel(f, t, y, h, A, B_HIGH, C, E5, E3)
            scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_high))
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

                beta = 1.0 / (order + 1)
                alpha = 0.4 * beta
                if err_prev < 0:
                    factor = SAFETY * (err_norm ** (-beta))
                else:
                    factor = SAFETY * (err_norm ** (-beta)) * (err_prev ** alpha)
                if factor < MIN_FACTOR:
                    factor = MIN_FACTOR
                if factor > MAX_FACTOR:
                    factor = MAX_FACTOR
                h = h * factor
                err_prev = err_norm
            else:
                factor = SAFETY * (err_norm ** (-err_exp))
                if factor < MIN_FACTOR:
                    factor = MIN_FACTOR
                h = h * factor
                if h < min_step:
                    h = min_step

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
            x = (t_q - t0s) / hseg
            if j != last_j:
                # Build extended K for dense output
                Kext = np.empty((n_stages_extended, dim), dtype=np.float64)
                Kseg = Ks[j]
                s_used = Kseg.shape[0]
                for r in range(s_used):
                    for d in range(dim):
                        Kext[r, d] = Kseg[r, d]
                # compute additional stages using A_EXTRA rows with hseg and stored y_old
                y_old = ys_arr[j]
                t_old = t0s
                start_row = s_used
                for srow in range(start_row, n_stages_extended):
                    # dy = (Kext[:srow].T @ A_full[srow, :srow]) * hseg
                    # Build y_stage vector
                    y_stage = np.empty(dim, dtype=np.float64)
                    for d in range(dim):
                        acc = 0.0
                        for r in range(srow):
                            a = A_full[srow, r]
                            if a != 0.0:
                                acc += a * Kext[r, d]
                        y_stage[d] = y_old[d] + hseg * acc
                    # Evaluate derivative at stage
                    t_stage = t_old + C_full[srow] * hseg
                    k_vec = f(t_stage, y_stage)
                    for d in range(dim):
                        Kext[srow, d] = k_vec[d]
                # Build F
                # F[0] = delta_y, F[1] = h f_old - delta_y, F[2] = 2*delta_y - h(f_new+f_old)
                f_old = dys_arr[j]
                f_new = dys_arr[j + 1]
                delta_y = ys_arr[j + 1] - ys_arr[j]
                for d in range(dim):
                    F_cache[0, d] = delta_y[d]
                    F_cache[1, d] = hseg * f_old[d] - delta_y[d]
                    F_cache[2, d] = 2.0 * delta_y[d] - hseg * (f_new[d] + f_old[d])
                # Remaining rows: h * (D @ Kext)
                rows_remaining = interpolator_power - 3
                for irem in range(rows_remaining):
                    for d in range(dim):
                        acc = 0.0
                        for r in range(n_stages_extended):
                            coeff = D[irem, r]
                            if coeff != 0.0:
                                acc += coeff * Kext[r, d]
                        F_cache[3 + irem, d] = hseg * acc
                last_j = j
            # Evaluate Dop853DenseOutput polynomial by Horner-like alternating x and (1-x)
            y_val = np.zeros(dim, dtype=np.float64)
            for i in range(interpolator_power - 1, -1, -1):
                for d in range(dim):
                    y_val[d] += F_cache[i, d]
                if (interpolator_power - 1 - i) % 2 == 0:
                    # even offset -> multiply by x
                    for d in range(dim):
                        y_val[d] *= x
                else:
                    # odd offset -> multiply by (1 - x)
                    one_minus_x = 1.0 - x
                    for d in range(dim):
                        y_val[d] *= one_minus_x
            # add y_old
            y_old = ys_arr[j]
            for d in range(dim):
                y_out[idx, d] = y_val[d] + y_old[d]

        derivs_out = np.empty_like(y_out)
        for idx in range(m):
            derivs_out[idx, :] = f(t_eval[idx], y_out[idx, :])

        return y_out, derivs_out


class RungeKutta:
    """Implement a factory class for creating fixed-step Runge-Kutta integrators.
    
    This factory provides convenient access to fixed-step Runge-Kutta methods
    of different orders. The available orders are 4, 6, and 8.
    
    Examples
    --------
    >>> rk4 = RungeKutta(order=4)
    >>> rk6 = RungeKutta(order=6)
    >>> rk8 = RungeKutta(order=8)
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


def _build_rhs_wrapper(system: _DynamicalSystem) -> Callable[[float, np.ndarray], np.ndarray]:
    """Return the compiled (t, y) RHS from the system.

    Ensures `system.rhs` has the expected two-argument signature and returns it.
    All systems now expose a compiled dispatcher.

    Parameters
    ----------
    system : :class:`~hiten.algorithms.dynamics.base._DynamicalSystem`
        The dynamical system to wrap.

    Returns
    -------
    Callable[[float, np.ndarray], np.ndarray]
        The compiled `(t, y)` RHS callable.
    
    See Also
    --------
    :class:`~hiten.algorithms.dynamics.base._DynamicalSystem` : Base class
    :func:`~hiten.algorithms.dynamics.base._DynamicalSystem._compile_rhs_function` : JIT compilation method

    Raises
    ------
    ValueError
        If `system.rhs` does not have the `(t, y)` signature.
    """

    rhs_func = system.rhs
    # Sanity check signature
    sig = inspect.signature(rhs_func)
    if len(sig.parameters) < 2:
        raise ValueError("System.rhs must have signature (t, y)")
    return rhs_func
