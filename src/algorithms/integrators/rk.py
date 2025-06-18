import inspect
from typing import Callable, Optional

import numpy as np
from numba import njit

from algorithms.dynamics.base import DynamicalSystem
from algorithms.dynamics.hamiltonian import HamiltonianSystem
from algorithms.integrators.base import Integrator, Solution
from algorithms.integrators.coefficients.dop853 import E3 as DOP853_E3
from algorithms.integrators.coefficients.dop853 import E5 as DOP853_E5
from algorithms.integrators.coefficients.dop853 import \
    N_STAGES as DOP853_N_STAGES
from algorithms.integrators.coefficients.dop853 import A as DOP853_A
from algorithms.integrators.coefficients.dop853 import B as DOP853_B
from algorithms.integrators.coefficients.dop853 import C as DOP853_C
from algorithms.integrators.coefficients.rk4 import A as RK4_A
from algorithms.integrators.coefficients.rk4 import B as RK4_B
from algorithms.integrators.coefficients.rk4 import C as RK4_C
from algorithms.integrators.coefficients.rk6 import A as RK6_A
from algorithms.integrators.coefficients.rk6 import B as RK6_B
from algorithms.integrators.coefficients.rk6 import C as RK6_C
from algorithms.integrators.coefficients.rk8 import A as RK8_A
from algorithms.integrators.coefficients.rk8 import B as RK8_B
from algorithms.integrators.coefficients.rk8 import C as RK8_C
from algorithms.integrators.coefficients.rk45 import B_HIGH as RK45_B_HIGH
from algorithms.integrators.coefficients.rk45 import B_LOW as RK45_B_LOW
from algorithms.integrators.coefficients.rk45 import A as RK45_A
from algorithms.integrators.coefficients.rk45 import C as RK45_C
from algorithms.integrators.symplectic import _eval_dH_dP, _eval_dH_dQ
from config import FASTMATH


class _RungeKuttaBase(Integrator):
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
    """Explicit fixed-step Runge-Kutta scheme (RK4/RK6/RK8)."""

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
        system: DynamicalSystem,
        y0: np.ndarray,
        t_vals: np.ndarray,
        **kwargs,
    ) -> Solution:
        self.validate_inputs(system, y0, t_vals)

        rhs_wrapped = _build_rhs_wrapper(system)

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

class _AdaptiveStepRK(_RungeKuttaBase):
    """Embedded adaptive Runge-Kutta using PI step-size control."""

    def __init__(self, name: str = "AdaptiveRK", rtol: float = 1e-10, atol: float = 1e-12,
                 max_step: float = np.inf, min_step: float = 0.0, **options):
        super().__init__(name, **options)
        self._rtol = rtol
        self._atol = atol
        self._max_step = max_step
        self._min_step = min_step
        if not hasattr(self, "_err_exp") or self._err_exp == 0:
            self._err_exp = 1.0 / (self._p)

    @property
    def order(self) -> int:
        return self._p

    def integrate(
        self,
        system: DynamicalSystem,
        y0: np.ndarray,
        t_vals: np.ndarray,
        **kwargs,
    ) -> Solution:
        self.validate_inputs(system, y0, t_vals)

        rhs_wrapped = _build_rhs_wrapper(system)

        forward = np.sign(t_vals[-1] - t_vals[0])

        def f(t, y):
            return forward * rhs_wrapped(t, y)

        t_span = (t_vals[0], t_vals[-1])
        t_eval = t_vals[1:-1] if t_vals.size > 2 else np.empty(0, dtype=np.float64)

        t = t_span[0]
        y = np.ascontiguousarray(y0, dtype=np.float64)
        ts, ys, dys = [t], [y.copy()], [f(t, y)]

        h = self._select_initial_step(f, t, y, t_span[1])
        idx_eval = 0

        while (t - t_span[1]) * forward < 0:
            if h > self._max_step:
                h = self._max_step
            if t + forward * h > t_span[1]:
                h = abs(t_span[1] - t)

            y_high, y_low, err_vec = self._rk_embedded_step(f, t, y, h)
            scale = self._atol + self._rtol * np.maximum(np.abs(y), np.abs(y_high))
            err_norm = np.sqrt(np.mean((err_vec / scale) ** 2))

            if err_norm <= 1.0:
                t_new = t + h
                y_new = y_high

                while idx_eval < t_eval.size and (t_eval[idx_eval] - t_new) * forward <= 0:
                    tau = (t_eval[idx_eval] - t) / (t_new - t)
                    y_eval = y + tau * (y_new - y)
                    ys.append(y_eval)
                    ts.append(t_eval[idx_eval])
                    dys.append(f(t_eval[idx_eval], y_eval))
                    idx_eval += 1

                ts.append(t_new)
                ys.append(y_new.copy())
                dys.append(f(t_new, y_new))
                t, y = t_new, y_new
                h *= self._update_factor(err_norm)
            else:
                h *= max(0.2, 0.9 * err_norm ** (-self._err_exp))
                if h < self._min_step:
                    raise RuntimeError("Step size underflow in adaptive RK integrator.")

        return Solution(times=np.asarray(ts), states=np.asarray(ys), derivatives=np.asarray(dys))

    def _select_initial_step(self, f, t0, y0, tf):
        dy0 = f(t0, y0)
        scale = self._atol + self._rtol * np.abs(y0)
        d0 = np.sqrt(np.mean((y0 / scale) ** 2))
        d1 = np.sqrt(np.mean((dy0 / scale) ** 2))
        h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
        return min(max(self._min_step, h0), abs(tf - t0))

    def _update_factor(self, err_norm):
        return np.clip(0.9 * err_norm ** (-self._err_exp), 0.2, 5.0)

class RK4(_FixedStepRK):
    def __init__(self, **opts):
        super().__init__("RK4", RK4_A, RK4_B, RK4_C, 4, **opts)

class RK6(_FixedStepRK):
    def __init__(self, **opts):
        super().__init__("RK6", RK6_A, RK6_B, RK6_C, 6, **opts)

class RK8(_FixedStepRK):
    def __init__(self, **opts):
        super().__init__("RK8", RK8_A, RK8_B, RK8_C, 8, **opts)

class RK45(_AdaptiveStepRK):
    _A = RK45_A
    _B_HIGH = RK45_B_HIGH
    _B_LOW = RK45_B_LOW
    _C = RK45_C
    _p = 5
    _err_exp = 1.0 / 5.0

    def __init__(self, **opts):
        super().__init__("RK45", **opts)

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
        """Perform one adaptive DOP853 step.

        Parameters
        ----------
        f : callable
            RHS function of the ODE system (already wrapped for sign and
            numba-compiled by the driver).
        t : float
            Current time.
        y : ndarray
            Current solution vector.
        h : float
            Proposed step size (may be adjusted by the caller).

        Returns
        -------
        y_high : ndarray
            8th-order accepted solution at *t + h*.
        y_low : ndarray
            A pseudo low-order solution constructed so that the difference
            ``y_high - y_low`` equals the local error estimate.  This enables
            reuse of the generic adaptive RK controller implemented in
            :class:`_AdaptiveStepRK`.
        err_vec : ndarray
            Local truncation error estimate.
        """
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

        err5 = np.dot(k.T, self._E5)  # 5th-order error component
        err3 = np.dot(k.T, self._E3)  # 3rd-order error component
        denom = np.hypot(np.abs(err5), 0.1 * np.abs(err3))
        correction_factor = np.ones_like(err5)
        mask = denom > 0.0
        correction_factor[mask] = np.abs(err5[mask]) / denom[mask]
        err_vec = h * err5 * correction_factor

        y_low = y_high - err_vec

        return y_high, y_low, err_vec

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


@njit(cache=True, fastmath=FASTMATH)
def _hamiltonian_rhs(y: np.ndarray, jac_H, clmo_H, n_dof: int) -> np.ndarray:  # type: ignore[valid-type]
    """Numba-compiled core that evaluates (dQ, dP) for a Hamiltonian system.

    Parameters
    ----------
    y : ndarray
        2*n_dof phase-space vector [Q, P].
    jac_H, clmo_H : numba.typed.List
        Polynomial Jacobian and coefficient-layout objects coming from the
        center-manifold build.  They are passed as *arguments* so that Numba
        does not need to embed them as compile-time constants.
    n_dof : int
        Number of degrees of freedom.
    """
    Q = y[:n_dof]
    P = y[n_dof : 2 * n_dof]

    dQ = _eval_dH_dP(Q, P, jac_H, clmo_H)
    dP = -_eval_dH_dQ(Q, P, jac_H, clmo_H)

    out = np.empty_like(y)
    out[:n_dof] = dQ
    out[n_dof : 2 * n_dof] = dP
    return out

def _build_rhs_wrapper(system: DynamicalSystem) -> Callable[[float, np.ndarray], np.ndarray]:

    if isinstance(system, HamiltonianSystem):
        n_dof = system.n_dof
        jac_H = system.jac_H
        clmo_H = system.clmo_H

        def _ham_rhs(t, y):
            return _hamiltonian_rhs(y, jac_H, clmo_H, n_dof)

        return _ham_rhs

    rhs_func = system.rhs

    try:
        sig = inspect.signature(rhs_func.py_func)
    except AttributeError:
        sig = inspect.signature(rhs_func)

    n_params = len(sig.parameters)

    if n_params >= 2:

        @njit(cache=False, fastmath=FASTMATH)
        def _rhs_two(t, y):
            return rhs_func(t, y)

        return _rhs_two

    elif n_params == 1:

        @njit(cache=False, fastmath=FASTMATH)
        def _rhs_one(t, y):
            return rhs_func(y)

        return _rhs_one

    else:
        raise ValueError(
            f"Unsupported rhs signature with {n_params} parameters. "
            "Only (t, y) or (y,) are currently supported."
        )
