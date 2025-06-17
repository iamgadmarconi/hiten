import inspect
from typing import Callable

import numpy as np
from numba import njit

from algorithms.dynamics.base import DynamicalSystem
from algorithms.dynamics.hamiltonian import HamiltonianSystem
from algorithms.integrators.base import Integrator, Solution
from algorithms.integrators.symplectic import _eval_dH_dP, _eval_dH_dQ
from config import FASTMATH

# RK4 (Classic 4th order)
RK4_A = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
], dtype=np.float64)

RK4_B = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0], dtype=np.float64)
RK4_C = np.array([0.0, 0.5, 0.5, 1.0], dtype=np.float64)

# RK6 (Dormand-Prince 6th order method)
RK6_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
    [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
    [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
    [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]
], dtype=np.float64)

RK6_B = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0], dtype=np.float64)
RK6_C = np.array([0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0], dtype=np.float64)

# RK8 (Dormand-Prince 8th order method)
RK8_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0/18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0/48.0, 1.0/16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0/32.0, 0.0, 3.0/32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.0/16.0, 0.0, -75.0/64.0, 75.0/64.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0/80.0, 0.0, 0.0, 3.0/16.0, 3.0/20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [29443841.0/614563906.0, 0.0, 0.0, 77736538.0/692538347.0, -28693883.0/1125000000.0, 23124283.0/1800000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [16016141.0/946692911.0, 0.0, 0.0, 61564180.0/158732637.0, 22789713.0/633445777.0, 545815736.0/2771057229.0, -180193667.0/1043307555.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [39632708.0/573591083.0, 0.0, 0.0, -433636366.0/683701615.0, -421739975.0/2616292301.0, 100302831.0/723423059.0, 790204164.0/839813087.0, 800635310.0/3783071287.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [246121993.0/1340847787.0, 0.0, 0.0, -37695042795.0/15268766246.0, -309121744.0/1061227803.0, -12992083.0/490766935.0, 6005943493.0/2108947869.0, 393006217.0/1396673457.0, 123872331.0/1001029789.0, 0.0, 0.0, 0.0, 0.0],
    [-1028468189.0/846180014.0, 0.0, 0.0, 8478235783.0/508512852.0, 1311729495.0/1432422823.0, -10304129995.0/1701304382.0, -48777925059.0/3047939560.0, 15336726248.0/1032824649.0, -45442868181.0/3398467696.0, 3065993473.0/597172653.0, 0.0, 0.0, 0.0],
    [185892177.0/718116043.0, 0.0, 0.0, -3185094517.0/667107341.0, -477755414.0/1098053517.0, -703635378.0/230739211.0, 5731566787.0/1027545527.0, 5232866602.0/850066563.0, -4093664535.0/808688257.0, 3962137247.0/1805957418.0, 65686358.0/487910083.0, 0.0, 0.0],
    [403863854.0/491063109.0, 0.0, 0.0, -5068492393.0/434740067.0, -411421997.0/543043805.0, 652783627.0/914296604.0, 11173962825.0/925320556.0, -13158990841.0/6184727034.0, 3936647629.0/1978049680.0, -160528059.0/685178525.0, 248638103.0/1413531060.0, 0.0, 0.0]
], dtype=np.float64)

RK8_B = np.array([14005451.0/335480064.0, 0.0, 0.0, 0.0, 0.0, -59238493.0/1068277825.0, 181606767.0/758867731.0, 561292985.0/797845732.0, -1041891430.0/1371343529.0, 760417239.0/1151165299.0, 118820643.0/751138087.0, -528747749.0/2220607170.0, 1.0/4.0], dtype=np.float64)
RK8_C = np.array([0.0, 1.0/18.0, 1.0/12.0, 1.0/8.0, 5.0/16.0, 3.0/8.0, 59.0/400.0, 93.0/200.0, 5490023248.0/9719169821.0, 13.0/20.0, 1201146811.0/1299019798.0, 1.0, 1.0], dtype=np.float64)


@njit(cache=False, fastmath=FASTMATH)
def _integrate_rk_generic(
    rhs_func: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_vals: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:

    n_steps = t_vals.shape[0]
    dim = y0.shape[0]

    n_stages = B.shape[0]

    traj = np.empty((n_steps, dim), dtype=np.float64)
    traj[0, :] = y0.copy()

    k = np.empty((n_stages, dim), dtype=np.float64)

    for step in range(n_steps - 1):
        t_n = t_vals[step]
        h = t_vals[step + 1] - t_n

        y_n = traj[step].copy()

        for s in range(n_stages):
            y_stage = y_n.copy()
            for j in range(s):
                a_sj = A[s, j]
                if a_sj != 0.0:
                    y_stage += h * a_sj * k[j]

            k[s] = rhs_func(t_n + C[s] * h, y_stage)

        y_np1 = y_n.copy()
        for s in range(n_stages):
            b_s = B[s]
            if b_s != 0.0:
                y_np1 += h * b_s * k[s]

        traj[step + 1] = y_np1

    return traj


@njit(cache=False, fastmath=FASTMATH)
def _integrate_rk_ham(
    y0: np.ndarray,
    t_vals: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    jac_H,
    clmo_H,
) -> np.ndarray:
    """Explicit RK integrator specialised for `HamiltonianSystem`."""

    n_steps = t_vals.shape[0]
    dim = y0.shape[0]
    n_stages = B.shape[0]
    traj = np.empty((n_steps, dim), dtype=np.float64)
    traj[0, :] = y0.copy()

    k = np.empty((n_stages, dim), dtype=np.float64)

    n_dof = dim // 2

    for step in range(n_steps - 1):
        t_n = t_vals[step]
        h = t_vals[step + 1] - t_n

        y_n = traj[step].copy()

        for s in range(n_stages):
            y_stage = y_n.copy()
            for j in range(s):
                a_sj = A[s, j]
                if a_sj != 0.0:
                    y_stage += h * a_sj * k[j]

            Q = y_stage[0:n_dof]
            P = y_stage[n_dof: 2 * n_dof]

            dQ = _eval_dH_dP(Q, P, jac_H, clmo_H)
            dP = -_eval_dH_dQ(Q, P, jac_H, clmo_H)

            k[s, 0:n_dof] = dQ
            k[s, n_dof: 2 * n_dof] = dP

        y_np1 = y_n.copy()
        for s in range(n_stages):
            b_s = B[s]
            if b_s != 0.0:
                y_np1 += h * b_s * k[s]

        traj[step + 1] = y_np1

    return traj


def _build_rhs_wrapper(system: DynamicalSystem) -> Callable[[float, np.ndarray], np.ndarray]:
    """Return a jit-compiled wrapper with signature ``f(t, y)``.

    The function inspects *system* and creates a suitable wrapper so that the
    underlying RHS regardless of its original argument list can be called
    from the Numba kernel using the uniform 2-argument signature expected by
    _integrate_rk.
    """

    if isinstance(system, HamiltonianSystem):
        n_dof = system.n_dof
        jac_H = system.jac_H
        clmo_H = system.clmo_H

        @njit(cache=False, fastmath=FASTMATH)
        def _ham_rhs(t, y):
            Q = y[:n_dof]
            P = y[n_dof:2 * n_dof]
            dQ = _eval_dH_dP(Q, P, jac_H, clmo_H)
            dP = - _eval_dH_dQ(Q, P, jac_H, clmo_H)
            out = np.empty_like(y)
            out[:n_dof] = dQ
            out[n_dof:2 * n_dof] = dP
            return out

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


def integrate_rk(
    system: DynamicalSystem,
    initial_state: np.ndarray,
    t_values: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    y0 = np.ascontiguousarray(initial_state, dtype=np.float64)
    t_vals = np.ascontiguousarray(t_values, dtype=np.float64)

    if isinstance(system, HamiltonianSystem):
        jac_H = system.jac_H
        clmo_H = system.clmo_H
        return _integrate_rk_ham(y0, t_vals, A, B, C, jac_H, clmo_H)

    # Generic system path
    rhs_wrapped = _build_rhs_wrapper(system)

    return _integrate_rk_generic(rhs_wrapped, y0, t_vals, A, B, C)


class RungeKutta(Integrator):

    def __init__(self, order: int = 4, **options):
        if order not in [4, 6, 8]:
            raise ValueError(f"RK order must be 4, 6, or 8, got {order}")
        
        super().__init__(f"RK{order}", **options)
        self._order = order
        
        # Select coefficients based on order
        if order == 4:
            self._A, self._B, self._C = RK4_A, RK4_B, RK4_C
        elif order == 6:
            self._A, self._B, self._C = RK6_A, RK6_B, RK6_C
        elif order == 8:
            self._A, self._B, self._C = RK8_A, RK8_B, RK8_C
    
    @property
    def order(self) -> int:
        """Order of accuracy of the RK method."""
        return self._order
    
    def integrate(
        self,
        system: DynamicalSystem,
        y0: np.ndarray,
        t_vals: np.ndarray,
        **kwargs
) -> Solution:

        self.validate_inputs(system, y0, t_vals)

        trajectory_array = integrate_rk(
            system=system,
            initial_state=y0,
            t_values=t_vals,
            A=self._A,
            B=self._B,
            C=self._C,
        )

        return Solution(times=t_vals.copy(), states=trajectory_array)