from typing import Callable, Literal, Sequence

import numba
import numpy as np
from scipy.integrate import solve_ivp

from algorithms.dynamics.base import _DirectedSystem, _DynamicalSystem
from algorithms.integrators.base import Solution
from algorithms.integrators.rk import AdaptiveRK, RungeKutta
from algorithms.integrators.symplectic import TaoSymplectic
from config import FASTMATH, TOL
from utils.log_config import logger


@numba.njit(fastmath=FASTMATH, cache=True)
def _crtbp_accel(state, mu):
    x, y, z, vx, vy, vz = state

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)

    ax = 2*vy + x - (1 - mu)*(x + mu) / r1**3 - mu*(x - 1 + mu) / r2**3
    ay = -2*vx + y - (1 - mu)*y / r1**3          - mu*y / r2**3
    az = -(1 - mu)*z / r1**3 - mu*z / r2**3

    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)

@numba.njit(fastmath=FASTMATH, cache=True)
def _jacobian_crtbp(x, y, z, mu):
    mu2 = 1.0 - mu

    r2 = (x + mu)**2 + y**2 + z**2
    R2 = (x - mu2)**2 + y**2 + z**2
    r3 = r2**1.5
    r5 = r2**2.5
    R3 = R2**1.5
    R5 = R2**2.5

    omgxx = 1.0 \
        + mu2/r5 * 3.0*(x + mu)**2 \
        + mu  /R5 * 3.0*(x - mu2)**2 \
        - (mu2/r3 + mu/R3)

    omgyy = 1.0 \
        + mu2/r5 * 3.0*(y**2) \
        + mu  /R5 * 3.0*(y**2) \
        - (mu2/r3 + mu/R3)

    omgzz = 0.0 \
        + mu2/r5 * 3.0*(z**2) \
        + mu  /R5 * 3.0*(z**2) \
        - (mu2/r3 + mu/R3)

    omgxy = 3.0*y * ( mu2*(x + mu)/r5 + mu*(x - mu2)/R5 )
    omgxz = 3.0*z * ( mu2*(x + mu)/r5 + mu*(x - mu2)/R5 )
    omgyz = 3.0*y*z*( mu2/r5 + mu/R5 )

    F = np.zeros((6, 6), dtype=np.float64)

    F[0, 3] = 1.0  # dx/dvx
    F[1, 4] = 1.0  # dy/dvy
    F[2, 5] = 1.0  # dz/dvz

    F[3, 0] = omgxx
    F[3, 1] = omgxy
    F[3, 2] = omgxz

    F[4, 0] = omgxy
    F[4, 1] = omgyy
    F[4, 2] = omgyz

    F[5, 0] = omgxz
    F[5, 1] = omgyz
    F[5, 2] = omgzz

    # Coriolis terms
    F[3, 4] = 2.0
    F[4, 3] = -2.0

    return F

@numba.njit(fastmath=FASTMATH, cache=True)
def _var_equations(t, PHI_vec, mu):
    phi_flat = PHI_vec[:36]
    x_vec    = PHI_vec[36:]  # [x, y, z, vx, vy, vz]

    Phi = phi_flat.reshape((6, 6))

    x, y, z, vx, vy, vz = x_vec

    F = _jacobian_crtbp(x, y, z, mu)

    phidot = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            s = 0.0 
            for k in range(6):
                s += F[i, k] * Phi[k, j]
            phidot[i, j] = s

    mu2 = 1.0 - mu
    r2 = (x + mu)**2 + y**2 + z**2
    R2 = (x - mu2)**2 + y**2 + z**2
    r3 = r2**1.5
    R3 = R2**1.5

    ax = ( x 
           - mu2*( (x+mu)/r3 ) 
           -  mu*( (x - mu2)/R3 ) 
           + 2.0*vy )
    ay = ( y
           - mu2*( y / r3 )
           -  mu*( y / R3 )
           - 2.0*vx )
    az = ( - mu2*( z / r3 ) 
           - mu  *( z / R3 ) )

    dPHI_vec = np.zeros_like(PHI_vec)

    dPHI_vec[:36] = phidot.ravel()

    dPHI_vec[36] = vx
    dPHI_vec[37] = vy
    dPHI_vec[38] = vz
    dPHI_vec[39] = ax
    dPHI_vec[40] = ay
    dPHI_vec[41] = az

    return dPHI_vec


def compute_stm(dynsys, x0, tf, steps=2000, forward=1, method: Literal["scipy", "rk", "symplectic", "adaptive"] = "scipy", order=8):
    PHI0 = np.zeros(42, dtype=np.float64)
    PHI0[:36] = np.eye(6, dtype=np.float64).ravel()
    PHI0[36:] = x0

    sol_obj = _propagate_dynsys(
        dynsys=dynsys,
        state0=PHI0,
        t0=0.0,
        tf=tf,
        forward=forward,
        steps=steps,
        method=method,
        order=order,
        flip_indices=slice(36, 42),
    )

    PHI = sol_obj.states

    x = PHI[:, 36:42]

    phi_tf_flat = PHI[-1, :36]
    phi_T = phi_tf_flat.reshape((6, 6))

    return x, sol_obj.times, phi_T, PHI


def monodromy_matrix(dynsys, x0, period):
    _, _, M, _ = compute_stm(dynsys, x0, period)
    return M


def stability_indices(monodromy):
    eigs = np.linalg.eigvals(monodromy)
    
    eigs = sorted(eigs, key=abs, reverse=True)

    nu1 = 0.5 * (eigs[2] + 1/eigs[2])
    nu2 = 0.5 * (eigs[4] + 1/eigs[4])
    
    return (nu1, nu2), eigs


class JacobianRHS(_DynamicalSystem):
    def __init__(self, mu: float, name: str = "CR3BP Jacobian"):
        super().__init__(3)
        self.name = name
        self.mu = float(mu)
        
        mu_val = self.mu

        @numba.njit(fastmath=FASTMATH, cache=True)
        def _jacobian_rhs(t: float, state, _mu=mu_val) -> np.ndarray:
            return _jacobian_crtbp(state[0], state[1], state[2], _mu)
        
        self._rhs = _jacobian_rhs

    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        return self._rhs

    def __repr__(self) -> str:
        return f"JacobianRHS(name='{self.name}', mu={self.mu})"


class VariationalEquationsRHS(_DynamicalSystem):
    def __init__(self, mu: float, name: str = "CR3BP Variational Equations"):
        super().__init__(42)
        self.name = name
        self.mu = float(mu)

        mu_val = self.mu

        @numba.njit(fastmath=FASTMATH, cache=True)
        def _var_eq_rhs(t: float, y: np.ndarray, _mu=mu_val) -> np.ndarray:
            return _var_equations(t, y, _mu)
        
        self._rhs = _var_eq_rhs

    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        return self._rhs

    def __repr__(self) -> str:
        return f"VariationalEquationsRHS(name='{self.name}', mu={self.mu})"


class RTBPRHS(_DynamicalSystem):
    def __init__(self, mu: float, name: str = "RTBP"):
        super().__init__(dim=6)
        self.name = name
        self.mu = float(mu)

        mu_val = self.mu

        @numba.njit(fastmath=FASTMATH, cache=True)
        def _crtbp_rhs(t: float, state: np.ndarray, _mu=mu_val) -> np.ndarray:
            return _crtbp_accel(state, _mu)

        self._rhs = _crtbp_rhs

    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        return self._rhs

    def __repr__(self) -> str:
        return f"RTBPRHS(name='{self.name}', mu={self.mu})"


def rtbp_dynsys(mu: float, name: str = "RTBP") -> RTBPRHS:
    return RTBPRHS(mu=mu, name=name)

def jacobian_dynsys(mu: float, name: str="Jacobian") -> JacobianRHS:
    return JacobianRHS(mu=mu, name=name)

def variational_dynsys(mu: float, name: str = "VarEq") -> VariationalEquationsRHS:
    return VariationalEquationsRHS(mu=mu, name=name)


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
) -> Solution:
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
        integrator = TaoSymplectic(order=order)
        sol = integrator.integrate(dynsys_dir, state0_np, t_eval)
        times = sol.times
        states = sol.states
        
    elif method == "adaptive":
        integrator = AdaptiveRK(order=order, max_step=1e4, rtol=1e-3, atol=1e-6)
        sol = integrator.integrate(dynsys_dir, state0_np, t_eval)
        times = sol.times
        states = sol.states

    times_signed = forward * times

    return Solution(times_signed, states)


def _validate_initial_state(state, expected_dim=6): 
    state_np = np.asarray(state, dtype=np.float64)
    if state_np.shape != (expected_dim,):
        msg = f"Initial state vector must have {expected_dim} elements, but got shape {state_np.shape}"
        logger.error(msg)
        raise ValueError(msg)
    return state_np
