import numpy as np
from numba import njit
from numba.typed import List

from algorithms.center.polynomial.base import CLMO_GLOBAL
from algorithms.center.polynomial.operations import (polynomial_evaluate)

# Yoshida 6th-order symmetric composition weights (Solution A)
# Order: w3, w2, w1, w0, w1, w2, w3
YOSHIDA6_WEIGHTS_A = np.array([
    0.784513610477560,      # w3_A
    0.235573213359357,      # w2_A
   -1.17767998417887,       # w1_A
    1.315186320683906,      # w0_A
   -1.17767998417887,       # w1_A
    0.235573213359357,      # w2_A
    0.784513610477560       # w3_A
], dtype=np.float64)
# Yoshida 6th-order symmetric composition weights (Solution B)
# Order: w3, w2, w1, w0, w1, w2, w3
YOSHIDA6_WEIGHTS_B = np.array([
    1.43984816797678,       # w3_B
    0.00426068187079180,    # w2_B
   -2.13228522200144,       # w1_B
    2.3763527443077364,     # w0_B
   -2.13228522200144,       # w1_B
    0.00426068187079180,    # w2_B
    1.43984816797678        # w3_B
], dtype=np.float64)
# Yoshida 6th-order symmetric composition weights (Solution C)
# Order: w3, w2, w1, w0, w1, w2, w3
YOSHIDA6_WEIGHTS_C = np.array([
    1.4477825623992997,     # w3_c
   -2.1440353163053898,     # w2_C
    0.0015288622842492212,  # w1_C
    2.3894477832436816,     # w0_C
    0.0015288622842492212,  # w1_C
   -2.1440353163053898,     # w2_C
    1.4477825623992997      # w3_C
], dtype=np.float64)

YOSHIDA6_WEIGHTS = YOSHIDA6_WEIGHTS_C


@njit(fastmath=True, cache=True)
def _flow_rhs_6d(state6: np.ndarray, dH_blocks: List[List[np.ndarray]], clmo: List[np.ndarray]) -> np.ndarray:
    """Return (dq₁/dt, dq₂/dt, dq₃/dt, dp₁/dt, dp₂/dt, dp₃/dt) at *state6*.
    The state vector *state6* is (q₁, q₂, q₃, p₁, p₂, p₃).
    """
    # Gradients w.r.t q₁, q₂, q₃, p₁, p₂, p₃
    # dH_blocks[0] = dH/dq₁
    # dH_blocks[1] = dH/dq₂
    # dH_blocks[2] = dH/dq₃
    # dH_blocks[3] = dH/dp₁
    # dH_blocks[4] = dH/dp₂
    # dH_blocks[5] = dH/dp₃
    dH_dq1 = polynomial_evaluate(dH_blocks[0], state6, clmo).real
    dH_dq2 = polynomial_evaluate(dH_blocks[1], state6, clmo).real
    dH_dq3 = polynomial_evaluate(dH_blocks[2], state6, clmo).real
    dH_dp1 = polynomial_evaluate(dH_blocks[3], state6, clmo).real
    dH_dp2 = polynomial_evaluate(dH_blocks[4], state6, clmo).real
    dH_dp3 = polynomial_evaluate(dH_blocks[5], state6, clmo).real

    return np.array(( dH_dp1,          # dq₁/dt
                      dH_dp2,          # dq₂/dt
                      dH_dp3,          # dq₃/dt
                     -dH_dq1,          # dp₁/dt
                     -dH_dq2,          # dp₂/dt
                     -dH_dq3),         # dp₃/dt
                    dtype=np.float64)


@njit(fastmath=True, cache=True)
def _implicit_midpoint_step(state: np.ndarray, dt: float, dH_blocks: List[List[np.ndarray]], clmo: List[np.ndarray], tol: float = 1e-12) -> np.ndarray:
    """Return state (6D) at t+dt using the implicit midpoint map.
    Uses a simple fixed point iteration which converges quadratically for
    sufficiently small dt (poly RHS).
    Input state is (q₁, q₂, q₃, p₁, p₂, p₃).
    """
    max_iter = 50
    s_old = state
    s_new = s_old.copy()

    for _ in range(max_iter):
        mid = 0.5 * (s_old + s_new)
        rhs = _flow_rhs_6d(mid, dH_blocks, clmo)
        s_next = s_old + dt * rhs
        if np.linalg.norm(s_next - s_new) < tol:
            return s_next
        s_new = s_next
    # Fallback or warning can be added here if max_iter is reached without convergence
    return s_new

@njit(fastmath=True, cache=True)
def yoshida6_step(state: np.ndarray, dt: float, dH_blocks: List[List[np.ndarray]], clmo: List[np.ndarray]) -> np.ndarray:
    """One global 6th-order symplectic step via Yoshida composition of the
    implicit midpoint kernel. Works for *any* Hamiltonian.
    Input state is (q₁, q₂, q₃, p₁, p₂, p₃).
    """
    s = state
    for w in YOSHIDA6_WEIGHTS:
        s = _implicit_midpoint_step(s, w * dt, dH_blocks, clmo)
    return s

@njit(fastmath=True, cache=True)
def yoshida6_integrate(state0: np.ndarray, t_end: float, dt: float, H_blocks: List[np.ndarray], dH_blocks: List[List[np.ndarray]], clmo=CLMO_GLOBAL) -> List[np.ndarray, np.ndarray]:
    """Integrate from *state0* (q₁,q₂,q₃,p₁,p₂,p₃) up to t_end with constant step *dt*.
    Returns trajectory array (Nx6) and the energy at each step.
    """
    n_steps = int(np.ceil(t_end / dt))
    traj = np.empty((n_steps + 1, 6), dtype=np.float64) # Changed from 4 to 6
    energy = np.empty(n_steps + 1, dtype=np.float64)

    # Initial state & energy
    traj[0] = state0.copy()
    energy[0] = polynomial_evaluate(H_blocks, state0, clmo).real # Use state0 directly

    s = state0.copy() # Ensure s is a copy
    for k in range(1, n_steps + 1):
        s = yoshida6_step(s, dt, dH_blocks, clmo)
        traj[k] = s
        energy[k] = polynomial_evaluate(H_blocks, s, clmo).real # Use current state s

    return traj, energy
