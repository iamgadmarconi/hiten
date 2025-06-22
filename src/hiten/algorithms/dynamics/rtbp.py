r"""
hiten.algorithms.dynamics.rtbp
========================

Numba-accelerated equations of motion, Jacobians and variational
systems for the circular restricted three-body problem (CR3BP) written
in the synodic (rotating) frame.  The module also provides lightweight
:pyclass:`_DynamicalSystem` wrappers and utility helpers for propagating
the state-transition matrix (STM), the monodromy matrix and linear
stability indices of periodic orbits.

References
----------
Szebehely, V. (1967). "Theory of Orbits".

Koon, W. S.; Lo, M. W.; Marsden, J. E.; Ross, S. D. (2011). "Dynamical
Systems, the Three-Body Problem and Space Mission Design".
"""

from typing import Callable, Literal, Sequence

import numba
import numpy as np
from scipy.integrate import solve_ivp

from hiten.algorithms.dynamics.base import _DirectedSystem, _DynamicalSystem
from hiten.algorithms.integrators.base import Solution
from hiten.algorithms.integrators.rk import AdaptiveRK, RungeKutta
from hiten.algorithms.integrators.symplectic import TaoSymplectic
from hiten.utils.config import FASTMATH, TOL
from hiten.utils.log_config import logger


@numba.njit(fastmath=FASTMATH, cache=False)
def _crtbp_accel(state, mu):
    x, y, z, vx, vy, vz = state

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)

    ax = 2*vy + x - (1 - mu)*(x + mu) / r1**3 - mu*(x - 1 + mu) / r2**3
    ay = -2*vx + y - (1 - mu)*y / r1**3          - mu*y / r2**3
    az = -(1 - mu)*z / r1**3 - mu*z / r2**3

    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)

@numba.njit(fastmath=FASTMATH, cache=False)
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

@numba.njit(fastmath=FASTMATH, cache=False)
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
    r"""
    Propagate the state-transition matrix (STM).

    Parameters
    ----------
    dynsys : _DynamicalSystem
        Dynamical system exposing the 42-dimensional variational
        equations of the CR3BP.
    x0 : array_like, shape (6,)
        Initial phase-space state :math:`(x, y, z, \dot x, \dot y, \dot z)`.
    tf : float
        Final integration time :math:`t_{\mathrm f}`.
    steps : int, default 2000
        Number of output points equally spaced in time.
    forward : int, {1, -1}, default 1
        Sign of time flow.  If negative the system is integrated
        backward in time while the momentum-like variables are
        sign-flipped via :pyclass:`_DirectedSystem`.
    method : {'scipy', 'rk', 'symplectic', 'adaptive'}, default 'scipy'
        Integration backend to be used.
    order : int, default 8
        Order of the Runge-Kutta or symplectic scheme.

    Returns
    -------
    x : numpy.ndarray
        Trajectory of the physical state, shape *(steps, 6)*.
    times : numpy.ndarray
        Time stamps corresponding to *x*.
    phi_T : numpy.ndarray
        State-transition matrix at *tf*, shape *(6, 6)*.
    PHI : numpy.ndarray
        Flattened STM and state history, shape *(steps, 42)*.

    Notes
    -----
    The STM is embedded into a 42-dimensional vector composed of the 36
    independent entries of :math:`\Phi(t)` followed by the phase-space
    variables.  The combined system is then advanced with the selected
    integrator.
    """
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
    r"""
    Return the monodromy matrix of a periodic CR3BP orbit.

    Parameters
    ----------
    dynsys : _DynamicalSystem
        Variational system to propagate.
    x0 : array_like, shape (6,)
        Initial state on the periodic orbit.
    period : float
        Orbital period :math:`T`.

    Returns
    -------
    numpy.ndarray
        Monodromy matrix :math:`\Phi(T)` of shape *(6, 6)*.
    """
    _, _, M, _ = compute_stm(dynsys, x0, period)
    return M


def stability_indices(monodromy):
    r"""
    Compute the classical linear stability indices.

    Parameters
    ----------
    monodromy : numpy.ndarray
        Monodromy matrix obtained from :pyfunc:`monodromy_matrix`.

    Returns
    -------
    tuple
        Pair :math:`(\nu_1, \nu_2)` where each index is defined as
        :math:`\nu_i = \tfrac{1}{2}(\lambda_i + 1/\lambda_i)` with
        :math:`\lambda_i` the corresponding eigenvalue.
    numpy.ndarray
        Eigenvalues of *monodromy* sorted by absolute value (descending).
    """
    eigs = np.linalg.eigvals(monodromy)
    
    eigs = sorted(eigs, key=abs, reverse=True)

    nu1 = 0.5 * (eigs[2] + 1/eigs[2])
    nu2 = 0.5 * (eigs[4] + 1/eigs[4])
    
    return (nu1, nu2), eigs


class JacobianRHS(_DynamicalSystem):
    r"""
    Right-hand side returning the Jacobian matrix of the CR3BP.

    Parameters
    ----------
    mu : float
        Mass parameter :math:`\mu \in (0, 1)`.
    name : str, default 'CR3BP Jacobian'
        Human-readable identifier.

    Attributes
    ----------
    mu : float
        Normalised mass parameter.
    name : str
        Display name used in :pyfunc:`repr`.
    rhs : Callable[[float, numpy.ndarray], numpy.ndarray]
        Vector field returning the Jacobian evaluated at the current
        position.
    """
    def __init__(self, mu: float, name: str = "CR3BP Jacobian"):
        super().__init__(3)
        self.name = name
        self.mu = float(mu)
        
        mu_val = self.mu

        @numba.njit(fastmath=FASTMATH, cache=False)
        def _jacobian_rhs(t: float, state, _mu=mu_val) -> np.ndarray:
            return _jacobian_crtbp(state[0], state[1], state[2], _mu)
        
        self._rhs = _jacobian_rhs

    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        return self._rhs

    def __repr__(self) -> str:
        return f"JacobianRHS(name='{self.name}', mu={self.mu})"


class VariationalEquationsRHS(_DynamicalSystem):
    r"""
    Variational equations of the CR3BP (42-dimensional system).

    Parameters
    ----------
    mu : float
        Mass parameter.
    name : str, default 'CR3BP Variational Equations'
        Display name.

    Notes
    -----
    The state vector contains the flattened STM (first 36 components)
    followed by the usual 6-component phase-space state.
    """
    def __init__(self, mu: float, name: str = "CR3BP Variational Equations"):
        super().__init__(42)
        self.name = name
        self.mu = float(mu)

        mu_val = self.mu

        @numba.njit(fastmath=FASTMATH, cache=False)
        def _var_eq_rhs(t: float, y: np.ndarray, _mu=mu_val) -> np.ndarray:
            return _var_equations(t, y, _mu)
        
        self._rhs = _var_eq_rhs

    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        return self._rhs

    def __repr__(self) -> str:
        return f"VariationalEquationsRHS(name='{self.name}', mu={self.mu})"


class RTBPRHS(_DynamicalSystem):
    r"""
    Equations of motion of the planar/3-D circular restricted three-body problem.

    Parameters
    ----------
    mu : float
        Mass parameter of the secondary body.
    name : str, default 'RTBP'
        Display name.

    Attributes
    ----------
    dim : int
        Always 6.
    rhs : Callable[[float, numpy.ndarray], numpy.ndarray]
        Vector field :math:`f(t,\mathbf x)`.
    """
    def __init__(self, mu: float, name: str = "RTBP"):
        super().__init__(dim=6)
        self.name = name
        self.mu = float(mu)

        mu_val = self.mu

        @numba.njit(fastmath=FASTMATH, cache=False)
        def _crtbp_rhs(t: float, state: np.ndarray, _mu=mu_val) -> np.ndarray:
            return _crtbp_accel(state, _mu)

        self._rhs = _crtbp_rhs

    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        return self._rhs

    def __repr__(self) -> str:
        return f"RTBPRHS(name='{self.name}', mu={self.mu})"


def rtbp_dynsys(mu: float, name: str = "RTBP") -> RTBPRHS:
    """Factory helper returning :pyclass:`RTBPRHS` with the given *mu*."""
    return RTBPRHS(mu=mu, name=name)

def jacobian_dynsys(mu: float, name: str="Jacobian") -> JacobianRHS:
    """Factory helper returning :pyclass:`JacobianRHS`."""
    return JacobianRHS(mu=mu, name=name)

def variational_dynsys(mu: float, name: str = "VarEq") -> VariationalEquationsRHS:
    """Factory helper returning :pyclass:`VariationalEquationsRHS`."""
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
    r"""
    Generic propagation routine shared by public helpers.

    This is an internal utility.  It normalises *state0*, applies the
    :pyclass:`_DirectedSystem` wrapper and delegates the actual
    integration to the requested backend.
    """
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
    r"""
    Validate shape of *state*.

    Raises
    ------
    ValueError
        If length of *state* does not equal *expected_dim*.
    """
    state_np = np.asarray(state, dtype=np.float64)
    if state_np.shape != (expected_dim,):
        msg = f"Initial state vector must have {expected_dim} elements, but got shape {state_np.shape}"
        logger.error(msg)
        raise ValueError(msg)
    return state_np
