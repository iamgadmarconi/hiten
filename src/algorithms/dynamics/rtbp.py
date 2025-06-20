from typing import Callable

import numba
import numpy as np
from algorithms.dynamics.rhs import create_rhs_system
from algorithms.integrators.rk import RungeKutta

from algorithms.dynamics.base import DynamicalSystem
from config import FASTMATH


@numba.njit(fastmath=FASTMATH, cache=True)
def crtbp_accel(state, mu):
    """
    Calculate the state derivative (acceleration) for the CR3BP.
    
    This function computes the time derivative of the state vector in the 
    Circular Restricted Three-Body Problem. It returns the velocity and 
    acceleration components that define the equations of motion in the 
    rotating reference frame.
    
    Parameters
    ----------
    state : array_like
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    ndarray
        Time derivative of the state vector [vx, vy, vz, ax, ay, az]
    
    Notes
    -----
    The equations of motion include gravitational forces from both primaries
    and the Coriolis and centrifugal forces from the rotating reference frame.
    This function is optimized using Numba for high-performance computations.
    """
    x, y, z, vx, vy, vz = state

    # Distances to each primary
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)      # from m1 at (-mu, 0, 0)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2) # from m2 at (1-mu, 0, 0)

    # Accelerations
    ax = 2*vy + x - (1 - mu)*(x + mu) / r1**3 - mu*(x - 1 + mu) / r2**3
    ay = -2*vx + y - (1 - mu)*y / r1**3          - mu*y / r2**3
    az = -(1 - mu)*z / r1**3 - mu*z / r2**3

    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)

@numba.njit(fastmath=FASTMATH, cache=True)
def jacobian_crtbp(x, y, z, mu):
    """
    Compute the Jacobian matrix for the CR3BP equations of motion.
    
    This function calculates the 6x6 Jacobian matrix (state transition matrix 
    derivative) for the 3D Circular Restricted Three-Body Problem in the 
    rotating reference frame. It's used in stability analysis and for computing
    the variational equations.
    
    Parameters
    ----------
    x : float
        x-coordinate in the rotating frame
    y : float
        y-coordinate in the rotating frame
    z : float
        z-coordinate in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    ndarray
        6x6 Jacobian matrix structured as:
        [ 0    0    0    1     0    0 ]
        [ 0    0    0    0     1    0 ]
        [ 0    0    0    0     0    1 ]
        [ omgxx omgxy omgxz  0     2    0 ]
        [ omgxy omgyy omgyz -2     0    0 ]
        [ omgxz omgyz omgzz  0     0    0 ]
    
    Notes
    -----
    The indices of the matrix correspond to: x=0, y=1, z=2, vx=3, vy=4, vz=5.
    The implementation matches the partial derivatives formulation common in
    astrodynamics literature.
    """

    # As in var3D.m:
    #   mu2 = 1 - mu (big mass fraction)
    mu2 = 1.0 - mu

    # Distances squared to the two primaries
    # r^2 = (x+mu)^2 + y^2 + z^2       (distance^2 to M1, which is at (-mu, 0, 0))
    # R^2 = (x - mu2)^2 + y^2 + z^2    (distance^2 to M2, which is at (1-mu, 0, 0))
    r2 = (x + mu)**2 + y**2 + z**2
    R2 = (x - mu2)**2 + y**2 + z**2
    r3 = r2**1.5
    r5 = r2**2.5
    R3 = R2**1.5
    R5 = R2**2.5

    # From var3D.m, the partial derivatives "omgxx," "omgyy," ...
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

    # Build the 6x6 matrix F
    F = np.zeros((6, 6), dtype=np.float64)

    # Identity block for velocity wrt position
    F[0, 3] = 1.0  # dx/dvx
    F[1, 4] = 1.0  # dy/dvy
    F[2, 5] = 1.0  # dz/dvz

    # The second derivatives block
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
def variational_equations(t, PHI_vec, mu, forward=1):
    """
    Compute the variational equations for the CR3BP.
    
    This function implements the 3D variational equations for the CR3BP,
    calculating the time derivatives of both the state transition matrix (STM)
    and the state vector simultaneously. It's used for sensitivity analysis,
    differential correction, and stability analysis.
    
    Parameters
    ----------
    t : float
        Current time (not used, but required for ODE integrators)
    PHI_vec : ndarray
        Combined 42-element vector containing:
        - First 36 elements: flattened 6x6 state transition matrix (STM)
        - Last 6 elements: state vector [x, y, z, vx, vy, vz]
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    forward : int, optional
        Direction of integration (1 for forward, -1 for backward). Default is 1.
    
    Returns
    -------
    ndarray
        42-element vector containing:
        - First 36 elements: time derivative of flattened STM (dΦ/dt)
        - Last 6 elements: state derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    
    Notes
    -----
    The variational equations integrate the STM Φ according to dΦ/dt = F·Φ,
    where F is the Jacobian of the system. This allows tracking how small 
    perturbations in initial conditions propagate through the system, which is
    essential for targeting and differential correction algorithms.
    """
    # 1) Unpack the STM (first 36) and the state (last 6)
    phi_flat = PHI_vec[:36]
    x_vec    = PHI_vec[36:]  # [x, y, z, vx, vy, vz]

    # Reshape the STM to 6x6
    Phi = phi_flat.reshape((6, 6))

    # Unpack the state
    x, y, z, vx, vy, vz = x_vec

    # 2) Build the 6x6 matrix F from the partial derivatives
    F = jacobian_crtbp(x, y, z, mu)

    # 3) dPhi/dt = F * Phi  (manually done to keep numba happy)
    phidot = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            s = 0.0 
            for k in range(6):
                s += F[i, k] * Phi[k, j]
            phidot[i, j] = s

    # 4) State derivatives, same formula as var3D.m
    #    xdot(4) = x(1) - mu2*( (x+mu)/r3 ) - mu*( (x-mu2)/R3 ) + 2*vy, etc.
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

    # 5) Build derivative of the 42-vector
    dPHI_vec = np.zeros_like(PHI_vec)

    # First 36 = flattened phidot
    dPHI_vec[:36] = phidot.ravel()

    # Last 6 = [vx, vy, vz, ax, ay, az], each multiplied by 'forward'
    dPHI_vec[36] = forward * vx
    dPHI_vec[37] = forward * vy
    dPHI_vec[38] = forward * vz
    dPHI_vec[39] = forward * ax
    dPHI_vec[40] = forward * ay
    dPHI_vec[41] = forward * az

    return dPHI_vec


def compute_stm(x0, mu, tf, **solve_kwargs):
    """
    Compute the State Transition Matrix (STM) for the CR3BP.
    
    This function integrates the combined CR3BP equations of motion and 
    variational equations from t=0 to t=tf to obtain the state transition 
    matrix Φ(tf, 0). The implementation mirrors MATLAB's var3D approach
    for compatibility with existing methods.
    
    Parameters
    ----------
    x0 : array_like
        Initial state vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    tf : float
        Final integration time (must be positive)
    forward : int, optional
        Direction of integration (1 for forward, -1 for backward). Default is 1.
    **solve_kwargs
        Additional keyword arguments
    
    Returns
    -------
    x : ndarray
        Array of shape (n_times, 6) containing the state trajectory
    t : ndarray
        Array of shape (n_times,) containing the time points
        (If forward=-1, times are negated to reflect backward integration)
    phi_T : ndarray
        The 6x6 state transition matrix Φ(tf, 0) at the final time
    PHI : ndarray
        Array of shape (n_times, 42) containing the full integrated
        solution at each time point, where each row is [flattened STM(36), state(6)]
    
    Notes
    -----
    The state transition matrix Φ(t, t0) maps perturbations in the initial state
    to perturbations at time t according to: δx(t) = Φ(t, t0)·δx(t0).
    
    The STM is initialized as the 6x6 identity matrix at t=0, representing
    the fact that initially, a perturbation in any direction affects only that
    direction with unit magnitude.
    
    For numerical reasons, the function always integrates forward in time
    (from 0 to |tf|), but controls the direction of the dynamics through the
    'forward' parameter, which is multiplied by the derivatives.
    """

    # Build initial 42-vector in MATLAB ordering: [flattened STM, state]
    PHI0 = np.zeros(42, dtype=np.float64)
    PHI0[:36] = np.eye(6, dtype=np.float64).ravel()
    PHI0[36:] = x0

    # Determine fixed-step grid.  If caller supplies 'steps' use it; else dt.
    steps = solve_kwargs.pop('steps', 4000)
    times = np.linspace(0.0, tf, steps, dtype=np.float64)

    # Create DynamicalSystem wrapper for the 42-dim variational equations
    def _var_eq(t, y):  # noqa: D401 (simple func)
        return variational_equations(t, y, mu, 1)

    rhs_system = create_rhs_system(_var_eq, dim=42, name="CR3BP STM")

    # Use high-order RK8 for accuracy unless caller overrides
    order = solve_kwargs.pop('rk_order', 6)
    integrator = RungeKutta(order=order)

    sol_obj = integrator.integrate(rhs_system, PHI0, times)

    PHI = sol_obj.states  # shape (n_times, 42)
    
    # The state is in columns [36..41] of PHI
    x = PHI[:, 36:42]   # shape (n_times, 6)

    # The final row's first 36 columns = flattened STM at t=tf
    phi_tf_flat = PHI[-1, :36]
    phi_T = phi_tf_flat.reshape((6, 6))

    return x, sol_obj.times, phi_T, PHI


def monodromy_matrix(x0, mu, period, **solve_kwargs):
    """
    Compute the monodromy matrix for a periodic orbit.
    
    The monodromy matrix is the state transition matrix evaluated over one 
    orbital period of a periodic orbit. Its eigenvalues (the Floquet multipliers)
    determine the stability properties of the orbit.
    
    Parameters
    ----------
    x0 : array_like
        Initial state vector [x, y, z, vx, vy, vz] representing a point on the periodic orbit
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    period : float
        Period of the orbit
    **solve_kwargs
        Additional keyword arguments passed to compute_stm
    
    Returns
    -------
    M : ndarray
        6x6 monodromy matrix
    """
    _, _, M, _ = compute_stm(x0, mu, period, **solve_kwargs)
    return M


def stability_indices(monodromy):
    """
    Compute stability indices from the monodromy matrix eigenvalues.
    
    For a periodic orbit in the CR3BP, the stability indices are calculated from
    the non-trivial eigenvalues of the monodromy matrix. These indices help 
    characterize the stability properties of the orbit.
    
    Parameters
    ----------
    monodromy : ndarray
        6x6 monodromy matrix
    
    Returns
    -------
    nu : tuple of float
        Stability indices calculated from the eigenvalues
    eigenvalues : ndarray
        The eigenvalues of the monodromy matrix
    """
    # Calculate eigenvalues of the monodromy matrix
    eigs = np.linalg.eigvals(monodromy)
    
    # Sort eigenvalues by magnitude
    eigs = sorted(eigs, key=abs, reverse=True)

    nu1 = 0.5 * (eigs[2] + 1/eigs[2])
    nu2 = 0.5 * (eigs[4] + 1/eigs[4])
    
    return (nu1, nu2), eigs


class RTBPSystem(DynamicalSystem):
    """Dynamical system wrapper for the Circular Restricted Three-Body Problem."""

    def __init__(self, mu: float, name: str = "RTBP"):
        super().__init__(dim=6)
        self.name = name
        self.mu = float(mu)

        mu_val = self.mu

        @numba.njit(fastmath=FASTMATH, cache=True)
        def _crtbp_rhs(t: float, state: np.ndarray, _mu=mu_val) -> np.ndarray:  # type: ignore[missing-return-type]
            return crtbp_accel(state, _mu)

        self._rhs = _crtbp_rhs

    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        return self._rhs

    @property
    def jacobian(self) -> Callable:
        return jacobian_crtbp

    @property
    def stm(self) -> Callable[[np.ndarray, float, int, dict], tuple]:
        return compute_stm

    def __repr__(self) -> str:
        return f"RTBPSystem(name='{self.name}', mu={self.mu})"


def create_rtbp_system(mu: float, name: str = "RTBP") -> RTBPSystem:
    """Factory matching the signature used in the test-suite."""
    return RTBPSystem(mu=mu, name=name)