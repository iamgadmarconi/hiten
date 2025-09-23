"""Provide a Circular Restricted Three-Body Problem (CR3BP) dynamics implementation.

This module provides JIT-compiled equations of motion, Jacobians, and variational
systems for the Circular Restricted Three-Body Problem in the synodic (rotating)
frame. All implementations are optimized with Numba for high-performance numerical
integration and stability analysis.

The implementation supports both forward and backward time integration with
appropriate handling of momentum variables for time-reversible systems.

References
----------
Szebehely, V. (1967). *Theory of Orbits: The Restricted Problem of Three Bodies*.
Academic Press.

Koon, W. S.; Lo, M. W.; Marsden, J. E.; Ross, S. D. (2011).
*Dynamical Systems, the Three-Body Problem and Space Mission Design*.
Caltech.
"""

from typing import Callable, Literal

import numba
import numpy as np

from hiten.algorithms.dynamics.base import _DynamicalSystem, _propagate_dynsys
from hiten.algorithms.utils.config import FASTMATH


@numba.njit(fastmath=FASTMATH, cache=True)
def _crtbp_accel(state, mu):
    r"""Compute CR3BP equations of motion in rotating synodic frame.
    
    JIT-compiled function that evaluates the full 6-dimensional equations of motion
    for the Circular Restricted Three-Body Problem, including gravitational forces
    from both primaries and Coriolis/centrifugal effects from the rotating frame.
    
    Parameters
    ----------
    state : ndarray, shape (6,)
        State vector [x, y, z, vx, vy, vz] in nondimensional rotating frame.
    mu : float
        Mass parameter mu = m2/(m1+m2) of the CR3BP system.
        
    Returns
    -------
    ndarray, shape (6,)
        Time derivative [vx, vy, vz, ax, ay, az] where accelerations include
        gravitational, Coriolis, and centrifugal terms.
        
    Notes
    -----
    - Primary m1 located at (-mu, 0, 0)
    - Secondary m2 located at (1-mu, 0, 0)
    - Includes 2*Omega x v Coriolis terms and Omega x (Omega x r) centrifugal terms
    - Uses nondimensional units with primary-secondary separation = 1
    
    See Also
    --------
    :class:`~hiten.algorithms.dynamics.rtbp._RTBPRHS` : Dynamical system
        wrapper for these equations
    :func:`~hiten.algorithms.dynamics.rtbp._jacobian_crtbp` : Analytical
        Jacobian of these equations
    """
    x, y, z, vx, vy, vz = state

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)

    ax = 2*vy + x - (1 - mu)*(x + mu) / r1**3 - mu*(x - 1 + mu) / r2**3
    ay = -2*vx + y - (1 - mu)*y / r1**3          - mu*y / r2**3
    az = -(1 - mu)*z / r1**3 - mu*z / r2**3

    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)

@numba.njit(fastmath=FASTMATH, cache=True)
def _jacobian_crtbp(x, y, z, mu):
    r"""Compute analytical Jacobian matrix of CR3BP equations of motion.
    
    JIT-compiled function that evaluates the 6x6 Jacobian matrix of the CR3BP
    vector field with respect to the state variables. Used for linearization,
    variational equations, and stability analysis.
    
    Parameters
    ----------
    x, y, z : float
        Position coordinates in nondimensional rotating frame.
    mu : float
        Mass parameter mu = m2/(m1+m2) of the CR3BP system.
        
    Returns
    -------
    ndarray, shape (6, 6)
        Jacobian matrix F = df/dy where f is the CR3BP vector field.
        Upper-right 3x3 block is identity (velocity terms).
        Lower-left 3x3 block contains gravitational and centrifugal derivatives.
        Off-diagonal terms (3,4) and (4,3) contain Coriolis coefficients.
        
    Notes
    -----
    - Analytical derivatives of gravitational potential and frame effects
    - Includes second derivatives of effective potential for acceleration terms
    - Coriolis terms appear as F[3,4] = 2, F[4,3] = -2 from rotating frame
    - Used internally by variational equations and stability analysis
    
    See Also
    --------
    :func:`~hiten.algorithms.dynamics.rtbp._var_equations` : Uses this
        Jacobian for STM propagation
    :class:`~hiten.algorithms.dynamics.rtbp._JacobianRHS` : Dynamical system
        wrapper for Jacobian evaluation
    :func:`~hiten.algorithms.dynamics.rtbp._crtbp_accel` : Vector field that
        this function differentiates
    """
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
    r"""Compute CR3BP variational equations for state transition matrix propagation.
    
    JIT-compiled function that evaluates the 42-dimensional variational system
    combining the 6x6 state transition matrix (STM) evolution with the base
    CR3BP dynamics. Used for computing fundamental matrix solutions and
    monodromy matrices of periodic orbits.
    
    Parameters
    ----------
    t : float
        Time variable (unused in autonomous system, required for ODE interface).
    PHI_vec : ndarray, shape (42,)
        Combined state vector: first 36 components are flattened 6x6 STM,
        last 6 components are physical state [x, y, z, vx, vy, vz].
    mu : float
        Mass parameter mu = m2/(m1+m2) of the CR3BP system.
        
    Returns
    -------
    ndarray, shape (42,)
        Time derivative of combined state: [d(STM)/dt flattened, dx/dt].
        STM evolution follows d(Phi)/dt = F(x) * Phi where F is the Jacobian.
        
    Notes
    -----
    - STM initialized as 6x6 identity matrix at t=0
    - Matrix multiplication F * Phi implemented with explicit loops for Numba compatibility
    - Physical state evolution uses same equations as _crtbp_accel
    - Combined system enables simultaneous propagation of trajectory and linearization
    
    See Also
    --------
    :func:`~hiten.algorithms.dynamics.rtbp._jacobian_crtbp` : Provides
        Jacobian matrix F for STM evolution
    :func:`~hiten.algorithms.dynamics.rtbp._crtbp_accel` : Base dynamics for
        physical state evolution
    :class:`~hiten.algorithms.dynamics.rtbp._VarEqRHS` : Dynamical system
        wrapper for these equations
    :func:`~hiten.algorithms.dynamics.rtbp._compute_stm` : Uses this
        function for STM propagation
    """
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


def _compute_stm(dynsys, x0, tf, steps=2000, forward=1, method: Literal["fixed", "adaptive", "symplectic"] = "adaptive", order=8, **kwargs):
    r"""Propagate state transition matrix (STM) along CR3BP trajectory.

    Integrates the 42-dimensional variational system to compute the fundamental
    matrix solution (state transition matrix) from initial time to final time.
    The STM describes how small perturbations to initial conditions evolve
    along the reference trajectory.

    Parameters
    ----------
    dynsys : :class:`~hiten.algorithms.dynamics.base._DynamicalSystem`
        Variational system implementing 42-dimensional CR3BP variational equations.
        Typically an instance of _VarEqRHS.
    x0 : array_like, shape (6,)
        Initial state vector [x, y, z, vx, vy, vz] in nondimensional rotating frame.
    tf : float
        Final integration time in nondimensional units.
    steps : int, optional
        Number of equally-spaced output time points. Default is 2000.
    forward : int, optional
        Integration direction: +1 for forward, -1 for backward time.
        Default is 1.
    method : {'fixed', 'adaptive', 'symplectic'}, optional
        Numerical integration method. Default is 'adaptive'.
    order : int, optional
        Integration order. Default is 8.
    **kwargs
        Additional keyword arguments passed to the integrator, including:
        - rtol: Relative tolerance for integration. If None, uses default from 
        :func:`~hiten.algorithms.dynamics.base._propagate_dynsys`.
        - atol: Absolute tolerance for integration. If None, uses default from 
        :func:`~hiten.algorithms.dynamics.base._propagate_dynsys`.


    Returns
    -------
    x : ndarray, shape (steps, 6)
        Reference trajectory in phase space.
    times : ndarray, shape (steps,)
        Time points corresponding to trajectory.
    phi_T : ndarray, shape (6, 6)
        State transition matrix Phi(tf) at final time.
    PHI : ndarray, shape (steps, 42)
        Complete evolution: flattened STM (36 components) + state (6 components).

    Notes
    -----
    - STM initialized as 6x6 identity matrix at t=0
    - Backward integration uses DirectedSystem with momentum sign flipping
    - Combined 42D system enables simultaneous trajectory and linearization
    - STM satisfies d(Phi)/dt = F(x(t)) * Phi(t) where F is the Jacobian

    See Also
    --------
    :func:`~hiten.algorithms.dynamics.rtbp._var_equations` : Variational equations used for integration
    :class:`~hiten.algorithms.dynamics.rtbp._VarEqRHS` : Dynamical system for variational equations
    :func:`~hiten.algorithms.dynamics.rtbp._compute_monodromy` : Specialized version for periodic orbits
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
        **kwargs
    )

    PHI = sol_obj.states

    x = PHI[:, 36:42]

    phi_tf_flat = PHI[-1, :36]
    phi_T = phi_tf_flat.reshape((6, 6))

    return x, sol_obj.times, phi_T, PHI


def _compute_monodromy(dynsys, x0, period):
    r"""Compute monodromy matrix for periodic CR3BP orbit.

    Calculates the monodromy matrix M = Phi(T) by integrating the variational
    equations over one complete orbital period. The monodromy matrix describes
    how small perturbations evolve after one complete orbit.

    Parameters
    ----------
    dynsys : _DynamicalSystem
        Variational system implementing CR3BP variational equations.
    x0 : array_like, shape (6,)
        Initial state on the periodic orbit.
    period : float
        Orbital period T in nondimensional time units.

    Returns
    -------
    ndarray, shape (6, 6)
        Monodromy matrix M = Phi(T) describing linearized return map.
        
    Notes
    -----
    - One eigenvalue is always 1 (tangent to periodic orbit)
    - For Hamiltonian systems, eigenvalues occur in reciprocal pairs
    - Stability determined by eigenvalue magnitudes relative to unit circle
    
    See Also
    --------
    :func:`~hiten.algorithms.dynamics.rtbp._compute_stm` : General STM
        computation used internally
    :func:`~hiten.algorithms.dynamics.rtbp._stability_indices` : Compute
        stability indices from monodromy matrix
    """
    _, _, M, _ = _compute_stm(dynsys, x0, period)
    return M


def _stability_indices(monodromy):
    r"""Compute Floquet stability indices for periodic orbit analysis.

    Calculates the classical stability indices nu_i = (lambda_i + 1/lambda_i)/2
    from the monodromy matrix eigenvalues. These indices characterize the
    linear stability of periodic orbits in Hamiltonian systems.

    Parameters
    ----------
    monodromy : ndarray, shape (6, 6)
        Monodromy matrix from periodic orbit integration.

    Returns
    -------
    tuple of float
        Stability indices (nu_1, nu_2) corresponding to the two non-trivial
        eigenvalue pairs. |nu_i| <= 1 indicates stability.
    ndarray, shape (6,)
        All eigenvalues sorted by absolute value (descending order).
        
    Notes
    -----
    - Uses eigenvalues at indices 2 and 4 (assumes sorted by magnitude)
    - For Hamiltonian systems, eigenvalues occur in reciprocal pairs
    - One eigenvalue is always 1 (tangent to periodic orbit)
    - Stable orbits have |nu_i| <= 1 for all i
    
    See Also
    --------
    :func:`~hiten.algorithms.dynamics.rtbp._compute_monodromy` : Provides
        monodromy matrix input
    :func:`~hiten.algorithms.common.linalg._stability_indices` :
        More robust version
    """
    eigs = np.linalg.eigvals(monodromy)
    
    eigs = sorted(eigs, key=abs, reverse=True)

    nu1 = 0.5 * (eigs[2] + 1/eigs[2])
    nu2 = 0.5 * (eigs[4] + 1/eigs[4])
    
    return (nu1, nu2), eigs


class _JacobianRHS(_DynamicalSystem):
    r"""Provide a dynamical system for CR3BP Jacobian matrix evaluation.

    Provides a dynamical system interface for evaluating the Jacobian matrix
    of the CR3BP vector field at specified positions. Used for linearization
    and sensitivity analysis applications.

    Parameters
    ----------
    mu : float
        Mass parameter mu = m2/(m1+m2) where 0 < mu < 1.
    name : str, optional
        Human-readable system identifier. Default is 'CR3BP Jacobian'.

    Attributes
    ----------
    mu : float
        Mass parameter of the CR3BP system.
    name : str
        System identifier string.
    rhs : Callable[[float, ndarray], ndarray]
        JIT-compiled function returning 6x6 Jacobian matrix.
        
    Notes
    -----
    - State dimension is 3 (only position coordinates x, y, z needed)
    - Returns full 6x6 Jacobian matrix flattened or as 2D array
    - Uses analytical derivatives for high accuracy
    - JIT-compiled for efficient repeated evaluation
    
    See Also
    --------
    :func:`~hiten.algorithms.dynamics.rtbp._jacobian_crtbp` : Core Jacobian
        computation function
    :class:`~hiten.algorithms.dynamics.rtbp._RTBPRHS` : Main CR3BP equations
        of motion
    :class:`~hiten.algorithms.dynamics.rtbp._VarEqRHS` : Variational equations
        using this Jacobian
    """
    def __init__(self, mu: float, name: str = "CR3BP Jacobian"):
        super().__init__(3)
        self.name = name
        self.mu = float(mu)
        
        self._mu_val = self.mu

    def _build_rhs_impl(self) -> Callable[[float, np.ndarray], np.ndarray]:
        mu_val = self._mu_val

        def _jacobian_rhs(t: float, state: np.ndarray, _mu=mu_val) -> np.ndarray:
            return _jacobian_crtbp(state[0], state[1], state[2], _mu)

        return _jacobian_rhs

    def __repr__(self) -> str:
        return f"_JacobianRHS(name='{self.name}', mu={self.mu})"


class _VarEqRHS(_DynamicalSystem):
    r"""Provide the CR3BP variational equations for state transition matrix propagation.

    Implements the 42-dimensional variational system that simultaneously
    evolves the 6x6 state transition matrix and the 6-dimensional phase
    space trajectory. Used for computing fundamental matrix solutions,
    monodromy matrices, and sensitivity analysis.

    Parameters
    ----------
    mu : float
        Mass parameter mu = m2/(m1+m2) of the CR3BP system.
    name : str, optional
        System identifier. Default is 'CR3BP Variational Equations'.

    Attributes
    ----------
    mu : float
        Mass parameter of the CR3BP system.
    name : str
        System identifier string.
    rhs : Callable[[float, ndarray], ndarray]
        JIT-compiled variational equations function.
        
    Notes
    -----
    - State dimension is 42: flattened 6x6 STM (36) + phase space state (6)
    - STM evolution follows d(Phi)/dt = F(x) * Phi where F is Jacobian
    - Physical state follows standard CR3BP equations of motion
    - Initialize STM as 6x6 identity matrix for fundamental solution
    
    Examples
    --------
    >>> # Create variational system
    >>> var_sys = _VarEqRHS(mu=0.01215)
    >>> # Initialize with identity STM and initial state
    >>> PHI0 = np.zeros(42)
    >>> PHI0[:36] = np.eye(6).flatten()  # Identity STM
    >>> PHI0[36:] = initial_state        # Initial 6D state
    
    See Also
    --------
    :func:`~hiten.algorithms.dynamics.rtbp._var_equations` : Core variational
        equations implementation
    :func:`~hiten.algorithms.dynamics.rtbp._compute_stm` : Uses this system
        for STM computation
    :class:`~hiten.algorithms.dynamics.rtbp._RTBPRHS` : Base CR3BP dynamics
    """
    def __init__(self, mu: float, name: str = "CR3BP Variational Equations"):
        super().__init__(42)
        self.name = name
        self.mu = float(mu)

        self._mu_val = self.mu

    def _build_rhs_impl(self) -> Callable[[float, np.ndarray], np.ndarray]:
        mu_val = self._mu_val

        def _var_eq_rhs(t: float, y: np.ndarray, _mu=mu_val) -> np.ndarray:
            return _var_equations(t, y, _mu)

        return _var_eq_rhs

    def __repr__(self) -> str:
        return f"_VarEqRHS(name='{self.name}', mu={self.mu})"


class _RTBPRHS(_DynamicalSystem):
    r"""Define the Circular Restricted Three-Body Problem equations of motion.

    Implements the full 6-dimensional CR3BP dynamics in the rotating synodic
    frame, including gravitational forces from both primaries and all
    rotating frame effects (Coriolis and centrifugal forces).

    Parameters
    ----------
    mu : float
        Mass parameter mu = m2/(m1+m2) where m1, m2 are primary/secondary masses.
    name : str, optional
        System identifier. Default is 'RTBP'.

    Attributes
    ----------
    dim : int
        State space dimension (always 6 for CR3BP).
    mu : float
        Mass parameter of the system.
    name : str
        System identifier string.
    rhs : Callable[[float, ndarray], ndarray]
        JIT-compiled CR3BP vector field function.
        
    Notes
    -----
    - Uses nondimensional units: distance = primary separation, time = orbital period/(2*pi)
    - Primary m1 located at (-mu, 0, 0), secondary m2 at (1-mu, 0, 0)
    - Includes all rotating frame effects: Coriolis, centrifugal, gravitational
    - Suitable for both planar and spatial (3D) motion
    - Autonomous system (no explicit time dependence)
    
    Examples
    --------
    >>> # Earth-Moon system (approximate)
    >>> em_system = _RTBPRHS(mu=0.01215, name="Earth-Moon CR3BP")
    >>> # Compute derivative at L1 point vicinity
    >>> state = np.array([0.8, 0, 0, 0, 0.1, 0])
    >>> derivative = em_system.rhs(0.0, state)
    
    See Also
    --------
    :func:`~hiten.algorithms.dynamics.rtbp._crtbp_accel` : Core equations of
        motion implementation
    :func:`~hiten.algorithms.dynamics.rtbp.rtbp_dynsys` : Factory function for
        creating instances
    :class:`~hiten.algorithms.dynamics.rtbp._VarEqRHS` : Variational equations
        based on this system
    """
    def __init__(self, mu: float, name: str = "RTBP"):
        super().__init__(dim=6)
        self.name = name
        self.mu = float(mu)

        self._mu_val = self.mu

    def _build_rhs_impl(self) -> Callable[[float, np.ndarray], np.ndarray]:
        mu_val = self._mu_val

        def _crtbp_rhs(t: float, state: np.ndarray, _mu=mu_val) -> np.ndarray:
            return _crtbp_accel(state, _mu)

        return _crtbp_rhs

    def __repr__(self) -> str:
        return f"_RTBPRHS(name='{self.name}', mu={self.mu})"


def rtbp_dynsys(mu: float, name: str = "RTBP") -> _RTBPRHS:
    """Create CR3BP dynamical system.
    
    Factory function for creating CR3BP equations of motion with specified
    mass parameter. Provides functional interface alternative to direct
    constructor usage.
    
    Parameters
    ----------
    mu : float
        Mass parameter mu = m2/(m1+m2) of the CR3BP system.
    name : str, optional
        System identifier. Default is "RTBP".
        
    Returns
    -------
    :class:`~hiten.algorithms.dynamics.rtbp._RTBPRHS`
        Configured CR3BP dynamical system.
        
    See Also
    --------
    :class:`~hiten.algorithms.dynamics.rtbp._RTBPRHS` : Direct constructor interface
    """
    return _RTBPRHS(mu=mu, name=name)

def jacobian_dynsys(mu: float, name: str="Jacobian") -> _JacobianRHS:
    """Create CR3BP Jacobian evaluation system.
    
    Factory function for creating Jacobian matrix evaluation system
    with specified mass parameter.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system.
    name : str, optional
        System identifier. Default is "Jacobian".
        
    Returns
    -------
    :class:`~hiten.algorithms.dynamics.rtbp._JacobianRHS`
        Configured Jacobian evaluation system.
        
    See Also
    --------
    :class:`~hiten.algorithms.dynamics.rtbp._JacobianRHS` : Direct constructor interface
    """
    return _JacobianRHS(mu=mu, name=name)

def variational_dynsys(mu: float, name: str = "VarEq") -> _VarEqRHS:
    """Create CR3BP variational equations system.
    
    Factory function for creating variational equations system for
    state transition matrix propagation.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system.
    name : str, optional
        System identifier. Default is "VarEq".
        
    Returns
    -------
    :class:`~hiten.algorithms.dynamics.rtbp._VarEqRHS`
        Configured variational equations system.
        
    See Also
    --------
    :class:`~hiten.algorithms.dynamics.rtbp._VarEqRHS` : Direct constructor interface
    :func:`~hiten.algorithms.dynamics.rtbp._compute_stm` : Uses this system for STM computation
    """
    return _VarEqRHS(mu=mu, name=name)
