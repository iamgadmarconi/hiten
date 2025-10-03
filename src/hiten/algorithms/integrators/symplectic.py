"""Provide high-order explicit symplectic integrators for polynomial Hamiltonian systems.
systems with n_dof=3.  The module provides two layers of
functionality:

The implementation follows the recursive operator-splitting strategy
originally proposed by Tao and yields a family of even-order schemes that
exactly preserve the symplectic structure and exhibit excellent long-term
energy behaviour.

References
----------
Tao, M. (2016). "Explicit symplectic approximation of non-separable
Hamiltonians: algorithm and long-time performance".
"""

import numpy as np
from numba import njit
from numba.typed import List

from hiten.algorithms.dynamics.protocols import _HamiltonianSystemProtocol
from hiten.algorithms.integrators.base import _Integrator, _Solution
from hiten.algorithms.integrators.utils import (_bisection_update,
                                                _bracket_converged,
                                                _crossed_direction,
                                                _event_crossed)
from hiten.algorithms.polynomial.operations import _polynomial_evaluate
from hiten.algorithms.types.configs import EventConfig
from hiten.algorithms.types.options import EventOptions
from hiten.algorithms.utils.config import FASTMATH

N_SYMPLECTIC_DOF = 3
N_VARS_POLY = 6
Q_POLY_INDICES = np.array([0, 1, 2], dtype=np.int64)
P_POLY_INDICES = np.array([3, 4, 5], dtype=np.int64)


@njit(fastmath=FASTMATH, cache=False)
def _get_tao_omega (delta: float, order: int, c: float = 10.0) -> float:
    """
    Calculate the frequency parameter for the symplectic integrator.
    
    Parameters
    ----------
    delta : float
        Time step size
    order : int
        Order of the symplectic integrator
    c : float, optional
        Scaling parameter, default is 10.0
        
    Returns
    -------
    float
        Frequency parameter tau*omega for the symplectic scheme
        
    Notes
    -----
    The calculated value scales with (c*delta)^(-order) to ensure 
    numerical stability for larger time steps.
    """
    return (c * delta)**(-float(order))


@njit(cache=False, fastmath=FASTMATH)
def _construct_6d_eval_point(Q_current_ndof: np.ndarray, P_current_ndof: np.ndarray) -> np.ndarray:
    """
    Construct a 6D evaluation point from N-DOF position and momentum vectors.
    Assumes N_SYMPLECTIC_DOF is 3 for this specific 6D polynomial evaluation context.
    
    Parameters
    ----------
    Q_current_ndof : numpy.ndarray
        Position vector (dimension N_SYMPLECTIC_DOF, e.g., [q1, q2, q3])
    P_current_ndof : numpy.ndarray
        Momentum vector (dimension N_SYMPLECTIC_DOF, e.g., [p1, p2, p3])
        
    Returns
    -------
    numpy.ndarray
        6D evaluation point for polynomial evaluation, ordered [q1,q2,q3,p1,p2,p3]
        
    Notes
    -----
    This function maps N-DOF coordinates to a 6D vector suitable for
    the polynomial evaluation, which expects variables in a specific order.
    """
    if Q_current_ndof.shape[0] != N_SYMPLECTIC_DOF or P_current_ndof.shape[0] != N_SYMPLECTIC_DOF:
        # This check is more for Numba's type inference and AOT compilation,
        # as it can't raise dynamic ValueErrors easily.
        # Consider how to handle errors if Numba context allows.
        pass

    point_6d = np.zeros(N_VARS_POLY, dtype=np.complex128) # Use complex for _polynomial_evaluate

    # Map Q and P variables to the 6D vector
    # Q_current_ndof = [q1, q2, q3] maps to point_6d[0], point_6d[1], point_6d[2]
    # P_current_ndof = [p1, p2, p3] maps to point_6d[3], point_6d[4], point_6d[5]
    for i in range(N_SYMPLECTIC_DOF):
        point_6d[Q_POLY_INDICES[i]] = Q_current_ndof[i]
        point_6d[P_POLY_INDICES[i]] = P_current_ndof[i]
        
    return point_6d

@njit(cache=False, fastmath=FASTMATH)
def _eval_dH_dQ(
    Q_eval_ndof: np.ndarray,
    P_eval_ndof: np.ndarray,
    jac_H: List[List[np.ndarray]],
    clmo_H: List[np.ndarray]
) -> np.ndarray:
    """
    Evaluate derivatives of Hamiltonian with respect to generalized position variables.
    
    Parameters
    ----------
    Q_eval_ndof : numpy.ndarray
        Position vector ([q1,q2,q3]) at which to evaluate derivatives
    P_eval_ndof : numpy.ndarray
        Momentum vector ([p1,p2,p3]) at which to evaluate derivatives
    jac_H : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as list of polynomial coefficients for 6 variables
    clmo_H : List[numpy.ndarray]
        List of coefficient layout mapping objects for the polynomials
        
    Returns
    -------
    numpy.ndarray
        Vector of partial derivatives dH/dQ (e.g., [dH/dq1, dH/dq2, dH/dq3])
    """
    eval_point_6d = _construct_6d_eval_point(Q_eval_ndof, P_eval_ndof)
    
    derivatives_Q = np.empty(N_SYMPLECTIC_DOF, dtype=np.float64)

    for i in range(N_SYMPLECTIC_DOF):
        poly_var_index = Q_POLY_INDICES[i]
        dH_dQi_poly = jac_H[poly_var_index]
        val_dH_dQi = _polynomial_evaluate(dH_dQi_poly, eval_point_6d, clmo_H)
        derivatives_Q[i] = val_dH_dQi.real
    
    return derivatives_Q

@njit(cache=False, fastmath=FASTMATH)
def _eval_dH_dP(
    Q_eval_ndof: np.ndarray,
    P_eval_ndof: np.ndarray,
    jac_H: List[List[np.ndarray]],
    clmo_H: List[np.ndarray]
) -> np.ndarray:
    """
    Evaluate derivatives of Hamiltonian with respect to generalized momentum variables.

    Parameters
    ----------
    Q_eval_ndof : numpy.ndarray
        Position vector ([q1,q2,q3]) at which to evaluate derivatives
    P_eval_ndof : numpy.ndarray
        Momentum vector ([p1,p2,p3]) at which to evaluate derivatives
    jac_H : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as list of polynomial coefficients for 6 variables
    clmo_H : List[numpy.ndarray]
        List of coefficient layout mapping objects for the polynomials
        
    Returns
    -------
    numpy.ndarray
        Vector of partial derivatives dH/dP (e.g., [dH/dp1, dH/dp2, dH/dp3])
    """
    eval_point_6d = _construct_6d_eval_point(Q_eval_ndof, P_eval_ndof)
    
    derivatives_P = np.empty(N_SYMPLECTIC_DOF, dtype=np.float64)

    for i in range(N_SYMPLECTIC_DOF):
        poly_var_index = P_POLY_INDICES[i]
        dH_dPi_poly = jac_H[poly_var_index]
        val_dH_dPi = _polynomial_evaluate(dH_dPi_poly, eval_point_6d, clmo_H)
        derivatives_P[i] = val_dH_dPi.real
        
    return derivatives_P

@njit(cache=False, fastmath=FASTMATH)
def _eval_hamiltonian_derivative(
    Q: np.ndarray,
    P: np.ndarray,
    jac_H: List[List[np.ndarray]],
    clmo_H: List[np.ndarray]
) -> np.ndarray:
    """Evaluate time derivative dy/dt = J * nabla_H for Hamiltonian system.
    
    For a Hamiltonian system with coordinates [Q, P]:
    - dQ/dt = dH/dP
    - dP/dt = -dH/dQ
    
    Parameters
    ----------
    Q : numpy.ndarray
        Position vector (dimension N_SYMPLECTIC_DOF)
    P : numpy.ndarray
        Momentum vector (dimension N_SYMPLECTIC_DOF)
    jac_H : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as list of polynomial coefficients
    clmo_H : List[numpy.ndarray]
        List of coefficient layout mapping objects
        
    Returns
    -------
    numpy.ndarray
        Time derivative vector [dQ/dt, dP/dt] of shape (2*N_SYMPLECTIC_DOF,)
        
    Notes
    -----
    This derivative is used for Hermite interpolation during event detection.
    The returned derivative represents the rate of change in the physical
    phase space [Q, P], not the extended phase space [Q, P, X, Y].
    """
    n_dof = Q.shape[0]
    derivative = np.empty(2 * n_dof, dtype=np.float64)
    
    # dQ/dt = ∂H/∂P (positive)
    dH_dP = _eval_dH_dP(Q, P, jac_H, clmo_H)
    derivative[0:n_dof] = dH_dP
    
    # dP/dt = -∂H/∂Q (negative)
    dH_dQ = _eval_dH_dQ(Q, P, jac_H, clmo_H)
    derivative[n_dof:2*n_dof] = -dH_dQ
    
    return derivative

@njit(cache=False, fastmath=FASTMATH)
def _hermite_eval_dense_symplectic(y0, f0, y1, f1, x, h):
    """Evaluate cubic Hermite interpolant for symplectic integrator events.
    
    Uses standard Hermite basis functions to interpolate between two states.
    This function is used during event refinement and is NOT symplectic,
    but provides accurate interpolation within a single timestep.
    
    Parameters
    ----------
    y0 : numpy.ndarray
        State at t0 (shape: 2*N_SYMPLECTIC_DOF)
    f0 : numpy.ndarray
        Derivative at t0 (shape: 2*N_SYMPLECTIC_DOF)
    y1 : numpy.ndarray
        State at t1 (shape: 2*N_SYMPLECTIC_DOF)
    f1 : numpy.ndarray
        Derivative at t1 (shape: 2*N_SYMPLECTIC_DOF)
    x : float
        Interpolation parameter in [0, 1]
    h : float
        Step size (t1 - t0)
        
    Returns
    -------
    numpy.ndarray
        Interpolated state at t0 + x*h
        
    Notes
    -----
    Uses cubic Hermite basis functions:
    - H00 = 2x^3 - 3x^2 + 1
    - H10 = x^3 - 2x^2 + x
    - H01 = -2x^3 + 3x^2
    - H11 = x^3 - x^2
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

@njit(cache=False, fastmath=FASTMATH)
def _hermite_refine_event_symplectic(
    event_fn,
    t0: float,
    y0: np.ndarray,
    f0: np.ndarray,
    t1: float,
    y1: np.ndarray,
    f1: np.ndarray,
    h: float,
    direction: int,
    xtol: float,
    gtol: float
):
    """Refine event location using bisection on Hermite interpolant.
    
    Performs bisection search on the interval [0, 1] to locate the
    precise time when the event function crosses zero.
    
    Parameters
    ----------
    event_fn : callable
        Compiled event function g(t, y) -> float
    t0 : float
        Time at start of step
    y0 : numpy.ndarray
        State at start of step
    f0 : numpy.ndarray
        Derivative at start of step
    t1 : float
        Time at end of step
    y1 : numpy.ndarray
        State at end of step
    f1 : numpy.ndarray
        Derivative at end of step
    h : float
        Step size (t1 - t0)
    direction : int
        Event direction (-1, 0, +1)
    xtol : float
        Tolerance on time location
    gtol : float
        Tolerance on event function value
        
    Returns
    -------
    t_hit : float
        Time of event
    y_hit : numpy.ndarray
        State at event
        
    Notes
    -----
    The bisection is performed on the normalized interval [0, 1],
    then mapped back to [t0, t1]. The Hermite interpolation provides
    smooth, differentiable approximation but is NOT symplectic.
    """
    a = 0.0
    b = 1.0
    g_left = event_fn(t0, y0)
    max_iter = 128
    
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        y_mid = _hermite_eval_dense_symplectic(y0, f0, y1, f1, mid, h)
        g_mid = event_fn(t0 + mid * h, y_mid)
        
        # Check if we've converged
        if abs(g_mid) <= gtol:
            x_hit = mid
            t_hit = t0 + x_hit * h
            y_hit = _hermite_eval_dense_symplectic(y0, f0, y1, f1, x_hit, h)
            return t_hit, y_hit
        
        # Update bracket
        crossed = _crossed_direction(g_left, g_mid, direction)
        a, b, g_left = _bisection_update(a, b, g_left, mid, g_mid, crossed)
        
        # Check bracket convergence
        if _bracket_converged(a, b, h, xtol):
            break
    
    # Return best estimate at end of bisection
    x_hit = b
    t_hit = t0 + x_hit * h
    y_hit = _hermite_eval_dense_symplectic(y0, f0, y1, f1, x_hit, h)
    return t_hit, y_hit

@njit(cache=False, fastmath=FASTMATH)
def _phi_H_a_update_poly(
    q_ext: np.ndarray, 
    delta: float, 
    jac_H: List[List[np.ndarray]], 
    clmo_H: List[np.ndarray]
    ):
    """
    Apply the first Hamiltonian splitting operator (phi_a) in the symplectic scheme.
    
    Parameters
    ----------
    q_ext : numpy.ndarray
        Extended state vector [Q, P, X, Y] to be updated in-place
    delta : float
        Time step size
    jac_H : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as list of polynomial coefficients
    clmo_H : List[numpy.ndarray]
        List of coefficient layout mapping objects for the polynomials
        
    Notes
    -----
    Implements the symplectic update step:
    - P <- P - delta * dH/dQ(Q,Y)
    - X <- X + delta * dH/dP(Q,Y)
    
    This modifies q_ext in-place through views/slices.
    Q, P, X, Y are now N_SYMPLECTIC_DOF dimensional.
    """
    Q_current = q_ext[0:N_SYMPLECTIC_DOF]
    P_current = q_ext[N_SYMPLECTIC_DOF : 2*N_SYMPLECTIC_DOF]
    X_current = q_ext[2*N_SYMPLECTIC_DOF : 3*N_SYMPLECTIC_DOF]
    Y_current = q_ext[3*N_SYMPLECTIC_DOF : 4*N_SYMPLECTIC_DOF]

    # dH/dq(q,y) means evaluate dH/dQ at (Q_current, Y_current)
    dH_dQ_at_QY = _eval_dH_dQ(Q_current, Y_current, jac_H, clmo_H)
    # dH/dp(q,y) means evaluate dH/dP at (Q_current, Y_current)
    dH_dP_at_QY = _eval_dH_dP(Q_current, Y_current, jac_H, clmo_H)

    # Update P and X (modifies q_ext in place via views)
    P_current -= delta * dH_dQ_at_QY
    X_current += delta * dH_dP_at_QY

@njit(cache=False, fastmath=FASTMATH)
def _phi_H_b_update_poly(
    q_ext: np.ndarray, 
    delta: float, 
    jac_H: List[List[np.ndarray]], 
    clmo_H: List[np.ndarray]
    ):
    """
    Apply the second Hamiltonian splitting operator (phi_b) in the symplectic scheme.
    
    Parameters
    ----------
    q_ext : numpy.ndarray
        Extended state vector [Q, P, X, Y] to be updated in-place
    delta : float
        Time step size
    jac_H : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as list of polynomial coefficients
    clmo_H : List[numpy.ndarray]
        List of coefficient layout mapping objects for the polynomials
        
    Notes
    -----
    Implements the symplectic update step:
    - Q <- Q + delta * dH/dP(X,P)
    - Y <- Y - delta * dH/dQ(X,P)
    
    This modifies q_ext in-place through views/slices.
    Q, P, X, Y are now N_SYMPLECTIC_DOF dimensional.
    """
    Q_current = q_ext[0:N_SYMPLECTIC_DOF]
    P_current = q_ext[N_SYMPLECTIC_DOF : 2*N_SYMPLECTIC_DOF]
    X_current = q_ext[2*N_SYMPLECTIC_DOF : 3*N_SYMPLECTIC_DOF]
    Y_current = q_ext[3*N_SYMPLECTIC_DOF : 4*N_SYMPLECTIC_DOF]

    # dH/dp(x,p) means evaluate dH/dP at (X_current, P_current)
    dH_dP_at_XP = _eval_dH_dP(X_current, P_current, jac_H, clmo_H)
    # dH/dq(x,p) means evaluate dH/dQ at (X_current, P_current)
    dH_dQ_at_XP = _eval_dH_dQ(X_current, P_current, jac_H, clmo_H)
    
    # Update Q and Y (modifies q_ext in place via views)
    Q_current += delta * dH_dP_at_XP
    Y_current -= delta * dH_dQ_at_XP

@njit(cache=False, fastmath=FASTMATH)
def _phi_omega_H_c_update_poly(q_ext: np.ndarray, delta: float, omega: float):
    """
    Apply the rotation operator (phi_c) in the symplectic scheme.
    
    Parameters
    ----------
    q_ext : numpy.ndarray
        Extended state vector [Q, P, X, Y] to be updated in-place
    delta : float
        Time step size
    omega : float
        Frequency parameter for the rotation
        
    Notes
    -----
    Implements a rotation in the extended phase space with mixing of coordinates.
    The transformation is implemented using trigonometric functions and temporary
    variables to ensure numerical stability.
    
    This step is crucial for high-order symplectic integration methods
    with the extended phase-space technique.
    Q, P, X, Y are now N_SYMPLECTIC_DOF dimensional.
    """
    Q = q_ext[0:N_SYMPLECTIC_DOF]
    P = q_ext[N_SYMPLECTIC_DOF : 2*N_SYMPLECTIC_DOF]
    X = q_ext[2*N_SYMPLECTIC_DOF : 3*N_SYMPLECTIC_DOF]
    Y = q_ext[3*N_SYMPLECTIC_DOF : 4*N_SYMPLECTIC_DOF]
    
    c = np.cos(2 * omega * delta)
    s = np.sin(2 * omega * delta)

    # Perform calculations using temporary arrays for intermediate results
    # to avoid issues with in-place updates on views if NumPy handles it subtely.
    q_plus_x  = Q + X
    q_minus_x = Q - X
    p_plus_y  = P + Y
    p_minus_y = P - Y
    
    # Store new values in temporary variables before assigning back to q_ext slices
    Q_new = 0.5 * (q_plus_x + c * q_minus_x + s * p_minus_y)
    P_new = 0.5 * (p_plus_y - s * q_minus_x + c * p_minus_y)
    X_new = 0.5 * (q_plus_x - c * q_minus_x - s * p_minus_y)
    Y_new = 0.5 * (p_plus_y + s * q_minus_x - c * p_minus_y)

    # Assign new values back to the slices of q_ext
    q_ext[0:N_SYMPLECTIC_DOF] = Q_new
    q_ext[N_SYMPLECTIC_DOF : 2*N_SYMPLECTIC_DOF] = P_new
    q_ext[2*N_SYMPLECTIC_DOF : 3*N_SYMPLECTIC_DOF] = X_new
    q_ext[3*N_SYMPLECTIC_DOF : 4*N_SYMPLECTIC_DOF] = Y_new

@njit(cache=False, fastmath=FASTMATH)
def _recursive_update_poly(
    q_ext: np.ndarray, 
    timestep: float, 
    order: int, 
    omega: float, 
    jac_H: List[List[np.ndarray]], 
    clmo_H: List[np.ndarray]
    ):
    """
    Apply recursive symplectic update of specified order.
    
    Parameters
    ----------
    q_ext : numpy.ndarray
        Extended state vector [Q, P, X, Y] to be updated in-place
    timestep : float
        Time step size
    order : int
        Order of the symplectic integrator (must be even and >= 2)
    omega : float
        Frequency parameter for the rotation
    jac_H : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as list of polynomial coefficients
    clmo_H : List[numpy.ndarray]
        List of coefficient layout mapping objects for the polynomials
        
    Notes
    -----
    For order=2, applies the basic second-order symplectic scheme:
        phi_a(delta/2) o phi_b(delta/2) o phi_c(delta) o phi_b(delta/2) o phi_a(delta/2)
    
    For higher orders, applies a recursive composition method with
    carefully chosen substeps to achieve the desired order of accuracy.
    """
    if order == 2:
        _phi_H_a_update_poly(q_ext, 0.5 * timestep, jac_H, clmo_H)
        _phi_H_b_update_poly(q_ext, 0.5 * timestep, jac_H, clmo_H)
        _phi_omega_H_c_update_poly(q_ext, timestep, omega)
        _phi_H_b_update_poly(q_ext, 0.5 * timestep, jac_H, clmo_H)
        _phi_H_a_update_poly(q_ext, 0.5 * timestep, jac_H, clmo_H)
    else:
        # Ensure float division for the exponent if order is large
        gamma = 1.0 / (2.0 - 2.0**(1.0 / (float(order) + 1.0)))
        lower_order = order - 2
        if lower_order < 2: # Ensure lower_order doesn't go below 2
            # This case should not be hit if initial order is >= 2 and even.
            # Or, handle error appropriately.
            pass 

        _recursive_update_poly(q_ext, gamma * timestep, lower_order, omega, jac_H, clmo_H)
        _recursive_update_poly(q_ext, (1.0 - 2.0 * gamma) * timestep, lower_order, omega, jac_H, clmo_H)
        _recursive_update_poly(q_ext, gamma * timestep, lower_order, omega, jac_H, clmo_H)


@njit(cache=False, fastmath=FASTMATH)
def _integrate_symplectic(
    initial_state_6d: np.ndarray,
    t_values: np.ndarray,
    jac_H: List[List[np.ndarray]],
    clmo_H: List[np.ndarray],
    order: int,
    c_omega_heuristic: float = 20.0
    ) -> np.ndarray:
    """
    Integrate Hamilton's equations using a high-order symplectic integrator
    for a system with N_SYMPLECTIC_DOF degrees of freedom (e.g., 3 DOF for a 6D phase space).
    
    Parameters
    ----------
    initial_state_6d : numpy.ndarray
        Initial state vector [Q, P] (e.g., [q1,q2,q3,p1,p2,p3]) 
        (shape: 2*N_SYMPLECTIC_DOF)
    t_values : numpy.ndarray
        Array of time points at which to compute the solution
    jac_H : List[List[np.ndarray]]
        Jacobian of Hamiltonian as a list of polynomial coefficients (for 6 variables)
    clmo_H : List[numpy.ndarray]
        List of coefficient layout mapping objects for the polynomials
    order : int
        Order of the symplectic integrator (must be even and >= 2)
    c_omega_heuristic : float, optional
        Scaling parameter for the frequency calculation, default is 20.0
        
    Returns
    -------
    numpy.ndarray
        Trajectory array of shape (len(t_values), 2*N_SYMPLECTIC_DOF)
        
    Notes
    -----
    Uses an extended phase-space technique to implement a high-order
    symplectic integration method for the polynomial Hamiltonian.
    
    The method is particularly suitable for center manifold dynamics where
    energy conservation over long time integration is crucial.
    
    The algorithm:
    1. Creates an extended phase space with auxiliary variables [Q, P, X, Y]
    2. Recursively applies composition of basic symplectic steps
    3. Returns trajectory only for the physical variables [Q, P]
    
    For optimal energy conservation, higher c_omega_heuristic values may be used
    at the cost of potentially smaller effective timesteps.
    """
    # Input validation (basic checks, more robust checks ideally in Python caller)
    valid_input = True
    if not (order > 0 and order % 2 == 0):
        valid_input = False
    if len(t_values) < 1:
        valid_input = False
    if initial_state_6d.shape[0] != 2 * N_SYMPLECTIC_DOF:
        valid_input = False
    
    if not valid_input:
        raise

    num_output_timesteps = len(t_values)
    trajectory = np.empty((num_output_timesteps, 2 * N_SYMPLECTIC_DOF), dtype=np.float64)
    
    if num_output_timesteps == 0:
        return trajectory
        
    trajectory[0, :] = initial_state_6d.copy()

    if num_output_timesteps == 1:
        return trajectory

    q_ext = np.empty(4 * N_SYMPLECTIC_DOF, dtype=np.float64)
    q_ext[0:N_SYMPLECTIC_DOF] = initial_state_6d[0:N_SYMPLECTIC_DOF].copy()
    q_ext[N_SYMPLECTIC_DOF : 2*N_SYMPLECTIC_DOF] = initial_state_6d[N_SYMPLECTIC_DOF : 2*N_SYMPLECTIC_DOF].copy()
    q_ext[2*N_SYMPLECTIC_DOF : 3*N_SYMPLECTIC_DOF] = initial_state_6d[0:N_SYMPLECTIC_DOF].copy()
    q_ext[3*N_SYMPLECTIC_DOF : 4*N_SYMPLECTIC_DOF] = initial_state_6d[N_SYMPLECTIC_DOF : 2*N_SYMPLECTIC_DOF].copy()

    timesteps_to_integrate = np.diff(t_values)

    for i in range(len(timesteps_to_integrate)):
        dt = timesteps_to_integrate[i]
    
        omega = _get_tao_omega(dt, order, c_omega_heuristic)
        
        _recursive_update_poly(q_ext, dt, order, omega, jac_H, clmo_H)
        trajectory[i + 1, 0:N_SYMPLECTIC_DOF] = q_ext[0:N_SYMPLECTIC_DOF].copy()
        trajectory[i + 1, N_SYMPLECTIC_DOF : 2*N_SYMPLECTIC_DOF] = q_ext[N_SYMPLECTIC_DOF : 2*N_SYMPLECTIC_DOF].copy()

    return trajectory


@njit(cache=False, fastmath=FASTMATH)
def _integrate_symplectic_until_event(
    initial_state_6d: np.ndarray,
    t_values: np.ndarray,
    jac_H: List[List[np.ndarray]],
    clmo_H: List[np.ndarray],
    order: int,
    event_fn,
    direction: int,
    xtol: float,
    gtol: float,
    c_omega_heuristic: float = 20.0
):
    """Integrate with symplectic method until an event is detected.
    
    This function performs symplectic integration and monitors an event
    function for zero crossings. When a crossing is detected, it uses
    Hermite interpolation to refine the event location within the step.
    
    Parameters
    ----------
    initial_state_6d : numpy.ndarray
        Initial state vector [Q, P] (shape: 2*N_SYMPLECTIC_DOF)
    t_values : numpy.ndarray
        Array of time points at which to check for events
    jac_H : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as list of polynomial coefficients
    clmo_H : List[numpy.ndarray]
        List of coefficient layout mapping objects
    order : int
        Order of the symplectic integrator
    event_fn : callable
        Compiled event function g(t, y) -> float
    direction : int
        Event crossing direction: -1 (negative), 0 (any), +1 (positive)
    xtol : float
        Tolerance on event time location
    gtol : float
        Tolerance on event function value
    c_omega_heuristic : float, optional
        Scaling parameter for frequency calculation
        
    Returns
    -------
    hit : bool
        True if event was detected, False otherwise
    t_hit : float
        Time of event (or final time if no event)
    y_hit : numpy.ndarray
        State at event (or final state if no event)
    trajectory : numpy.ndarray
        Trajectory up to (but not including) the event
        
    Notes
    -----
    The main integration remains fully symplectic. Event refinement
    uses Hermite interpolation which is NOT symplectic, but provides
    accurate event location within the last step.
    
    The symplectic structure is preserved throughout the trajectory
    except for the final interpolated point when an event is detected.
    """
    num_timesteps = len(t_values)
    trajectory = np.empty((num_timesteps, 2 * N_SYMPLECTIC_DOF), dtype=np.float64)
    
    # Initialize
    trajectory[0, :] = initial_state_6d.copy()
    y_old = initial_state_6d.copy()
    t_old = t_values[0]
    
    # Split into Q and P for derivative evaluation
    Q_old = y_old[0:N_SYMPLECTIC_DOF]
    P_old = y_old[N_SYMPLECTIC_DOF:2*N_SYMPLECTIC_DOF]
    
    # Evaluate derivative and event function at initial point
    f_old = _eval_hamiltonian_derivative(Q_old, P_old, jac_H, clmo_H)
    g_old = event_fn(t_old, y_old)
    
    # Initialize extended phase space for symplectic integration
    q_ext = np.empty(4 * N_SYMPLECTIC_DOF, dtype=np.float64)
    q_ext[0:N_SYMPLECTIC_DOF] = Q_old.copy()
    q_ext[N_SYMPLECTIC_DOF:2*N_SYMPLECTIC_DOF] = P_old.copy()
    q_ext[2*N_SYMPLECTIC_DOF:3*N_SYMPLECTIC_DOF] = Q_old.copy()
    q_ext[3*N_SYMPLECTIC_DOF:4*N_SYMPLECTIC_DOF] = P_old.copy()
    
    # Integration loop
    for i in range(num_timesteps - 1):
        dt = t_values[i + 1] - t_values[i]
        t_new = t_values[i + 1]
        
        # Take symplectic step in extended phase space
        omega = _get_tao_omega(dt, order, c_omega_heuristic)
        _recursive_update_poly(q_ext, dt, order, omega, jac_H, clmo_H)
        
        # Extract physical state [Q, P] from extended space
        y_new = np.empty(2 * N_SYMPLECTIC_DOF, dtype=np.float64)
        y_new[0:N_SYMPLECTIC_DOF] = q_ext[0:N_SYMPLECTIC_DOF].copy()
        y_new[N_SYMPLECTIC_DOF:2*N_SYMPLECTIC_DOF] = q_ext[N_SYMPLECTIC_DOF:2*N_SYMPLECTIC_DOF].copy()
        
        # Evaluate derivative and event function at new point
        Q_new = y_new[0:N_SYMPLECTIC_DOF]
        P_new = y_new[N_SYMPLECTIC_DOF:2*N_SYMPLECTIC_DOF]
        f_new = _eval_hamiltonian_derivative(Q_new, P_new, jac_H, clmo_H)
        g_new = event_fn(t_new, y_new)
        
        # Check for event crossing
        crossed = _event_crossed(g_old, g_new, direction)
        
        if crossed:
            # Event detected - refine location using Hermite interpolation
            t_hit, y_hit = _hermite_refine_event_symplectic(
                event_fn, t_old, y_old, f_old,
                t_new, y_new, f_new,
                dt, direction, xtol, gtol
            )
            # Return trajectory up to (but not including) event
            return True, t_hit, y_hit, trajectory[:i+1]
        
        # No event detected - continue integration
        trajectory[i + 1, :] = y_new
        y_old = y_new
        t_old = t_new
        f_old = f_new
        g_old = g_new
    
    # No event detected in entire interval
    return False, t_values[-1], trajectory[-1], trajectory


class _ExtendedSymplectic(_Integrator):
    """
    High-order explicit Tao symplectic integrator for polynomial
    Hamiltonian systems.

    Parameters
    ----------
    order : int, optional
        Even order of the underlying scheme (>= 2). Default is 6.
    c_omega_heuristic : float, optional
        Scaling coefficient used in the heuristic
        :func:`~hiten.algorithms.integrators.symplectic._get_tao_omega` frequency.  
        Default is 20.0.
    **options
        Additional keyword options stored verbatim in
        :attr:`~hiten.algorithms.integrators.base._Integrator.options`.

    Attributes
    ----------
    name : str
        Human-readable identifier, e.g. ``"Symplectic6"``.
    order : int
        Same as the *order* constructor argument.
    c_omega_heuristic : float
        Same as the constructor argument.

    Examples
    --------
    >>> from hiten.algorithms.integrators.symplectic import _ExtendedSymplectic
    >>> integrator = _ExtendedSymplectic(order=8, c_omega_heuristic=25.0)
    >>> sol = integrator.integrate(hamiltonian_system, y0, t_vals)

    Notes
    -----
    The target *system* must expose three public attributes:

    - **jac_H** - Jacobian of the Hamiltonian given as a nested list of
        polynomial coefficient arrays compatible with
        :func:`~hiten.algorithms.polynomial.operations._polynomial_evaluate`.
    - **clmo_H** - co-efficient layout mapping objects for the same
        polynomials.
    - **n_dof** - number of degrees of freedom (must equal 3 for this
        implementation).
    """
    
    def __init__(self, order: int = 6, c_omega_heuristic: float = 20.0, **options):
        if order < 2 or order % 2 != 0:
            raise ValueError(f"Symplectic integrator order must be even and >= 2, got {order}")
        
        super().__init__(f"Symplectic{order}", **options)
        self._order = order
        self.c_omega_heuristic = c_omega_heuristic
    
    @property
    def order(self) -> int:
        """
        Order of accuracy of the symplectic method.
        """
        return self._order
    
    def validate_system(self, system: _HamiltonianSystemProtocol) -> None:
        """
        Validate that the system is compatible with symplectic integration.
        
        Parameters
        ----------
        system : :class:`~hiten.algorithms.dynamics.protocols._HamiltonianSystemProtocol`
            The system to validate
            
        Raises
        ------
        ValueError
            If the system doesn't provide the required Hamiltonian structure
        """
        super().validate_system(system)
        
        # Check for required Hamiltonian system attributes
        required_attrs = ['jac_H', 'clmo_H', 'n_dof']
        missing_attrs = [attr for attr in required_attrs if not hasattr(system, attr)]
        
        if missing_attrs:
            raise ValueError(
                f"System must provide Hamiltonian structure for symplectic integration. "
                f"Missing attributes: {missing_attrs}"
            )
        
        # Validate dimensions
        if hasattr(system, 'dim') and system.dim != 2 * system.n_dof:
            raise ValueError(
                f"System dimension {system.dim} must equal 2 * n_dof = {2 * system.n_dof}"
            )
    
    def integrate(
        self,
        system: _HamiltonianSystemProtocol,
        y0: np.ndarray,
        t_vals: np.ndarray,
        *,
        event_fn=None,
        event_cfg: EventConfig | None = None,
        event_options: "EventOptions | None" = None,
        **kwargs
    ) -> _Solution:
        """
        Integrate the Hamiltonian system using symplectic method with event detection.
        
        Parameters
        ----------
        system : :class:`~hiten.algorithms.dynamics.protocols._HamiltonianSystemProtocol`
            The Hamiltonian system to integrate (must provide polynomial structure)
        y0 : numpy.ndarray
            Initial state vector [Q, P], shape (2*n_dof,)
        t_vals : numpy.ndarray
            Array of time points at which to evaluate the solution
        event_fn : Callable[[float, numpy.ndarray], float], optional
            Scalar event function evaluated as ``g(t, y)``. A zero
            crossing may terminate integration or mark an event.
        event_cfg : :class:`~hiten.algorithms.types.configs.EventConfig` | None
            Configuration controlling event directionality and terminal behavior.
        event_options : :class:`~hiten.algorithms.types.options.EventOptions` | None
            Runtime tuning options controlling event detection tolerances.
        **kwargs
            Additional integration options (currently unused)
            
        Returns
        -------
        :class:`~hiten.algorithms.integrators.base._Solution`
            Integration results containing times and states
            
        Notes
        -----
        This method delegates to the existing _integrate_symplectic function,
        which expects the system to provide:
        - jac_H: Jacobian polynomial coefficients
        - clmo_H: Coefficient layout mapping objects  
        - n_dof: Number of degrees of freedom
        
        The integration preserves the symplectic structure exactly, making it
        ideal for long-term integration of Hamiltonian systems where energy
        conservation is critical.
        
        When event detection is enabled, the main trajectory remains symplectic.
        Event refinement uses Hermite interpolation which is accurate but not
        symplectic within the final step.
        """
        # Validate inputs and system compatibility
        self.validate_inputs(system, y0, t_vals)
        
        # Extract required data from the Hamiltonian system
        jac_H_typed = system.jac_H
        clmo_H = system.clmo_H
        n_dof = system.n_dof
        
        # Validate state vector dimension
        expected_dim = 2 * n_dof
        if len(y0) != expected_dim:
            raise ValueError(
                f"Initial state dimension {len(y0)} != expected {expected_dim} (2*n_dof)"
            )

        # The sign of time direction is carried by the *system* wrapper via
        # attribute ``_fwd`` (set to +-1 in _DirectedSystem).  We keep the
        # user-supplied time grid strictly ascending and inject the sign into
        # the integration through a transformed copy that the low-level
        # integrator sees.  This avoids forcing callers to reverse the grid.

        fwd = getattr(system, "_fwd", 1)

        # Provide a signed version of the time array for the low-level routine.
        t_vals_int = t_vals if fwd == 1 else t_vals * (-1.0)

        # Event-enabled path: check for zero crossings and refine with Hermite interpolation
        if event_fn is not None:
            event_compiled = self._compile_event_function(event_fn)
            
            # Extract event configuration parameters
            direction = 0 if event_cfg is None else int(event_cfg.direction)
            xtol = float(event_options.xtol if event_options is not None else 1.0e-12)
            gtol = float(event_options.gtol if event_options is not None else 1.0e-12)
            
            # Integrate until event
            hit, t_hit, y_hit, trajectory = _integrate_symplectic_until_event(
                initial_state_6d=y0,
                t_values=t_vals_int,
                jac_H=jac_H_typed,
                clmo_H=clmo_H,
                order=self._order,
                event_fn=event_compiled,
                direction=direction,
                xtol=xtol,
                gtol=gtol,
                c_omega_heuristic=self.c_omega_heuristic,
            )
            
            if hit:
                # Event detected - return trajectory to event
                # Apply time direction sign
                t_hit_out = t_hit * fwd
                times_out = np.array([t_vals[0], t_hit_out], dtype=np.float64)
                states_out = np.vstack([y0, y_hit])
                return _Solution(times=times_out, states=states_out)
            else:
                # No event - return full trajectory
                times_out = t_vals[:trajectory.shape[0]].copy() * fwd
                return _Solution(times=times_out, states=trajectory)

        # Standard non-event path
        trajectory_array = _integrate_symplectic(
            initial_state_6d=y0,
            t_values=t_vals_int,
            jac_H=jac_H_typed,
            clmo_H=clmo_H,
            order=self._order,
            c_omega_heuristic=self.c_omega_heuristic,
        )

        # Return times with the intended sign convention (multiplying back).
        times_out = t_vals.copy() * fwd

        return _Solution(times=times_out, states=trajectory_array)


class ExtendedSymplectic:
    """Implement a factory for extended symplectic integrators.

    This factory class provides a convenient way to create symplectic
    integrators of different orders without directly instantiating the
    underlying implementation classes.

    Parameters
    ----------
    order : int, default 6
        Order of the symplectic method. Must be 2, 4, 6, or 8.
    **opts
        Additional options passed to the integrator constructor.

    Returns
    -------
    :class:`~hiten.algorithms.integrators.symplectic._ExtendedSymplectic`
        A symplectic integrator instance of the specified order.

    Raises
    ------
    ValueError
        If the specified order is not supported.

    Examples
    --------
    >>> integrator = ExtendedSymplectic(order=6)
    >>> solution = integrator.integrate(hamiltonian_system, y0, t_vals)
    """
    _map = {2: _ExtendedSymplectic, 4: _ExtendedSymplectic, 6: _ExtendedSymplectic, 8: _ExtendedSymplectic}
    def __new__(cls, order=6, **opts):
        """Create a symplectic integrator of the specified order.
        
        Parameters
        ----------
        order : int, default 6
            Order of the symplectic method. Must be 2, 4, 6, or 8.
        **opts
            Additional options passed to the integrator constructor.
            
        Returns
        -------
        :class:`~hiten.algorithms.integrators.symplectic._ExtendedSymplectic`
            A symplectic integrator instance of the specified order.
            
        Raises
        ------
        ValueError
            If the specified order is not supported.
        """
        if order not in cls._map:
            raise ValueError("Extended symplectic order not supported")
        return cls._map[order](order=order, **opts)