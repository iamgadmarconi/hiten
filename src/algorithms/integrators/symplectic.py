import numpy as np
from numba import njit
from numba.typed import List

from algorithms.center.polynomial.operations import polynomial_evaluate

N_CM_DOF = 2
N_VARS_POLY = 6
CM_Q_POLY_INDICES = np.array([1, 2], dtype=np.int64)
CM_P_POLY_INDICES = np.array([4, 5], dtype=np.int64)


@njit(fastmath=True, cache=True)
def _get_tao_omega (delta: float, order: int, c: float = 10.0) -> float:
    """
    Uses a heuristic to give an estimate for the binding coefficient omega 
    in the nonseparable Hamiltonian integrator.
    The rule from the Tao paper is that the timestep `delta` must be much
    smaller than `omega**(-1/order)`.  If we define `x is much smaller than y` 
    to be `c*x <= y` for a large positive constant c, then the condition is equivalent to

        omega <= (c*delta)**(-order),

    and we will use that bound as the heuristic assignment.
    """
    return (c * delta)**(-float(order))


@njit(cache=True)
def _construct_6d_eval_point(Q_cm_current_ndof: np.ndarray, P_cm_current_ndof: np.ndarray) -> np.ndarray:
    """
    Constructs the 6D point for polynomial evaluation from CM Q-variables and CM P-variables.
    The CM Hamiltonian H_cm_rn, while effectively a function of 2*N_CM_DOF variables,
    is still formally represented as a polynomial in N_VARS_POLY variables where
    coefficients for q1_cn and p1_cn related terms are zero.

    Args:
        Q_cm_current_ndof (np.ndarray): Current CM Q-variables [Q_cm1, Q_cm2, ... N_CM_DOF terms].
        P_cm_current_ndof (np.ndarray): Current CM P-variables [P_cm1, P_cm2, ... N_CM_DOF terms].

    Returns:
        np.ndarray: 6D point (q1,q2,q3,p1,p2,p3)_cn with q1_cn=0, p1_cn=0,
                    and other components filled from CM variables.
    """
    if Q_cm_current_ndof.shape[0] != N_CM_DOF or P_cm_current_ndof.shape[0] != N_CM_DOF:
        # This check is more for Numba's type inference and AOT compilation,
        # as it can't raise dynamic ValueErrors easily.
        # Consider how to handle errors if Numba context allows.
        pass

    point_6d = np.zeros(N_VARS_POLY, dtype=np.complex128) # Use complex for polynomial_evaluate

    # Map CM Q and P variables to their respective positions in the 6D vector
    for i in range(N_CM_DOF):
        point_6d[CM_Q_POLY_INDICES[i]] = Q_cm_current_ndof[i]
        point_6d[CM_P_POLY_INDICES[i]] = P_cm_current_ndof[i]
        
    # q1_cn (poly_idx 0) and p1_cn (poly_idx 3) are implicitly zero already.
    return point_6d

@njit(cache=True)
def _eval_dH_dQ_cm(
    Q_cm_eval_ndof: np.ndarray,
    P_cm_eval_ndof: np.ndarray,
    jac_H_cm_rn_typed: List[List[np.ndarray]], # List of (List of np.ndarray(complex128))
    clmo_H_typed: List[np.ndarray] # Numba typed List of np.ndarray(uint32)
    # polynomial_evaluate_func: Callable # Pass the actual function if not globally visible
) -> np.ndarray:
    """
    Evaluates [dH/dQ_cm1, dH/dQ_cm2, ...] at the given CM state.
    The order of derivatives corresponds to the order in CM_Q_POLY_INDICES.

    Args:
        Q_cm_eval_ndof (np.ndarray): CM Q-variables for evaluation.
        P_cm_eval_ndof (np.ndarray): CM P-variables for evaluation.
        jac_H_cm_rn_typed: Numba-typed Jacobian of H_cm_rn. jac_H_cm_rn_typed[k] is dH/d(poly_var_k).
        clmo_H_typed: Numba-typed CLMO table for H_cm_rn.
        # polynomial_evaluate_func: The actual polynomial_evaluate function.

    Returns:
        np.ndarray: Real parts of [dH/dQ_cm1, dH/dQ_cm2, ...].
    """

    eval_point_6d = _construct_6d_eval_point(Q_cm_eval_ndof, P_cm_eval_ndof)
    
    derivatives_Q_cm = np.empty(N_CM_DOF, dtype=np.float64)

    for i in range(N_CM_DOF):
        poly_var_index = CM_Q_POLY_INDICES[i]
        dH_dQi_poly = jac_H_cm_rn_typed[poly_var_index]
        val_dH_dQi = polynomial_evaluate(dH_dQi_poly, eval_point_6d, clmo_H_typed)
        derivatives_Q_cm[i] = val_dH_dQi.real
    
    return derivatives_Q_cm

@njit(cache=True)
def _eval_dH_dP_cm(
    Q_cm_eval_ndof: np.ndarray,
    P_cm_eval_ndof: np.ndarray,
    jac_H_cm_rn_typed: List[List[np.ndarray]],
    clmo_H_typed: List[np.ndarray]
    # polynomial_evaluate_func: Callable
) -> np.ndarray:
    """
    Evaluates [dH/dP_cm1, dH/dP_cm2, ...] at the given CM state.
    The order of derivatives corresponds to the order in CM_P_POLY_INDICES.
    (Structurally similar to _eval_dH_dQ_cm)

    Returns:
        np.ndarray: Real parts of [dH/dP_cm1, dH/dP_cm2, ...].
    """
    eval_point_6d = _construct_6d_eval_point(Q_cm_eval_ndof, P_cm_eval_ndof)
    
    derivatives_P_cm = np.empty(N_CM_DOF, dtype=np.float64)

    for i in range(N_CM_DOF):
        poly_var_index = CM_P_POLY_INDICES[i]
        dH_dPi_poly = jac_H_cm_rn_typed[poly_var_index]
        val_dH_dPi = polynomial_evaluate(dH_dPi_poly, eval_point_6d, clmo_H_typed)
        derivatives_P_cm[i] = val_dH_dPi.real
        
    return derivatives_P_cm

@njit(cache=True)
def _phi_H_a_update_poly(
    q_ext: np.ndarray, 
    delta: float, 
    jac_H_cm_rn_typed: List[List[np.ndarray]], 
    clmo_H_typed: List[np.ndarray]
    ):
    """
    Implements the phi_H_a update step for Tao's integrator.
    p -= delta * dH/dq(q,y)
    x += delta * dH/dp(q,y)

    Args:
        q_ext (np.ndarray): The 8D extended phase space state.
        delta (float): The time step for this update part.
        jac_H_cm_rn_typed: Typed Jacobian of H_cm_rn.
        clmo_H_typed: Typed CLMO table for H_cm_rn.
    """
    # Extract current Q, P, X, Y views (these are N_CM_DOF dimensional)
    Q_current = q_ext[0:N_CM_DOF]
    P_current = q_ext[N_CM_DOF : 2*N_CM_DOF]
    X_current = q_ext[2*N_CM_DOF : 3*N_CM_DOF]
    Y_current = q_ext[3*N_CM_DOF : 4*N_CM_DOF]

    # dH/dq(q,y) means evaluate dH/dQ_cm at (Q_current, Y_current)
    dH_dQ_at_QY = _eval_dH_dQ_cm(Q_current, Y_current, jac_H_cm_rn_typed, clmo_H_typed)
    # dH/dp(q,y) means evaluate dH/dP_cm at (Q_current, Y_current)
    dH_dP_at_QY = _eval_dH_dP_cm(Q_current, Y_current, jac_H_cm_rn_typed, clmo_H_typed)

    # Update P and X (modifies q_ext in place via views)
    P_current -= delta * dH_dQ_at_QY
    X_current += delta * dH_dP_at_QY

@njit(cache=True)
def _phi_H_b_update_poly(
    q_ext: np.ndarray, 
    delta: float, 
    jac_H_cm_rn_typed: List[List[np.ndarray]], 
    clmo_H_typed: List[np.ndarray]
    ):
    """
    Implements the phi_H_b update step for Tao's integrator.
    q += delta * dH/dp(x,p)
    y -= delta * dH/dq(x,p)

    Args:
        q_ext (np.ndarray): The 8D extended phase space state.
        delta (float): The time step for this update part.
        jac_H_cm_rn_typed: Typed Jacobian of H_cm_rn.
        clmo_H_typed: Typed CLMO table for H_cm_rn.
    """
    Q_current = q_ext[0:N_CM_DOF]
    P_current = q_ext[N_CM_DOF : 2*N_CM_DOF]
    X_current = q_ext[2*N_CM_DOF : 3*N_CM_DOF]
    Y_current = q_ext[3*N_CM_DOF : 4*N_CM_DOF]

    # dH/dp(x,p) means evaluate dH/dP_cm at (X_current, P_current)
    dH_dP_at_XP = _eval_dH_dP_cm(X_current, P_current, jac_H_cm_rn_typed, clmo_H_typed)
    # dH/dq(x,p) means evaluate dH/dQ_cm at (X_current, P_current)
    dH_dQ_at_XP = _eval_dH_dQ_cm(X_current, P_current, jac_H_cm_rn_typed, clmo_H_typed)
    
    # Update Q and Y (modifies q_ext in place via views)
    Q_current += delta * dH_dP_at_XP
    Y_current -= delta * dH_dQ_at_XP

@njit(cache=True)
def _phi_omega_H_c_update_poly(q_ext: np.ndarray, delta: float, omega: float):
    """
    Implements the phi_omega_H_c update step (linear rotation).

    Args:
        q_ext (np.ndarray): The 8D extended phase space state.
        delta (float): The time step for this update part.
        omega (float): The coupling constant.
    """
    # Create views for Q, P, X, Y for clarity
    Q = q_ext[0:N_CM_DOF]
    P = q_ext[N_CM_DOF : 2*N_CM_DOF]
    X = q_ext[2*N_CM_DOF : 3*N_CM_DOF]
    Y = q_ext[3*N_CM_DOF : 4*N_CM_DOF]
    
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
    q_ext[0:N_CM_DOF] = Q_new
    q_ext[N_CM_DOF : 2*N_CM_DOF] = P_new
    q_ext[2*N_CM_DOF : 3*N_CM_DOF] = X_new
    q_ext[3*N_CM_DOF : 4*N_CM_DOF] = Y_new

@njit(cache=True)
def _recursive_update_poly(
    q_ext: np.ndarray, 
    timestep: float, 
    order: int, 
    omega: float, 
    jac_H_cm_rn_typed: List[List[np.ndarray]], 
    clmo_H_typed: List[np.ndarray]
    ):
    """
    Recursively constructs the integrator step.
    Base case is order 2.
    """
    if order == 2:
        _phi_H_a_update_poly(q_ext, 0.5 * timestep, jac_H_cm_rn_typed, clmo_H_typed)
        _phi_H_b_update_poly(q_ext, 0.5 * timestep, jac_H_cm_rn_typed, clmo_H_typed)
        _phi_omega_H_c_update_poly(q_ext, timestep, omega)
        _phi_H_b_update_poly(q_ext, 0.5 * timestep, jac_H_cm_rn_typed, clmo_H_typed)
        _phi_H_a_update_poly(q_ext, 0.5 * timestep, jac_H_cm_rn_typed, clmo_H_typed)
    else:
        # Ensure float division for the exponent if order is large
        gamma = 1.0 / (2.0 - 2.0**(1.0 / (float(order) + 1.0)))
        lower_order = order - 2
        if lower_order < 2: # Ensure lower_order doesn't go below 2
            # This case should not be hit if initial order is >= 2 and even.
            # Or, handle error appropriately.
            pass 

        _recursive_update_poly(q_ext, gamma * timestep, lower_order, omega, jac_H_cm_rn_typed, clmo_H_typed)
        _recursive_update_poly(q_ext, (1.0 - 2.0 * gamma) * timestep, lower_order, omega, jac_H_cm_rn_typed, clmo_H_typed)
        _recursive_update_poly(q_ext, gamma * timestep, lower_order, omega, jac_H_cm_rn_typed, clmo_H_typed)


@njit(cache=True)
def integrate_symplectic(
    initial_cm_state_4d: np.ndarray,
    t_values: np.ndarray,
    jac_H_cm_rn_typed: List[List[np.ndarray]], # Numba typed List of (Numba typed List of np.ndarray(complex128))
    clmo_H_typed: List[np.ndarray], # Numba typed List of np.ndarray(uint32)
    order: int,
    c_omega_heuristic: float = 20.0 # Increased from 10.0 to 20.0 for better energy conservation
    ) -> np.ndarray:
    """
    Integrates Hamiltonian dynamics on the center manifold using Tao's method
    for non-separable polynomial Hamiltonians.

    Args:
        initial_cm_state_4d (np.ndarray): Initial state [Q_cm1, Q_cm2, P_cm1, P_cm2].
                                          Shape (2 * N_CM_DOF,).
        t_values (np.ndarray): Array of time points at which to output the state.
        jac_H_cm_rn_typed: Numba-typed Jacobian of H_cm_rn.
                           It's a List of N_VARS_POLY items,
                           where each item is a polynomial (List of np.ndarray coeffs).
        clmo_H_typed: Numba-typed CLMO table compatible with H_cm_rn and its Jacobian.
        order (int): Order of the integrator (must be a positive, even integer).
        c_omega_heuristic (float, optional): Factor 'c' for Tao's omega heuristic. Defaults to 20.0.

    Returns:
        np.ndarray: Trajectory array of shape (len(t_values), 2 * N_CM_DOF).
    """
    # Input validation (basic checks, more robust checks ideally in Python caller)
    valid_input = True
    if not (order > 0 and order % 2 == 0):
        valid_input = False
    if len(t_values) < 1:
        valid_input = False
    if initial_cm_state_4d.shape[0] != 2 * N_CM_DOF:
        valid_input = False
    
    if not valid_input:
        raise

    num_output_timesteps = len(t_values)
    trajectory = np.empty((num_output_timesteps, 2 * N_CM_DOF), dtype=np.float64)
    
    if num_output_timesteps == 0:
        return trajectory
        
    trajectory[0, :] = initial_cm_state_4d.copy()

    if num_output_timesteps == 1:
        return trajectory

    q_ext = np.empty(4 * N_CM_DOF, dtype=np.float64)
    q_ext[0:N_CM_DOF] = initial_cm_state_4d[0:N_CM_DOF].copy() 
    q_ext[N_CM_DOF : 2*N_CM_DOF] = initial_cm_state_4d[N_CM_DOF : 2*N_CM_DOF].copy()
    q_ext[2*N_CM_DOF : 3*N_CM_DOF] = initial_cm_state_4d[0:N_CM_DOF].copy()
    q_ext[3*N_CM_DOF : 4*N_CM_DOF] = initial_cm_state_4d[N_CM_DOF : 2*N_CM_DOF].copy()

    timesteps_to_integrate = np.diff(t_values)

    for i in range(len(timesteps_to_integrate)):
        dt = timesteps_to_integrate[i]
    
        omega = _get_tao_omega(dt, order, c_omega_heuristic)
        
        _recursive_update_poly(q_ext, dt, order, omega, jac_H_cm_rn_typed, clmo_H_typed)
        trajectory[i + 1, 0:N_CM_DOF] = q_ext[0:N_CM_DOF].copy()
        trajectory[i + 1, N_CM_DOF : 2*N_CM_DOF] = q_ext[N_CM_DOF : 2*N_CM_DOF].copy()

    return trajectory
