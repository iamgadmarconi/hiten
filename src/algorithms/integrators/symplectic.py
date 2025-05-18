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


@njit(cache=True)
def _construct_6d_eval_point(Q_cm_current_ndof: np.ndarray, P_cm_current_ndof: np.ndarray) -> np.ndarray:
    """
    Construct a 6D evaluation point from center manifold position and momentum vectors.
    
    Parameters
    ----------
    Q_cm_current_ndof : numpy.ndarray
        Position vector in center manifold coordinates (dimension N_CM_DOF)
    P_cm_current_ndof : numpy.ndarray
        Momentum vector in center manifold coordinates (dimension N_CM_DOF)
        
    Returns
    -------
    numpy.ndarray
        6D evaluation point for polynomial evaluation
        
    Notes
    -----
    Maps the reduced center manifold coordinates (Q,P) to their 
    corresponding positions in the full 6D state vector, where:
    - Center manifold Q variables map to indices 1,2 (q2,q3)
    - Center manifold P variables map to indices 4,5 (p2,p3)
    - Indices 0,3 (q1,p1) remain zero (not in center manifold)
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
) -> np.ndarray:
    """
    Evaluate derivatives of Hamiltonian with respect to position variables.
    
    Parameters
    ----------
    Q_cm_eval_ndof : numpy.ndarray
        Position vector at which to evaluate derivatives
    P_cm_eval_ndof : numpy.ndarray
        Momentum vector at which to evaluate derivatives
    jac_H_cm_rn_typed : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as list of polynomial coefficients
    clmo_H_typed : List[numpy.ndarray]
        List of coefficient layout mapping objects for the polynomials
        
    Returns
    -------
    numpy.ndarray
        Vector of partial derivatives ∂H/∂Q at the evaluation point
        
    Notes
    -----
    Computes the gradient of the Hamiltonian with respect to position variables
    by evaluating the appropriate polynomials in the Jacobian at the given point.
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
) -> np.ndarray:
    """
    Evaluate derivatives of Hamiltonian with respect to momentum variables.
    
    Parameters
    ----------
    Q_cm_eval_ndof : numpy.ndarray
        Position vector at which to evaluate derivatives
    P_cm_eval_ndof : numpy.ndarray
        Momentum vector at which to evaluate derivatives
    jac_H_cm_rn_typed : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as list of polynomial coefficients
    clmo_H_typed : List[numpy.ndarray]
        List of coefficient layout mapping objects for the polynomials
        
    Returns
    -------
    numpy.ndarray
        Vector of partial derivatives ∂H/∂P at the evaluation point
        
    Notes
    -----
    Computes the gradient of the Hamiltonian with respect to momentum variables
    by evaluating the appropriate polynomials in the Jacobian at the given point.
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
    Apply the first Hamiltonian splitting operator (φₐ) in the symplectic scheme.
    
    Parameters
    ----------
    q_ext : numpy.ndarray
        Extended state vector [Q, P, X, Y] to be updated in-place
    delta : float
        Time step size
    jac_H_cm_rn_typed : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as list of polynomial coefficients
    clmo_H_typed : List[numpy.ndarray]
        List of coefficient layout mapping objects for the polynomials
        
    Notes
    -----
    Implements the symplectic update step:
    - P ← P - δ·∂H/∂Q(Q,Y)
    - X ← X + δ·∂H/∂P(Q,Y)
    
    This modifies q_ext in-place through views/slices.
    """
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
    Apply the second Hamiltonian splitting operator (φᵦ) in the symplectic scheme.
    
    Parameters
    ----------
    q_ext : numpy.ndarray
        Extended state vector [Q, P, X, Y] to be updated in-place
    delta : float
        Time step size
    jac_H_cm_rn_typed : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as list of polynomial coefficients
    clmo_H_typed : List[numpy.ndarray]
        List of coefficient layout mapping objects for the polynomials
        
    Notes
    -----
    Implements the symplectic update step:
    - Q ← Q + δ·∂H/∂P(X,P)
    - Y ← Y - δ·∂H/∂Q(X,P)
    
    This modifies q_ext in-place through views/slices.
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
    Apply the rotation operator (φᶜ) in the symplectic scheme.
    
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
    """
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
    jac_H_cm_rn_typed : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as list of polynomial coefficients
    clmo_H_typed : List[numpy.ndarray]
        List of coefficient layout mapping objects for the polynomials
        
    Notes
    -----
    For order=2, applies the basic second-order symplectic scheme:
        φₐ(δ/2) ∘ φᵦ(δ/2) ∘ φᶜ(δ) ∘ φᵦ(δ/2) ∘ φₐ(δ/2)
    
    For higher orders, applies a recursive composition method with
    carefully chosen substeps to achieve the desired order of accuracy.
    The composition follows Yoshida's technique for constructing higher-order
    symplectic integrators.
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
    Integrate Hamilton's equations using a high-order symplectic integrator.
    
    Parameters
    ----------
    initial_cm_state_4d : numpy.ndarray
        Initial state vector [Q, P] in center manifold coordinates (shape: 2*N_CM_DOF)
    t_values : numpy.ndarray
        Array of time points at which to compute the solution
    jac_H_cm_rn_typed : List[List[numpy.ndarray]]
        Jacobian of Hamiltonian as a list of polynomial coefficients
    clmo_H_typed : List[numpy.ndarray]
        List of coefficient layout mapping objects for the polynomials
    order : int
        Order of the symplectic integrator (must be even and >= 2)
    c_omega_heuristic : float, optional
        Scaling parameter for the frequency calculation, default is 20.0
        
    Returns
    -------
    numpy.ndarray
        Trajectory array of shape (len(t_values), 2*N_CM_DOF)
        
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
