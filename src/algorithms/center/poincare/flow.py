import numpy as np
from numba.typed import List

from algorithms.center.polynomial.operations import polynomial_jacobian
from algorithms.integrators.symplectic import (N_SYMPLECTIC_DOF,
                                               integrate_symplectic)


def generate_hamiltonian_flow(
    hamiltonian_poly_coeffs: List[np.ndarray],
    max_deg_hamiltonian: int,
    psi_table: np.ndarray,
    clmo_table: List[np.ndarray],
    encode_dict_list: List,
    initial_cm_state_4d: np.ndarray, # Represents [q_cm1, q_cm2, p_cm1, p_cm2]
    t_values: np.ndarray,
    integrator_order: int,
    c_omega_heuristic: float = 20.0
) -> np.ndarray:
    """
    Generates the flow (trajectory) for a Hamiltonian system restricted 
    to its center manifold dynamics, using a 6D symplectic integrator.

    The input `initial_cm_state_4d` represents the 2-DOF state on the CM,
    which is embedded into a 6D state [0, q_cm1, q_cm2, 0, p_cm1, p_cm2]
    for the integrator. The Hamiltonian provided should be the one
    restricted to the center manifold (i.e., terms with q1, p1 are zero).

    Parameters
    ----------
    hamiltonian_poly_coeffs : List[np.ndarray]
        Polynomial coefficients of the restricted Hamiltonian (6-variable form).
    max_deg_hamiltonian : int
        Maximum degree of the Hamiltonian polynomial.
    psi_table : np.ndarray
        Combinatorial table from init_index_tables.
    clmo_table : List[np.ndarray]
        List of arrays containing packed multi-indices for the Hamiltonian.
    encode_dict_list : List
        Encoding dictionary list from _create_encode_dict_from_clmo.
    initial_cm_state_4d : np.ndarray
        Initial 4D state on the center manifold [q_cm1, q_cm2, p_cm1, p_cm2].
        These correspond to q2,q3,p2,p3 in the 6D polynomial evaluation context.
    t_values : np.ndarray
        Array of time points at which to compute the solution.
    integrator_order : int
        Order of the symplectic integrator.
    c_omega_heuristic : float, optional
        Scaling parameter for the integrator's frequency calculation, default is 20.0.

    Returns
    -------
    np.ndarray
        Trajectory array of shape (len(t_values), 2 * N_SYMPLECTIC_DOF), 
        representing [q1(t), q2(t), q3(t), p1(t), p2(t), p3(t)].
        q1(t) and p1(t) should remain zero if the Hamiltonian is correctly restricted.
    """

    # 1. Compute Jacobian of the restricted Hamiltonian
    # This jacobian is for the 6-variable polynomial representation
    jacobian_H_cm_rn = polynomial_jacobian(
        poly_p=hamiltonian_poly_coeffs,
        max_deg=max_deg_hamiltonian,
        psi_table=psi_table,
        clmo_table=clmo_table,
        encode_dict_list=encode_dict_list
    )

    # Prepare 6D initial state for the integrator
    # initial_cm_state_4d is [q_cm1, q_cm2, p_cm1, p_cm2]
    # This maps to a 6D state [0, q_cm1, q_cm2, 0, p_cm1, p_cm2]
    # where q_cm1 -> q2, q_cm2 -> q3, p_cm1 -> p2, p_cm2 -> p3 for the polynomial indices
    if initial_cm_state_4d.shape[0] != 4:
        raise ValueError("initial_cm_state_4d must be a 4-element array [q_cm1, q_cm2, p_cm1, p_cm2]")

    initial_state_6d_for_integrator = np.zeros(2 * N_SYMPLECTIC_DOF, dtype=np.float64)
    initial_state_6d_for_integrator[1] = initial_cm_state_4d[0]  # q2 <- q_cm1
    initial_state_6d_for_integrator[2] = initial_cm_state_4d[1]  # q3 <- q_cm2
    initial_state_6d_for_integrator[N_SYMPLECTIC_DOF + 1] = initial_cm_state_4d[2] # p2 <- p_cm1 (p1 is at index N_SYMPLECTIC_DOF)
    initial_state_6d_for_integrator[N_SYMPLECTIC_DOF + 2] = initial_cm_state_4d[3] # p3 <- p_cm2
    # q1 (index 0) and p1 (index N_SYMPLECTIC_DOF) remain 0.0

    # 2. Call the symplectic integrator
    trajectory_6d = integrate_symplectic(
        initial_state_6d=initial_state_6d_for_integrator,
        t_values=t_values,
        jac_H_rn_typed=jacobian_H_cm_rn, 
        clmo_H_typed=clmo_table,      
        order=integrator_order,
        c_omega_heuristic=c_omega_heuristic
    )

    return trajectory_6d
