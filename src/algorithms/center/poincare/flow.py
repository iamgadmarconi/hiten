import numpy as np
from numba.typed import List

from algorithms.center.polynomial.operations import polynomial_jacobian
from algorithms.integrators.symplectic import integrate_symplectic
from algorithms.center.polynomial.base import init_index_tables # For MAX_DEG if passed directly for jacobian's psi/clmo

# It might be cleaner to pass psi and clmo directly if they are already computed
# instead of MAX_DEG and recomputing them, but let's stick to the core logic first.

def generate_hamiltonian_flow(
    hamiltonian_poly_coeffs: List[np.ndarray],
    max_deg_hamiltonian: int,
    psi_table: np.ndarray,
    clmo_table: List[np.ndarray],
    initial_cm_state_4d: np.ndarray,
    t_values: np.ndarray,
    integrator_order: int,
    c_omega_heuristic: float = 20.0
) -> np.ndarray:
    """
    Generates the Hamiltonian flow for a given Hamiltonian in polynomial form,
    restricted to the center manifold.

    Args:
        hamiltonian_poly_coeffs (List[np.ndarray]): The Hamiltonian represented as a list
                                                   of coefficient arrays by degree.
        max_deg_hamiltonian (int): The maximum degree of the Hamiltonian polynomial.
        psi_table (np.ndarray): The PSI table corresponding to the Hamiltonian's structure.
        clmo_table (List[np.ndarray]): The CLMO table corresponding to the Hamiltonian's structure.
        initial_cm_state_4d (np.ndarray): Initial state [Q_cm1, Q_cm2, P_cm1, P_cm2].
        t_values (np.ndarray): Array of time points at which to output the state.
        integrator_order (int): Order of the symplectic integrator (must be even and positive).
        c_omega_heuristic (float, optional): Factor 'c' for Tao's omega heuristic. Defaults to 20.0.

    Returns:
        np.ndarray: Trajectory array of shape (len(t_values), 4).
    """

    # 1. Compute the Jacobian of the Hamiltonian
    # The polynomial_jacobian function expects psi and clmo tables that are appropriate
    # for the derivatives as well. Assuming the input psi/clmo are for original_max_deg
    # and derivatives will have max_deg = original_max_deg - 1.
    # The current implementation of polynomial_jacobian allows passing the same psi/clmo
    # for both original and derivative if they are large enough.
    
    jacobian_H_cm_rn = polynomial_jacobian(
        poly_coeffs=hamiltonian_poly_coeffs,
        original_max_deg=max_deg_hamiltonian,
        psi_table=psi_table,
        clmo_table=clmo_table
    )

    # 2. Call the symplectic integrator
    trajectory = integrate_symplectic(
        initial_cm_state_4d=initial_cm_state_4d,
        t_values=t_values,
        jac_H_cm_rn_typed=jacobian_H_cm_rn, # This is List[List[np.ndarray]]
        clmo_H_typed=clmo_table,           # This is List[np.ndarray]
        order=integrator_order,
        c_omega_heuristic=c_omega_heuristic
    )

    return trajectory
