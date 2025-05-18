import numpy as np
from numba.typed import List

from algorithms.center.polynomial.operations import polynomial_jacobian
from algorithms.integrators.symplectic import integrate_symplectic


def generate_hamiltonian_flow(
    hamiltonian_poly_coeffs: List[np.ndarray],
    max_deg_hamiltonian: int,
    psi_table: np.ndarray,
    clmo_table: List[np.ndarray],
    encode_dict_list: List,
    initial_cm_state_4d: np.ndarray,
    t_values: np.ndarray,
    integrator_order: int,
    c_omega_heuristic: float = 20.0
) -> np.ndarray:

    jacobian_H_cm_rn = polynomial_jacobian(
        poly_coeffs=hamiltonian_poly_coeffs,
        original_max_deg=max_deg_hamiltonian,
        psi_table=psi_table,
        clmo_table=clmo_table,
        encode_dict_list=encode_dict_list
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
