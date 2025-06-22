
import numpy as np
from numba import complex128, cuda, float64

from hiten.algorithms.polynomial.cuda.evaluate import (
    CUDAEvaluate, poly_evaluate_degree_device)


N_VARS = 6
N_DOF = 3
THREADS_PER_BLOCK = 256


@cuda.jit(device=True)
def hamiltonian_rhs_device(state, jac_coeffs_data, jac_metadata, 
                          clmo_flat, clmo_offsets, result):
    """
    Device function to compute Hamiltonian RHS for a single state.
    
    Parameters
    ----------
    state : array of float64, length 2*n_dof
        Current state vector [q1, q2, q3, p1, p2, p3]
    jac_coeffs_data : device array
        Flattened Jacobian polynomial coefficients
    jac_metadata : device array, shape (2*n_dof, max_degree+1, 2)
        Metadata for Jacobian polynomials
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
    result : array of float64, length 2*n_dof
        Output array for RHS values
    """
    # Convert state to complex for polynomial evaluation
    state_complex = cuda.local.array(N_VARS, dtype=complex128)
    for i in range(N_VARS):
        state_complex[i] = complex128(state[i])
    
    # Evaluate all Jacobian components
    max_degree = jac_metadata.shape[1] - 1
    
    # Compute dH/dQ (first n_dof components)
    for i in range(N_DOF):
        value = complex128(0.0)
        for degree in range(max_degree + 1):
            offset = jac_metadata[i, degree, 0]
            n_coeffs = jac_metadata[i, degree, 1]
            
            if n_coeffs > 0:
                coeffs_degree = jac_coeffs_data[offset:offset+n_coeffs]
                value += poly_evaluate_degree_device(
                    coeffs_degree, n_coeffs, degree, state_complex,
                    clmo_flat, clmo_offsets
                )
        
        # Store negative of dH/dQ for dp/dt
        result[N_DOF + i] = -value.real
    
    # Compute dH/dP (second n_dof components)
    for i in range(N_DOF):
        value = complex128(0.0)
        for degree in range(max_degree + 1):
            offset = jac_metadata[N_DOF + i, degree, 0]
            n_coeffs = jac_metadata[N_DOF + i, degree, 1]
            
            if n_coeffs > 0:
                coeffs_degree = jac_coeffs_data[offset:offset+n_coeffs]
                value += poly_evaluate_degree_device(
                    coeffs_degree, n_coeffs, degree, state_complex,
                    clmo_flat, clmo_offsets
                )
        
        # Store dH/dP for dq/dt
        result[i] = value.real


@cuda.jit
def hamiltonian_rhs_kernel(states, jac_coeffs_data, jac_metadata,
                          clmo_flat, clmo_offsets, rhs_results):
    """
    CUDA kernel to compute Hamiltonian RHS for multiple states.
    
    Parameters
    ----------
    states : device array, shape (n_states, 2*n_dof)
        State vectors
    jac_coeffs_data : device array
        Flattened Jacobian polynomial coefficients
    jac_metadata : device array, shape (2*n_dof, max_degree+1, 2)
        Metadata for Jacobian polynomials
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
    rhs_results : device array, shape (n_states, 2*n_dof)
        Output array for RHS values
    """
    tid = cuda.grid(1)
    
    if tid >= states.shape[0]:
        return
    
    # Load state into local memory
    state = cuda.local.array(6, dtype=float64)
    for i in range(2 * N_DOF):
        state[i] = states[tid, i]
    
    # Compute RHS
    rhs = cuda.local.array(6, dtype=float64)
    hamiltonian_rhs_device(state, jac_coeffs_data, jac_metadata,
                          clmo_flat, clmo_offsets, rhs)
    
    # Store result
    for i in range(2 * N_DOF):
        rhs_results[tid, i] = rhs[i]


@cuda.jit
def hamiltonian_rhs_single_kernel(state, jac_coeffs_data, jac_metadata,
                                 clmo_flat, clmo_offsets, rhs_result):
    """
    CUDA kernel to compute Hamiltonian RHS for a single state.
    Useful for device-side calls within other kernels.
    
    Parameters
    ----------
    state : device array, shape (2*n_dof,)
        State vector
    jac_coeffs_data : device array
        Flattened Jacobian polynomial coefficients
    jac_metadata : device array, shape (2*n_dof, max_degree+1, 2)
        Metadata for Jacobian polynomials
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
    rhs_result : device array, shape (2*n_dof,)
        Output array for RHS values
    """
    tid = cuda.grid(1)
    
    if tid == 0:  # Only first thread does the work
        hamiltonian_rhs_device(state, jac_coeffs_data, jac_metadata,
                              clmo_flat, clmo_offsets, rhs_result)


class HamiltonianRHSEvaluatorCUDA:
    """
    Helper class to manage Hamiltonian RHS evaluation on GPU.
    """
    def __init__(self, jac_H, clmo):
        """
        Initialize the evaluator with Jacobian polynomials.
        
        Parameters
        ----------
        jac_H : List[List[np.ndarray]]
            List of 2*n_dof Jacobian polynomial components
        clmo : List[np.ndarray]
            CLMO lookup table
        """
        # Use the polynomial evaluator to handle data flattening

        self.poly_evaluator = CUDAEvaluate(jac_H, clmo)
        self.n_components = len(jac_H)
        self.n_dof = self.n_components // 2
        
        # Store device pointers for kernel calls
        self.d_jac_coeffs_data = self.poly_evaluator.d_coeffs_data
        self.d_jac_metadata = self.poly_evaluator.d_coeffs_metadata
        self.d_clmo_flat = self.poly_evaluator.d_clmo_flat
        self.d_clmo_offsets = self.poly_evaluator.d_clmo_offsets
    
    def evaluate_batch(self, states):
        """
        Evaluate Hamiltonian RHS for multiple states.
        
        Parameters
        ----------
        states : np.ndarray, shape (n_states, 2*n_dof)
            State vectors
            
        Returns
        -------
        np.ndarray, shape (n_states, 2*n_dof)
            RHS values for each state
        """
        n_states = states.shape[0]
        
        # Transfer to device
        d_states = cuda.to_device(states.astype(np.float64))
        d_rhs_results = cuda.device_array_like(d_states)
        
        # Launch kernel
        threads_per_block = THREADS_PER_BLOCK
        blocks_per_grid = (n_states + threads_per_block - 1) // threads_per_block
        
        hamiltonian_rhs_kernel[blocks_per_grid, threads_per_block](
            d_states, self.d_jac_coeffs_data, self.d_jac_metadata,
            self.d_clmo_flat, self.d_clmo_offsets, d_rhs_results
        )
        
        return d_rhs_results.copy_to_host()
    
    def evaluate_single(self, state):
        """
        Evaluate Hamiltonian RHS for a single state.
        
        Parameters
        ----------
        state : np.ndarray, shape (2*n_dof,)
            State vector
            
        Returns
        -------
        np.ndarray, shape (2*n_dof,)
            RHS values
        """
        # For single evaluation, just reshape and use batch method
        states = state.reshape(1, -1)
        result = self.evaluate_batch(states)
        return result[0]
    
    def get_device_arrays(self):
        """
        Get device array pointers for use in other kernels.
        
        Returns
        -------
        tuple
            (jac_coeffs_data, jac_metadata, clmo_flat, clmo_offsets)
        """
        return (self.d_jac_coeffs_data, self.d_jac_metadata,
                self.d_clmo_flat, self.d_clmo_offsets)
