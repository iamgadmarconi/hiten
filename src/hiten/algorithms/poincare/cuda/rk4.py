import numpy as np
from numba import cuda, float64

from hiten.algorithms.poincare.cuda.hrhs import (HamiltonianRHSEvaluatorCUDA,
                                                  hamiltonian_rhs_device)


THREADS_PER_BLOCK = 256

@cuda.jit(device=True)
def rk4_step_device(state, dt, jac_coeffs_data, jac_metadata,
                   clmo_flat, clmo_offsets, state_out):
    """
    Device function to perform a single RK4 step.
    
    Parameters
    ----------
    state : array of float64, length 2*n_dof
        Current state vector
    dt : float64
        Time step
    jac_coeffs_data : device array
        Flattened Jacobian polynomial coefficients
    jac_metadata : device array
        Metadata for Jacobian polynomials
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
    state_out : array of float64, length 2*n_dof
        Output array for new state
    """
    # n_vars = 2 * N_DOF # This will be 6
    
    # Allocate local arrays for RK4 stages
    k1 = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
    k2 = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
    k3 = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
    k4 = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
    temp_state = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
    
    # Stage 1: k1 = f(state)
    hamiltonian_rhs_device(state, jac_coeffs_data, jac_metadata,
                          clmo_flat, clmo_offsets, k1)
    
    # Stage 2: k2 = f(state + 0.5 * dt * k1)
    for i in range(6): # n_vars replaced with 6
        temp_state[i] = state[i] + 0.5 * dt * k1[i]
    hamiltonian_rhs_device(temp_state, jac_coeffs_data, jac_metadata,
                          clmo_flat, clmo_offsets, k2)
    
    # Stage 3: k3 = f(state + 0.5 * dt * k2)
    for i in range(6): # n_vars replaced with 6
        temp_state[i] = state[i] + 0.5 * dt * k2[i]
    hamiltonian_rhs_device(temp_state, jac_coeffs_data, jac_metadata,
                          clmo_flat, clmo_offsets, k3)
    
    # Stage 4: k4 = f(state + dt * k3)
    for i in range(6): # n_vars replaced with 6
        temp_state[i] = state[i] + dt * k3[i]
    hamiltonian_rhs_device(temp_state, jac_coeffs_data, jac_metadata,
                          clmo_flat, clmo_offsets, k4)
    
    # Combine: state_new = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    dt_over_6 = dt / 6.0
    for i in range(6): # n_vars replaced with 6
        state_out[i] = state[i] + dt_over_6 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])


@cuda.jit
def rk4_step_kernel(states, dt, jac_coeffs_data, jac_metadata,
                   clmo_flat, clmo_offsets, states_out):
    """
    CUDA kernel to perform RK4 steps for multiple states.
    
    Parameters
    ----------
    states : device array, shape (n_states, 2*n_dof)
        Current state vectors
    dt : float64
        Time step
    jac_coeffs_data : device array
        Flattened Jacobian polynomial coefficients
    jac_metadata : device array
        Metadata for Jacobian polynomials
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
    states_out : device array, shape (n_states, 2*n_dof)
        Output array for new states
    """
    tid = cuda.grid(1)
    
    if tid >= states.shape[0]:
        return
    
    # n_vars = 2 * N_DOF # This will be 6
    
    # Load state into local memory
    state = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
    for i in range(6): # n_vars replaced with 6
        state[i] = states[tid, i]
    
    # Perform RK4 step
    state_new = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
    rk4_step_device(state, dt, jac_coeffs_data, jac_metadata,
                   clmo_flat, clmo_offsets, state_new)
    
    # Store result
    for i in range(6): # n_vars replaced with 6
        states_out[tid, i] = state_new[i]


@cuda.jit
def rk4_trajectory_kernel(initial_states, t_values, jac_coeffs_data, jac_metadata,
                         clmo_flat, clmo_offsets, trajectories):
    """
    CUDA kernel to integrate trajectories using RK4.
    
    Parameters
    ----------
    initial_states : device array, shape (n_trajectories, 2*n_dof)
        Initial state vectors
    t_values : device array, shape (n_time_points,)
        Time points at which to evaluate (must include t=0)
    jac_coeffs_data : device array
        Flattened Jacobian polynomial coefficients
    jac_metadata : device array
        Metadata for Jacobian polynomials
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
    trajectories : device array, shape (n_trajectories, n_time_points, 2*n_dof)
        Output array for trajectories
    """
    tid = cuda.grid(1)
    
    if tid >= initial_states.shape[0]:
        return
    
    # n_vars = 2 * N_DOF # This will be 6
    n_time_points = t_values.shape[0]
    
    # Initialize current state
    state = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
    state_new = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
    
    # Copy initial state
    for i in range(6): # n_vars replaced with 6
        state[i] = initial_states[tid, i]
        trajectories[tid, 0, i] = state[i]
    
    # Integrate through time points
    for t_idx in range(1, n_time_points):
        dt = t_values[t_idx] - t_values[t_idx-1]
        
        # Perform RK4 step
        rk4_step_device(state, dt, jac_coeffs_data, jac_metadata,
                       clmo_flat, clmo_offsets, state_new)
        
        # Update state and store
        for i in range(6): # n_vars replaced with 6
            state[i] = state_new[i]
            trajectories[tid, t_idx, i] = state[i]


class RK4IntegratorCUDA:
    """
    Helper class to manage RK4 integration on GPU.
    """
    def __init__(self, jac_H, clmo):
        """
        Initialize the integrator with Jacobian polynomials.
        
        Parameters
        ----------
        jac_H : List[List[np.ndarray]]
            List of 2*n_dof Jacobian polynomial components
        clmo : List[np.ndarray]
            CLMO lookup table
        """

        self.rhs_evaluator = HamiltonianRHSEvaluatorCUDA(jac_H, clmo)
        self.n_dof = self.rhs_evaluator.n_dof
        
        # Get device arrays
        (self.d_jac_coeffs_data, self.d_jac_metadata,
         self.d_clmo_flat, self.d_clmo_offsets) = self.rhs_evaluator.get_device_arrays()
    
    def step(self, states, dt):
        """
        Perform a single RK4 step for multiple states.
        
        Parameters
        ----------
        states : np.ndarray, shape (n_states, 2*n_dof)
            Current state vectors
        dt : float
            Time step
            
        Returns
        -------
        np.ndarray, shape (n_states, 2*n_dof)
            New state vectors after RK4 step
        """
        n_states = states.shape[0]
        
        # Transfer to device
        d_states = cuda.to_device(states.astype(np.float64))
        d_states_out = cuda.device_array_like(d_states)
        
        # Launch kernel
        threads_per_block = THREADS_PER_BLOCK
        blocks_per_grid = (n_states + threads_per_block - 1) // threads_per_block
        
        rk4_step_kernel[blocks_per_grid, threads_per_block](
            d_states, dt, self.d_jac_coeffs_data, self.d_jac_metadata,
            self.d_clmo_flat, self.d_clmo_offsets, d_states_out
        )
        
        return d_states_out.copy_to_host()
    
    def integrate(self, initial_states, t_values):
        """
        Integrate trajectories from initial states.
        
        Parameters
        ----------
        initial_states : np.ndarray, shape (n_trajectories, 2*n_dof)
            Initial state vectors
        t_values : np.ndarray, shape (n_time_points,)
            Time points at which to evaluate (must start at 0)
            
        Returns
        -------
        np.ndarray, shape (n_trajectories, n_time_points, 2*n_dof)
            Trajectories evaluated at specified time points
        """
        n_trajectories = initial_states.shape[0]
        n_time_points = len(t_values)
        
        # Allocate output
        trajectories = np.zeros((n_trajectories, n_time_points, 2*self.n_dof))
        
        # Transfer to device
        d_initial_states = cuda.to_device(initial_states.astype(np.float64))
        d_t_values = cuda.to_device(t_values.astype(np.float64))
        d_trajectories = cuda.device_array(trajectories.shape, dtype=np.float64)
        
        # Launch kernel
        threads_per_block = THREADS_PER_BLOCK
        blocks_per_grid = (n_trajectories + threads_per_block - 1) // threads_per_block
        
        rk4_trajectory_kernel[blocks_per_grid, threads_per_block](
            d_initial_states, d_t_values, self.d_jac_coeffs_data, self.d_jac_metadata,
            self.d_clmo_flat, self.d_clmo_offsets, d_trajectories
        )
        
        return d_trajectories.copy_to_host()
    
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


# Convenience function
def rk4_step_cuda(states, dt, jac_H, clmo, n_dof=3):
    """
    Perform RK4 step for multiple states using CUDA.
    
    This is a convenience wrapper. For better performance with repeated
    integrations, create an RK4IntegratorCUDA instance and reuse it.
    """
    integrator = RK4IntegratorCUDA(jac_H, clmo)
    return integrator.step(states, dt)
