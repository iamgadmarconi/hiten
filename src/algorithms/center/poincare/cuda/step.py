import math

import numpy as np
from numba import boolean, cuda, float64, int32

from algorithms.center.poincare.cuda.hrhs import hamiltonian_rhs_device
from algorithms.center.poincare.cuda.rk4 import (RK4IntegratorCUDA,
                                                 rk4_step_device)

# Constants
N_DOF = 3
THREADS_PER_BLOCK = 256

@cuda.jit(device=True)
def hermite_interpolate_device(y0, y1, dy0_dt, dy1_dt, alpha):
    """
    Cubic Hermite interpolation.
    
    Parameters
    ----------
    y0, y1 : float64
        Function values at t=0 and t=dt
    dy0_dt, dy1_dt : float64
        Derivatives scaled by dt
    alpha : float64
        Interpolation parameter (0 to 1)
    """
    h00 = (1.0 + 2.0*alpha) * (1.0 - alpha)**2
    h10 = alpha * (1.0 - alpha)**2
    h01 = alpha**2 * (3.0 - 2.0*alpha)
    h11 = alpha**2 * (alpha - 1.0)
    
    return h00 * y0 + h10 * dy0_dt + h01 * y1 + h11 * dy1_dt


@cuda.jit(device=True)
def poincare_step_device(q2, p2, p3, dt, jac_coeffs_data, jac_metadata,
                        clmo_flat, clmo_offsets, max_steps,
                        q2_out, p2_out, p3_out):
    """
    Device function to find a Poincaré crossing.
    
    Parameters
    ----------
    q2, p2, p3 : float64
        Initial state on the section (q3=0)
    dt : float64
        Integration time step
    jac_coeffs_data, jac_metadata : device arrays
        Jacobian polynomial data
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
    max_steps : int32
        Maximum integration steps
    q2_out, p2_out, p3_out : references to float64
        Output values at the crossing
        
    Returns
    -------
    int32
        1 if crossing found, 0 otherwise
    """
    # n_vars = 2 * N_DOF # This will be 6
    
    # Initialize state
    state_old = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
    state_new = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
    
    # Set initial conditions
    for i in range(6): # n_vars replaced with 6
        state_old[i] = 0.0
    state_old[1] = q2  # q2
    state_old[2] = 0.0  # q3 always zero on section
    state_old[N_DOF + 1] = p2  # p2
    state_old[N_DOF + 2] = p3  # p3
    
    # Integration loop
    for step in range(max_steps):
        # Perform RK4 step
        rk4_step_device(state_old, dt, jac_coeffs_data, jac_metadata,
                       clmo_flat, clmo_offsets, state_new)
        
        # Extract relevant values
        q3_old = state_old[2]
        q3_new = state_new[2]
        p3_new = state_new[N_DOF + 2]
        
        # Check for crossing: q3 changes sign AND p3 > 0
        if (q3_old * q3_new < 0.0) and (p3_new > 0.0):
            # Found a crossing - perform Hermite interpolation
            
            # 1) Linear first guess for crossing time
            alpha = q3_old / (q3_old - q3_new)
            
            # 2) Compute RHS at both endpoints
            rhs_old = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
            rhs_new = cuda.local.array(6, dtype=float64) # n_vars replaced with 6
            
            hamiltonian_rhs_device(state_old, jac_coeffs_data, jac_metadata,
                                 clmo_flat, clmo_offsets, rhs_old)
            hamiltonian_rhs_device(state_new, jac_coeffs_data, jac_metadata,
                                 clmo_flat, clmo_offsets, rhs_new)
            
            # 3) Hermite polynomial coefficients for q3
            m0 = rhs_old[2] * dt  # dq3/dt at t=0, scaled by dt
            m1 = rhs_new[2] * dt  # dq3/dt at t=dt, scaled by dt
            
            d = q3_old
            c = m0
            b = 3.0*(q3_new - q3_old) - (2.0*m0 + m1)
            a = 2.0*(q3_old - q3_new) + (m0 + m1)
            
            # 4) One Newton iteration to refine alpha
            f = ((a*alpha + b)*alpha + c)*alpha + d
            fp = (3.0*a*alpha + 2.0*b)*alpha + c
            if abs(fp) > 1e-12:  # Avoid division by zero
                alpha -= f / fp
            
            # Clamp alpha to [0, 1]
            alpha = max(0.0, min(1.0, alpha))
            
            # 5) Interpolate q2, p2, p3 using same cubic basis
            q2_out[0] = hermite_interpolate_device(
                state_old[1], state_new[1],
                rhs_old[1] * dt, rhs_new[1] * dt, alpha
            )
            
            p2_out[0] = hermite_interpolate_device(
                state_old[N_DOF + 1], state_new[N_DOF + 1],
                rhs_old[N_DOF + 1] * dt, rhs_new[N_DOF + 1] * dt, alpha
            )
            
            p3_old = state_old[N_DOF + 2]
            p3_out[0] = hermite_interpolate_device(
                p3_old, p3_new,
                rhs_old[N_DOF + 2] * dt, rhs_new[N_DOF + 2] * dt, alpha
            )
            
            return 1  # Success
        
        # Copy new state to old for next iteration
        for i in range(6): # n_vars replaced with 6
            state_old[i] = state_new[i]
    
    # No crossing found within max_steps
    return 0


@cuda.jit
def poincare_step_kernel(initial_q2, initial_p2, initial_p3, dt,
                        jac_coeffs_data, jac_metadata,
                        clmo_flat, clmo_offsets, max_steps,
                        success_flags, q2_out, p2_out, p3_out):
    """
    CUDA kernel to find Poincaré crossings for multiple initial conditions.
    
    Parameters
    ----------
    initial_q2, initial_p2, initial_p3 : device arrays, shape (n_seeds,)
        Initial conditions
    dt : float64
        Integration time step
    jac_coeffs_data, jac_metadata : device arrays
        Jacobian polynomial data
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
    max_steps : int32
        Maximum integration steps
    success_flags : device array, shape (n_seeds,)
        Output flags (1 if crossing found, 0 otherwise)
    q2_out, p2_out, p3_out : device arrays, shape (n_seeds,)
        Output values at crossings
    """
    tid = cuda.grid(1)
    
    if tid >= initial_q2.shape[0]:
        return
    
    # Get initial conditions for this thread
    q2 = initial_q2[tid]
    p2 = initial_p2[tid]
    p3 = initial_p3[tid]
    
    # Temporary storage for outputs
    q2_result = cuda.local.array(1, dtype=float64)
    p2_result = cuda.local.array(1, dtype=float64)
    p3_result = cuda.local.array(1, dtype=float64)
    
    # Find crossing
    success = poincare_step_device(
        q2, p2, p3, dt, jac_coeffs_data, jac_metadata,
        clmo_flat, clmo_offsets, max_steps,
        q2_result, p2_result, p3_result
    )
    
    # Store results
    success_flags[tid] = success
    if success == 1:
        q2_out[tid] = q2_result[0]
        p2_out[tid] = p2_result[0]
        p3_out[tid] = p3_result[0]
    else:
        q2_out[tid] = 0.0
        p2_out[tid] = 0.0
        p3_out[tid] = 0.0


@cuda.jit
def poincare_iterate_kernel(seeds_q2, seeds_p2, seeds_p3, dt,
                          jac_coeffs_data, jac_metadata,
                          clmo_flat, clmo_offsets, max_steps,
                          n_iterations, output_points, output_count):
    """
    CUDA kernel to iterate Poincaré map multiple times per seed.
    
    Parameters
    ----------
    seeds_q2, seeds_p2, seeds_p3 : device arrays, shape (n_seeds,)
        Initial seed values
    dt : float64
        Integration time step
    jac_coeffs_data, jac_metadata : device arrays
        Jacobian polynomial data
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
    max_steps : int32
        Maximum integration steps per crossing
    n_iterations : int32
        Number of crossings to find per seed
    output_points : device array, shape (max_output_points, 2)
        Output array for (q2, p2) pairs
    output_count : device array, shape (1,)
        Atomic counter for output array
    """
    tid = cuda.grid(1)
    
    if tid >= seeds_q2.shape[0]:
        return
    
    # Initialize current state
    q2 = seeds_q2[tid]
    p2 = seeds_p2[tid]
    p3 = seeds_p3[tid]
    
    # Temporary storage
    q2_new = cuda.local.array(1, dtype=float64)
    p2_new = cuda.local.array(1, dtype=float64)
    p3_new = cuda.local.array(1, dtype=float64)
    
    # Iterate to find multiple crossings
    for iter_idx in range(n_iterations):
        success = poincare_step_device(
            q2, p2, p3, dt, jac_coeffs_data, jac_metadata,
            clmo_flat, clmo_offsets, max_steps,
            q2_new, p2_new, p3_new
        )
        
        if success == 1:
            # Store the crossing point
            idx = cuda.atomic.add(output_count, 0, 1)
            if idx < output_points.shape[0]:
                output_points[idx, 0] = q2_new[0]
                output_points[idx, 1] = p2_new[0]
            
            # Update state for next iteration
            q2 = q2_new[0]
            p2 = p2_new[0]
            p3 = p3_new[0]
        else:
            # Failed to find crossing, stop iterating this seed
            break


class PoincareMapCUDA:
    """
    Helper class to compute Poincaré maps on GPU.
    """
    def __init__(self, jac_H, clmo):
        """
        Initialize with Jacobian polynomials.
        
        Parameters
        ----------
        jac_H : List[List[np.ndarray]]
            Jacobian polynomial components
        clmo : List[np.ndarray]
            CLMO lookup table
        """

        self.integrator = RK4IntegratorCUDA(jac_H, clmo)
        (self.d_jac_coeffs_data, self.d_jac_metadata,
         self.d_clmo_flat, self.d_clmo_offsets) = self.integrator.get_device_arrays()
    
    def find_crossings(self, initial_conditions, dt=1e-3, max_steps=20000):
        """
        Find Poincaré crossings for multiple initial conditions.
        
        Parameters
        ----------
        initial_conditions : np.ndarray, shape (n_seeds, 3)
            Initial (q2, p2, p3) values
        dt : float
            Integration time step
        max_steps : int
            Maximum integration steps per crossing
            
        Returns
        -------
        success_flags : np.ndarray, shape (n_seeds,)
            1 if crossing found, 0 otherwise
        crossings : np.ndarray, shape (n_seeds, 3)
            (q2', p2', p3') values at crossings
        """
        n_seeds = initial_conditions.shape[0]
        
        # Transfer to device
        d_q2 = cuda.to_device(initial_conditions[:, 0].astype(np.float64))
        d_p2 = cuda.to_device(initial_conditions[:, 1].astype(np.float64))
        d_p3 = cuda.to_device(initial_conditions[:, 2].astype(np.float64))
        
        # Allocate outputs
        d_success = cuda.device_array(n_seeds, dtype=np.int32)
        d_q2_out = cuda.device_array(n_seeds, dtype=np.float64)
        d_p2_out = cuda.device_array(n_seeds, dtype=np.float64)
        d_p3_out = cuda.device_array(n_seeds, dtype=np.float64)
        
        # Launch kernel
        threads_per_block = THREADS_PER_BLOCK
        blocks_per_grid = (n_seeds + threads_per_block - 1) // threads_per_block
        
        poincare_step_kernel[blocks_per_grid, threads_per_block](
            d_q2, d_p2, d_p3, dt,
            self.d_jac_coeffs_data, self.d_jac_metadata,
            self.d_clmo_flat, self.d_clmo_offsets, max_steps,
            d_success, d_q2_out, d_p2_out, d_p3_out
        )
        
        # Copy results
        success_flags = d_success.copy_to_host()
        crossings = np.column_stack([
            d_q2_out.copy_to_host(),
            d_p2_out.copy_to_host(),
            d_p3_out.copy_to_host()
        ])
        
        return success_flags, crossings
    
    def iterate_map(self, seeds, n_iterations, dt=1e-3, max_steps=20000):
        """
        Iterate the Poincaré map multiple times for each seed.
        
        Parameters
        ----------
        seeds : np.ndarray, shape (n_seeds, 3)
            Initial (q2, p2, p3) values
        n_iterations : int
            Number of crossings to find per seed
        dt : float
            Integration time step
        max_steps : int
            Maximum integration steps per crossing
            
        Returns
        -------
        np.ndarray, shape (n_points, 2)
            Collected (q2, p2) points from all iterations
        """
        n_seeds = seeds.shape[0]
        max_output_points = n_seeds * n_iterations
        
        # Transfer seeds to device
        d_q2 = cuda.to_device(seeds[:, 0].astype(np.float64))
        d_p2 = cuda.to_device(seeds[:, 1].astype(np.float64))
        d_p3 = cuda.to_device(seeds[:, 2].astype(np.float64))
        
        # Allocate output array and counter
        d_output_points = cuda.device_array((max_output_points, 2), dtype=np.float64)
        d_output_count = cuda.to_device(np.array([0], dtype=np.int32))
        
        # Launch kernel
        threads_per_block = min(THREADS_PER_BLOCK, n_seeds)
        blocks_per_grid = (n_seeds + threads_per_block - 1) // threads_per_block
        
        poincare_iterate_kernel[blocks_per_grid, threads_per_block](
            d_q2, d_p2, d_p3, dt,
            self.d_jac_coeffs_data, self.d_jac_metadata,
            self.d_clmo_flat, self.d_clmo_offsets, max_steps,
            n_iterations, d_output_points, d_output_count
        )
        
        # Get actual number of points found
        n_points = d_output_count.copy_to_host()[0]
        
        # Copy only the valid points
        if n_points > 0:
            output_points = d_output_points[:n_points].copy_to_host()
            return output_points
        else:
            return np.empty((0, 2), dtype=np.float64)
