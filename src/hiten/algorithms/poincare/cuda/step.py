import numpy as np
from numba import cuda, float64

from hiten.algorithms.poincare.cuda.hrhs import hamiltonian_rhs_device
from hiten.algorithms.poincare.cuda.rk4 import (RK4IntegratorCUDA,
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
def get_section_value_device(state, section_coord_int):
    """Return the section coordinate value."""
    if section_coord_int == 0:  # "q2"
        return state[1]
    elif section_coord_int == 1:  # "p2"
        return state[4]
    elif section_coord_int == 2:  # "q3"
        return state[2]
    elif section_coord_int == 3:  # "p3"
        return state[5]
    else:
        return state[2]  # Default to q3


@cuda.jit(device=True)
def check_momentum_condition_device(state, rhs, section_coord_int):
    """Check momentum condition for crossing direction."""
    if section_coord_int == 0:  # "q2" section
        return state[N_DOF + 1] > 0.0  # p2 > 0
    elif section_coord_int == 1:  # "p2" section
        return rhs[1] > 0.0  # dq2/dt > 0
    elif section_coord_int == 2:  # "q3" section
        return state[N_DOF + 2] > 0.0  # p3 > 0
    elif section_coord_int == 3:  # "p3" section
        return rhs[2] > 0.0  # dq3/dt > 0
    else:
        return True  # Default


@cuda.jit(device=True)
def get_derivative_index_device(section_coord_int):
    """Get the derivative index for Hermite interpolation."""
    if section_coord_int == 0:  # "q2"
        return 1  # dq2/dt
    elif section_coord_int == 1:  # "p2"
        return N_DOF + 1  # dp2/dt
    elif section_coord_int == 2:  # "q3"
        return 2  # dq3/dt
    elif section_coord_int == 3:  # "p3"
        return N_DOF + 2  # dp3/dt
    else:
        return 2  # Default to q3


@cuda.jit(device=True)
def poincare_step_device(q2, p2, q3, p3, dt, jac_coeffs_data, jac_metadata,
                        clmo_flat, clmo_offsets, max_steps, section_coord_int,
                        q2_out, p2_out, p3_out, q3_out):
    """
    Device function to find a Poincaré crossing.
    
    Parameters
    ----------
    q2, p2, q3, p3 : float64
        Initial state components
    dt : float64
        Integration time step
    jac_coeffs_data, jac_metadata : device arrays
        Jacobian polynomial data
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
    max_steps : int32
        Maximum integration steps
    section_coord_int : int32
        Section coordinate: 0="q2", 1="p2", 2="q3", 3="p3"
    q2_out, p2_out, p3_out, q3_out : references to float64
        Output values at the crossing
        
    Returns
    -------
    int32
        1 if crossing found, 0 otherwise
    """
    # Initialize state
    state_old = cuda.local.array(6, dtype=float64)
    state_new = cuda.local.array(6, dtype=float64)
    
    # Set initial conditions
    for i in range(6):
        state_old[i] = 0.0
    state_old[1] = q2  # q2
    state_old[2] = q3  # q3
    state_old[N_DOF + 1] = p2  # p2
    state_old[N_DOF + 2] = p3  # p3
    
    # Integration loop
    for step in range(max_steps):
        # Perform RK4 step
        rk4_step_device(state_old, dt, jac_coeffs_data, jac_metadata,
                       clmo_flat, clmo_offsets, state_new)
        
        # Extract section values
        f_old = get_section_value_device(state_old, section_coord_int)
        f_new = get_section_value_device(state_new, section_coord_int)
        
        # Check for crossing: section coordinate changes sign
        if f_old * f_new < 0.0:
            # Compute RHS for momentum check
            rhs_new = cuda.local.array(6, dtype=float64)
            hamiltonian_rhs_device(state_new, jac_coeffs_data, jac_metadata,
                                 clmo_flat, clmo_offsets, rhs_new)
            
            # Check direction-dependent momentum condition
            momentum_check = check_momentum_condition_device(state_new, rhs_new, section_coord_int)
            
            if momentum_check:
                # Found a crossing - perform Hermite interpolation
                
                # 1) Linear first guess for crossing time
                alpha = f_old / (f_old - f_new)
                
                # 2) Compute RHS at both endpoints
                rhs_old = cuda.local.array(6, dtype=float64)
                
                hamiltonian_rhs_device(state_old, jac_coeffs_data, jac_metadata,
                                     clmo_flat, clmo_offsets, rhs_old)
                
                # 3) Get derivative index for section coordinate
                deriv_idx = get_derivative_index_device(section_coord_int)
                
                # 4) Hermite polynomial coefficients for section coordinate
                m0 = rhs_old[deriv_idx] * dt  # section derivative at t=0, scaled by dt
                m1 = rhs_new[deriv_idx] * dt  # section derivative at t=dt, scaled by dt
                
                d = f_old
                c = m0
                b = 3.0*(f_new - f_old) - (2.0*m0 + m1)
                a = 2.0*(f_old - f_new) + (m0 + m1)
                
                # 5) One Newton iteration to refine alpha
                f = ((a*alpha + b)*alpha + c)*alpha + d
                fp = (3.0*a*alpha + 2.0*b)*alpha + c
                if abs(fp) > 1e-12:  # Avoid division by zero
                    alpha -= f / fp
                
                # Clamp alpha to [0, 1]
                alpha = max(0.0, min(1.0, alpha))
                
                # 6) Interpolate all coordinates using same cubic basis
                q2_out[0] = hermite_interpolate_device(
                    state_old[1], state_new[1],
                    rhs_old[1] * dt, rhs_new[1] * dt, alpha
                )
                
                p2_out[0] = hermite_interpolate_device(
                    state_old[N_DOF + 1], state_new[N_DOF + 1],
                    rhs_old[N_DOF + 1] * dt, rhs_new[N_DOF + 1] * dt, alpha
                )
                
                q3_out[0] = hermite_interpolate_device(
                    state_old[2], state_new[2],
                    rhs_old[2] * dt, rhs_new[2] * dt, alpha
                )
                
                p3_out[0] = hermite_interpolate_device(
                    state_old[N_DOF + 2], state_new[N_DOF + 2],
                    rhs_old[N_DOF + 2] * dt, rhs_new[N_DOF + 2] * dt, alpha
                )
                
                return 1  # Success
        
        # Copy new state to old for next iteration
        for i in range(6):
            state_old[i] = state_new[i]
    
    # No crossing found within max_steps
    return 0


@cuda.jit
def poincare_step_kernel(initial_q2, initial_p2, initial_q3, initial_p3, dt,
                        jac_coeffs_data, jac_metadata,
                        clmo_flat, clmo_offsets, max_steps, section_coord_int,
                        success_flags, q2_out, p2_out, q3_out, p3_out):
    """
    CUDA kernel to find Poincaré crossings for multiple initial conditions.
    
    Parameters
    ----------
    initial_q2, initial_p2, initial_q3, initial_p3 : device arrays, shape (n_seeds,)
        Initial conditions
    dt : float64
        Integration time step
    jac_coeffs_data, jac_metadata : device arrays
        Jacobian polynomial data
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
    max_steps : int32
        Maximum integration steps
    section_coord_int : int32
        Section coordinate: 0="q2", 1="p2", 2="q3", 3="p3"
    success_flags : device array, shape (n_seeds,)
        Output flags (1 if crossing found, 0 otherwise)
    q2_out, p2_out, q3_out, p3_out : device arrays, shape (n_seeds,)
        Output values at crossings
    """
    tid = cuda.grid(1)
    
    if tid >= initial_q2.shape[0]:
        return
    
    # Get initial conditions for this thread
    q2 = initial_q2[tid]
    p2 = initial_p2[tid]
    q3 = initial_q3[tid]
    p3 = initial_p3[tid]
    
    # Temporary storage for outputs
    q2_result = cuda.local.array(1, dtype=float64)
    p2_result = cuda.local.array(1, dtype=float64)
    q3_result = cuda.local.array(1, dtype=float64)
    p3_result = cuda.local.array(1, dtype=float64)
    
    # Find crossing
    success = poincare_step_device(
        q2, p2, q3, p3, dt, jac_coeffs_data, jac_metadata,
        clmo_flat, clmo_offsets, max_steps, section_coord_int,
        q2_result, p2_result, p3_result, q3_result
    )
    
    # Store results
    success_flags[tid] = success
    if success == 1:
        q2_out[tid] = q2_result[0]
        p2_out[tid] = p2_result[0]
        q3_out[tid] = q3_result[0]
        p3_out[tid] = p3_result[0]
    else:
        q2_out[tid] = 0.0
        p2_out[tid] = 0.0
        q3_out[tid] = 0.0
        p3_out[tid] = 0.0


@cuda.jit
def poincare_iterate_kernel(seeds_q2, seeds_p2, seeds_q3, seeds_p3, dt,
                          jac_coeffs_data, jac_metadata,
                          clmo_flat, clmo_offsets, max_steps, section_coord_int,
                          n_iterations, output_points, output_count):
    """
    CUDA kernel to iterate Poincaré map multiple times per seed.
    
    Parameters
    ----------
    seeds_q2, seeds_p2, seeds_q3, seeds_p3 : device arrays, shape (n_seeds,)
        Initial seed values
    dt : float64
        Integration time step
    jac_coeffs_data, jac_metadata : device arrays
        Jacobian polynomial data
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
    max_steps : int32
        Maximum integration steps per crossing
    section_coord_int : int32
        Section coordinate: 0="q2", 1="p2", 2="q3", 3="p3"
    n_iterations : int32
        Number of crossings to find per seed
    output_points : device array, shape (max_output_points, 2)
        Output array for coordinate pairs
    output_count : device array, shape (1,)
        Atomic counter for output array
    """
    tid = cuda.grid(1)
    
    if tid >= seeds_q2.shape[0]:
        return
    
    # Initialize current state
    q2 = seeds_q2[tid]
    p2 = seeds_p2[tid]
    q3 = seeds_q3[tid]
    p3 = seeds_p3[tid]
    
    # Temporary storage
    q2_new = cuda.local.array(1, dtype=float64)
    p2_new = cuda.local.array(1, dtype=float64)
    q3_new = cuda.local.array(1, dtype=float64)
    p3_new = cuda.local.array(1, dtype=float64)
    
    # Iterate to find multiple crossings
    for iter_idx in range(n_iterations):
        success = poincare_step_device(
            q2, p2, q3, p3, dt, jac_coeffs_data, jac_metadata,
            clmo_flat, clmo_offsets, max_steps, section_coord_int,
            q2_new, p2_new, p3_new, q3_new
        )
        
        if success == 1:
            # Store the crossing point based on section type
            idx = cuda.atomic.add(output_count, 0, 1)
            if idx < output_points.shape[0]:
                if section_coord_int == 0:  # "q2" section -> (q3, p3)
                    output_points[idx, 0] = q3_new[0]
                    output_points[idx, 1] = p3_new[0]
                elif section_coord_int == 1:  # "p2" section -> (q3, p3)
                    output_points[idx, 0] = q3_new[0]
                    output_points[idx, 1] = p3_new[0]
                elif section_coord_int == 2:  # "q3" section -> (q2, p2)
                    output_points[idx, 0] = q2_new[0]
                    output_points[idx, 1] = p2_new[0]
                elif section_coord_int == 3:  # "p3" section -> (q2, p2)
                    output_points[idx, 0] = q2_new[0]
                    output_points[idx, 1] = p2_new[0]
            
            # Update state for next iteration based on section type
            if section_coord_int == 0:  # "q2" section
                q2 = 0.0  # q2 is fixed at 0
                p2 = p2_new[0]
                q3 = q3_new[0]
                p3 = p3_new[0]
            elif section_coord_int == 1:  # "p2" section
                q2 = q2_new[0]
                p2 = 0.0  # p2 is fixed at 0
                q3 = q3_new[0]
                p3 = p3_new[0]
            elif section_coord_int == 2:  # "q3" section
                q2 = q2_new[0]
                p2 = p2_new[0]
                q3 = 0.0  # q3 is fixed at 0
                p3 = p3_new[0]
            elif section_coord_int == 3:  # "p3" section
                q2 = q2_new[0]
                p2 = p2_new[0]
                q3 = q3_new[0]  # Keep q3 from crossing
                p3 = 0.0  # p3 is fixed at 0
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
    
    def _section_coord_to_int(self, section_coord):
        """Convert section coordinate string to integer for CUDA."""
        coord_map = {"q2": 0, "p2": 1, "q3": 2, "p3": 3}
        return coord_map.get(section_coord, 2)  # Default to q3
    
    def find_crossings(self, initial_conditions, dt=1e-3, max_steps=20000, section_coord="q3"):
        """
        Find Poincaré crossings for multiple initial conditions.
        
        Parameters
        ----------
        initial_conditions : np.ndarray, shape (n_seeds, 4)
            Initial (q2, p2, q3, p3) values
        dt : float
            Integration time step
        max_steps : int
            Maximum integration steps per crossing
        section_coord : str
            Section coordinate ("q2", "p2", "q3", or "p3")
            
        Returns
        -------
        success_flags : np.ndarray, shape (n_seeds,)
            1 if crossing found, 0 otherwise
        crossings : np.ndarray, shape (n_seeds, 4)
            (q2', p2', q3', p3') values at crossings
        """
        n_seeds = initial_conditions.shape[0]
        section_coord_int = self._section_coord_to_int(section_coord)
        
        # Transfer to device
        d_q2 = cuda.to_device(initial_conditions[:, 0].astype(np.float64))
        d_p2 = cuda.to_device(initial_conditions[:, 1].astype(np.float64))
        d_q3 = cuda.to_device(initial_conditions[:, 2].astype(np.float64))
        d_p3 = cuda.to_device(initial_conditions[:, 3].astype(np.float64))
        
        # Allocate outputs
        d_success = cuda.device_array(n_seeds, dtype=np.int32)
        d_q2_out = cuda.device_array(n_seeds, dtype=np.float64)
        d_p2_out = cuda.device_array(n_seeds, dtype=np.float64)
        d_q3_out = cuda.device_array(n_seeds, dtype=np.float64)
        d_p3_out = cuda.device_array(n_seeds, dtype=np.float64)
        
        # Launch kernel
        threads_per_block = THREADS_PER_BLOCK
        blocks_per_grid = (n_seeds + threads_per_block - 1) // threads_per_block
        
        poincare_step_kernel[blocks_per_grid, threads_per_block](
            d_q2, d_p2, d_q3, d_p3, dt,
            self.d_jac_coeffs_data, self.d_jac_metadata,
            self.d_clmo_flat, self.d_clmo_offsets, max_steps, section_coord_int,
            d_success, d_q2_out, d_p2_out, d_q3_out, d_p3_out
        )
        
        # Copy results
        success_flags = d_success.copy_to_host()
        crossings = np.column_stack([
            d_q2_out.copy_to_host(),
            d_p2_out.copy_to_host(),
            d_q3_out.copy_to_host(),
            d_p3_out.copy_to_host()
        ])
        
        return success_flags, crossings
    
    def iterate_map(self, seeds, n_iterations, dt=1e-3, max_steps=20000, section_coord="q3"):
        """
        Iterate the Poincaré map multiple times for each seed.
        
        Parameters
        ----------
        seeds : np.ndarray, shape (n_seeds, 4)
            Initial (q2, p2, q3, p3) values
        n_iterations : int
            Number of crossings to find per seed
        dt : float
            Integration time step
        max_steps : int
            Maximum integration steps per crossing
        section_coord : str
            Section coordinate ("q2", "p2", "q3", or "p3")
            
        Returns
        -------
        np.ndarray, shape (n_points, 2)
            Collected coordinate pairs from all iterations
        """
        n_seeds = seeds.shape[0]
        max_output_points = n_seeds * n_iterations
        section_coord_int = self._section_coord_to_int(section_coord)
        
        # Transfer seeds to device
        d_q2 = cuda.to_device(seeds[:, 0].astype(np.float64))
        d_p2 = cuda.to_device(seeds[:, 1].astype(np.float64))
        d_q3 = cuda.to_device(seeds[:, 2].astype(np.float64))
        d_p3 = cuda.to_device(seeds[:, 3].astype(np.float64))
        
        # Allocate output array and counter
        d_output_points = cuda.device_array((max_output_points, 2), dtype=np.float64)
        d_output_count = cuda.to_device(np.array([0], dtype=np.int32))
        
        # Launch kernel
        threads_per_block = min(THREADS_PER_BLOCK, n_seeds)
        blocks_per_grid = (n_seeds + threads_per_block - 1) // threads_per_block
        
        poincare_iterate_kernel[blocks_per_grid, threads_per_block](
            d_q2, d_p2, d_q3, d_p3, dt,
            self.d_jac_coeffs_data, self.d_jac_metadata,
            self.d_clmo_flat, self.d_clmo_offsets, max_steps, section_coord_int,
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
