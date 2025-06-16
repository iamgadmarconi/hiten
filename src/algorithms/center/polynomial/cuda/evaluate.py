import numpy as np
from numba import complex128, cuda

N_VARS = 6
THREADS_PER_BLOCK = 256

@cuda.jit(device=True, inline=True)
def decode_multiindex_device(pos, degree, clmo_flat, clmo_offsets):
    """
    Device function to decode a packed multi-index.
    
    Parameters
    ----------
    pos : int
        Position in the clmo[degree] array
    degree : int
        Degree of the monomial
    clmo_flat : device array
        Flattened CLMO data
    clmo_offsets : device array
        Offsets for each degree in clmo_flat
    
    Returns
    -------
    tuple of 6 ints
        Exponents (k0, k1, k2, k3, k4, k5)
    """
    # Get the packed value from flattened CLMO array
    offset = clmo_offsets[degree]
    packed = clmo_flat[offset + pos]
    
    k1 =  packed        & 0x3F
    k2 = (packed >>  6) & 0x3F
    k3 = (packed >> 12) & 0x3F
    k4 = (packed >> 18) & 0x3F
    k5 = (packed >> 24) & 0x3F
    k0 = degree - (k1 + k2 + k3 + k4 + k5)
    
    return k0, k1, k2, k3, k4, k5


@cuda.jit(device=True)
def poly_evaluate_degree_device(coeffs, n_coeffs, degree, point, clmo_flat, clmo_offsets):
    """
    Device function to evaluate a homogeneous polynomial of given degree.
    
    Parameters
    ----------
    coeffs : device array
        Coefficient array for this degree
    n_coeffs : int
        Number of coefficients
    degree : int
        Degree of the polynomial
    point : device array
        Point at which to evaluate (length N_VARS)
    clmo_flat, clmo_offsets : device arrays
        CLMO lookup table data
        
    Returns
    -------
    complex128
        Value of the polynomial at the point
    """
    if n_coeffs == 0:
        return complex128(0.0)
    
    # Build power table in registers/local memory
    # For N_VARS=6 and reasonable degrees, this should fit in registers
    pow_table = cuda.local.array((N_VARS, 32), dtype=complex128)  # Assuming max degree < 32
    
    for v in range(N_VARS):
        pow_table[v, 0] = complex128(1.0)
        base = point[v]
        for e in range(1, degree + 1):
            pow_table[v, e] = pow_table[v, e-1] * base
    
    # Accumulate polynomial value
    result = complex128(0.0)
    
    for i in range(n_coeffs):
        coeff_val = coeffs[i]
        if coeff_val == 0.0:
            continue
            
        # Decode the multi-index
        k0, k1, k2, k3, k4, k5 = decode_multiindex_device(i, degree, clmo_flat, clmo_offsets)
        
        # Compute monomial value
        term_val = (pow_table[0, k0] * pow_table[1, k1] * 
                   pow_table[2, k2] * pow_table[3, k3] * 
                   pow_table[4, k4] * pow_table[5, k5])
        
        result += coeff_val * term_val
    
    return result


@cuda.jit
def polynomial_evaluate_kernel(points, coeffs_flat, coeffs_offsets, coeffs_sizes,
                             clmo_flat, clmo_offsets, max_degree, results):
    """
    CUDA kernel to evaluate polynomials at multiple points.
    
    Parameters
    ----------
    points : device array, shape (n_points, N_VARS)
        Points at which to evaluate
    coeffs_flat : device array
        Flattened coefficient arrays for all degrees
    coeffs_offsets : device array, shape (max_degree+1,)
        Offsets into coeffs_flat for each degree
    coeffs_sizes : device array, shape (max_degree+1,)
        Number of coefficients for each degree
    clmo_flat : device array
        Flattened CLMO lookup table
    clmo_offsets : device array
        Offsets into clmo_flat for each degree
    max_degree : int
        Maximum polynomial degree
    results : device array, shape (n_points,)
        Output array for results
    """
    tid = cuda.grid(1)
    
    if tid >= points.shape[0]:
        return
    
    # Load point into shared memory for better access pattern
    point = cuda.local.array(N_VARS, dtype=complex128)
    for i in range(N_VARS):
        point[i] = points[tid, i]
    
    # Evaluate polynomial by summing all homogeneous parts
    total = complex128(0.0)
    
    for degree in range(max_degree + 1):
        n_coeffs = coeffs_sizes[degree]
        if n_coeffs > 0:
            offset = coeffs_offsets[degree]
            # Create a view of coefficients for this degree
            coeffs_degree = coeffs_flat[offset:offset+n_coeffs]
            
            total += poly_evaluate_degree_device(
                coeffs_degree, n_coeffs, degree, point, 
                clmo_flat, clmo_offsets
            )
    
    results[tid] = total


@cuda.jit
def polynomial_evaluate_batch_kernel(points, poly_indices, coeffs_data, coeffs_metadata,
                                   clmo_flat, clmo_offsets, results):
    """
    CUDA kernel to evaluate different polynomials at different points.
    
    Parameters
    ----------
    points : device array, shape (n_evals, N_VARS)
        Points at which to evaluate
    poly_indices : device array, shape (n_evals,)
        Which polynomial to use for each evaluation
    coeffs_data : device array
        All polynomial coefficients packed together
    coeffs_metadata : device array, shape (n_polys, max_degree+1, 2)
        For each polynomial and degree: (offset, size)
    clmo_flat : device array
        Flattened CLMO lookup table
    clmo_offsets : device array
        Offsets into clmo_flat for each degree
    results : device array, shape (n_evals,)
        Output array for results
    """
    tid = cuda.grid(1)
    
    if tid >= points.shape[0]:
        return
    
    poly_idx = poly_indices[tid]
    max_degree = coeffs_metadata.shape[1] - 1
    
    # Load point
    point = cuda.local.array(N_VARS, dtype=complex128)
    for i in range(N_VARS):
        point[i] = points[tid, i]
    
    # Evaluate polynomial
    total = complex128(0.0)
    
    for degree in range(max_degree + 1):
        offset = coeffs_metadata[poly_idx, degree, 0]
        n_coeffs = coeffs_metadata[poly_idx, degree, 1]
        
        if n_coeffs > 0:
            coeffs_degree = coeffs_data[offset:offset+n_coeffs]
            total += poly_evaluate_degree_device(
                coeffs_degree, n_coeffs, degree, point,
                clmo_flat, clmo_offsets
            )
    
    results[tid] = total


class CUDAEvaluate:
    """
    Helper class to manage polynomial evaluation on GPU.
    """
    def __init__(self, poly_p_list, clmo_list):
        """
        Initialize the evaluator with polynomial coefficients and CLMO tables.
        
        Parameters
        ----------
        poly_p_list : List of List[np.ndarray]
            List of polynomials, each as a list of coefficient arrays by degree
        clmo_list : List[np.ndarray]
            CLMO lookup table
        """
        self.n_polys = len(poly_p_list)
        self.max_degree = max(len(p) - 1 for p in poly_p_list)
        
        # Flatten polynomial coefficients
        self._flatten_polynomials(poly_p_list)
        
        # Flatten CLMO table
        self._flatten_clmo(clmo_list)
        
        # Transfer to GPU
        self.d_coeffs_data = cuda.to_device(self.coeffs_data)
        self.d_coeffs_metadata = cuda.to_device(self.coeffs_metadata)
        self.d_clmo_flat = cuda.to_device(self.clmo_flat)
        self.d_clmo_offsets = cuda.to_device(self.clmo_offsets)
    
    def _flatten_polynomials(self, poly_p_list):
        """Flatten polynomial coefficients for GPU access."""
        # Calculate total size needed
        total_size = 0
        metadata = []
        
        for poly in poly_p_list:
            poly_metadata = []
            for degree in range(self.max_degree + 1):
                if degree < len(poly) and poly[degree].shape[0] > 0:
                    size = poly[degree].shape[0]
                    poly_metadata.append((total_size, size))
                    total_size += size
                else:
                    poly_metadata.append((0, 0))
            metadata.append(poly_metadata)
        
        # Allocate flattened array
        self.coeffs_data = np.zeros(total_size, dtype=np.complex128)
        self.coeffs_metadata = np.array(metadata, dtype=np.int32)
        
        # Copy data
        idx = 0
        for poly in poly_p_list:
            for degree in range(len(poly)):
                if poly[degree].shape[0] > 0:
                    size = poly[degree].shape[0]
                    self.coeffs_data[idx:idx+size] = poly[degree]
                    idx += size
    
    def _flatten_clmo(self, clmo_list):
        """Flatten CLMO table for GPU access."""
        # Calculate offsets and total size
        offsets = [0]
        total_size = 0
        
        for clmo_degree in clmo_list:
            total_size += len(clmo_degree)
            offsets.append(total_size)
        
        # Allocate and fill flattened array
        self.clmo_flat = np.zeros(total_size, dtype=np.int32)
        self.clmo_offsets = np.array(offsets[:-1], dtype=np.int32)
        
        idx = 0
        for clmo_degree in clmo_list:
            size = len(clmo_degree)
            if size > 0:
                self.clmo_flat[idx:idx+size] = clmo_degree
                idx += size
    
    def evaluate_single(self, poly_idx, points):
        """
        Evaluate a single polynomial at multiple points.
        
        Parameters
        ----------
        poly_idx : int
            Index of polynomial to evaluate
        points : np.ndarray, shape (n_points, N_VARS)
            Points at which to evaluate
            
        Returns
        -------
        np.ndarray, shape (n_points,)
            Polynomial values at the points
        """
        n_points = points.shape[0]
        
        # Prepare data
        d_points = cuda.to_device(points.astype(np.complex128))
        d_results = cuda.device_array(n_points, dtype=np.complex128)
        
        # Extract metadata for this polynomial and ensure contiguity
        coeffs_offsets_for_poly = np.ascontiguousarray(self.coeffs_metadata[poly_idx, :, 0])
        coeffs_sizes_for_poly = np.ascontiguousarray(self.coeffs_metadata[poly_idx, :, 1])

        d_coeffs_offsets = cuda.to_device(coeffs_offsets_for_poly)
        d_coeffs_sizes = cuda.to_device(coeffs_sizes_for_poly)
        
        # Launch kernel
        threads_per_block = THREADS_PER_BLOCK
        blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block
        
        polynomial_evaluate_kernel[blocks_per_grid, threads_per_block](
            d_points, self.d_coeffs_data, d_coeffs_offsets, d_coeffs_sizes,
            self.d_clmo_flat, self.d_clmo_offsets, self.max_degree, d_results
        )
        
        return d_results.copy_to_host()
    
    def evaluate_batch(self, poly_indices, points):
        """
        Evaluate different polynomials at different points.
        
        Parameters
        ----------
        poly_indices : np.ndarray, shape (n_evals,)
            Which polynomial to use for each evaluation
        points : np.ndarray, shape (n_evals, N_VARS)
            Points at which to evaluate
            
        Returns
        -------
        np.ndarray, shape (n_evals,)
            Polynomial values
        """
        n_evals = points.shape[0]
        
        # Transfer to device
        d_points = cuda.to_device(points.astype(np.complex128))
        d_poly_indices = cuda.to_device(poly_indices.astype(np.int32))
        d_results = cuda.device_array(n_evals, dtype=np.complex128)
        
        # Launch kernel
        threads_per_block = THREADS_PER_BLOCK
        blocks_per_grid = (n_evals + threads_per_block - 1) // threads_per_block
        
        polynomial_evaluate_batch_kernel[blocks_per_grid, threads_per_block](
            d_points, d_poly_indices, self.d_coeffs_data, self.d_coeffs_metadata,
            self.d_clmo_flat, self.d_clmo_offsets, d_results
        )
        
        return d_results.copy_to_host()


def polynomial_evaluate_cuda(poly_p, points, clmo):
    evaluator = CUDAEvaluate([poly_p], clmo)
    return evaluator.evaluate_single(0, points)