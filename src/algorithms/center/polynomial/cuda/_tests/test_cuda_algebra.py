import numpy as np
import pytest
from numba import cuda

from algorithms.center.polynomial.cuda.operations import poly_mul_cuda
from algorithms.center.polynomial.algebra import _poly_mul
from algorithms.center.polynomial.base import CLMO_GLOBAL, ENCODE_DICT_GLOBAL, PSI_GLOBAL


def test_poly_mul_cuda_against_cpu_reference():
    """Test that GPU polynomial multiplication matches CPU reference implementation."""
    # Skip test if CUDA is not available
    if not cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Test parameters (matching the original test)
    deg_p = 4
    deg_q = 5
    rng = np.random.default_rng(0)
    
    # Host-side random blocks
    p_coeff = rng.standard_normal(CLMO_GLOBAL[deg_p].shape[0])
    q_coeff = rng.standard_normal(CLMO_GLOBAL[deg_q].shape[0])
    
    # Short-cut references
    p_exp = CLMO_GLOBAL[deg_p].astype(np.int32)
    q_exp = CLMO_GLOBAL[deg_q].astype(np.int32)
    
    # Create the encode_dict for the output degree
    deg_out = deg_p + deg_q
    encode_dict_out = np.full(2**30, -1, dtype=np.int32)
    
    # Populate the encode dict array from the dictionary
    if deg_out < len(ENCODE_DICT_GLOBAL):
        encode_dict = ENCODE_DICT_GLOBAL[deg_out]
        for packed_exp, pos in encode_dict.items():
            if packed_exp < len(encode_dict_out):
                encode_dict_out[packed_exp] = pos
    
    # GPU path
    n_valid = CLMO_GLOBAL[deg_out].shape[0]
    gpu_out = poly_mul_cuda(
        p_coeff,
        p_exp,
        q_coeff,
        q_exp,
        encode_dict_out,
        n_valid,
    )
    
    # CPU reference
    cpu_out = _poly_mul(
        p_coeff,
        deg_p,
        q_coeff,
        deg_q,
        PSI_GLOBAL,
        CLMO_GLOBAL,
        ENCODE_DICT_GLOBAL,
    )
    
    # Compare results
    if np.allclose(cpu_out, gpu_out, atol=1e-10):
        assert True, "GPU matches CPU reference to 1e-10"
    else:
        max_diff = np.max(np.abs(cpu_out - gpu_out))
        assert False, f"Result mismatch! Max abs diff: {max_diff}"


def test_poly_mul_cuda_basic_properties():
    """Test basic properties of the CUDA polynomial multiplication."""
    if not cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Simple test case
    deg_p = 2
    deg_q = 2
    
    # Create simple coefficient arrays
    p_coeff = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    q_coeff = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    
    # Get the corresponding exponent arrays
    p_exp = CLMO_GLOBAL[deg_p].astype(np.int32)
    q_exp = CLMO_GLOBAL[deg_q].astype(np.int32)
    
    # Create encode dict for output degree
    deg_out = deg_p + deg_q
    encode_dict_out = np.full(2**15, -1, dtype=np.int32)
    if deg_out < len(ENCODE_DICT_GLOBAL):
        encode_dict = ENCODE_DICT_GLOBAL[deg_out]
        for packed_exp, pos in encode_dict.items():
            if packed_exp < len(encode_dict_out):
                encode_dict_out[packed_exp] = pos
    
    # Run the multiplication
    n_valid = CLMO_GLOBAL[deg_out].shape[0]
    result = poly_mul_cuda(
        p_coeff,
        p_exp,
        q_coeff,
        q_exp,
        encode_dict_out,
        n_valid,
    )
    
    # Basic checks
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == p_coeff.dtype
    assert len(result) > 0


def test_poly_mul_cuda_smaller_example():
    """Test with a smaller, more controlled example."""
    if not cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Use smaller degrees for more predictable testing
    deg_p = 1
    deg_q = 1
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    # Create test data
    p_coeff = rng.standard_normal(CLMO_GLOBAL[deg_p].shape[0])
    q_coeff = rng.standard_normal(CLMO_GLOBAL[deg_q].shape[0])
    
    p_exp = CLMO_GLOBAL[deg_p].astype(np.int32)
    q_exp = CLMO_GLOBAL[deg_q].astype(np.int32)
    
    # Create encode dict for output degree
    deg_out = deg_p + deg_q
    encode_dict_out = np.full(2**30, -1, dtype=np.int32)
    if deg_out < len(ENCODE_DICT_GLOBAL):
        encode_dict = ENCODE_DICT_GLOBAL[deg_out]
        for packed_exp, pos in encode_dict.items():
            if packed_exp < len(encode_dict_out):
                encode_dict_out[packed_exp] = pos

    n_valid = CLMO_GLOBAL[deg_out].shape[0]
    # GPU computation
    gpu_out = poly_mul_cuda(
        p_coeff,
        p_exp,
        q_coeff,
        q_exp,
        encode_dict_out,
        n_valid,
    )
    
    # CPU reference
    cpu_out = _poly_mul(
        p_coeff,
        deg_p,
        q_coeff,
        deg_q,
        PSI_GLOBAL,
        CLMO_GLOBAL,
        ENCODE_DICT_GLOBAL,
    )
    
    # Verify they match
    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-12, 
                             err_msg="GPU and CPU results should match for small example")
