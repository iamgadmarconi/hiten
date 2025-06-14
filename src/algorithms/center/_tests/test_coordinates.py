import numpy as np
import pytest

from algorithms.center.coordinates import (solve_real,
                                           solve_complex)
from algorithms.center.manifold import center_manifold_real
from algorithms.center.polynomial.base import init_index_tables

from system.libration import L1Point

# Constants for tests
MU_EM = 0.0121505816  # Earth-Moon mass parameter (example)
MAX_DEGREE_TEST = 5   
TOL_TEST = 1e-15      
RANDOM_SEED = 42 # For reproducible random numbers

@pytest.fixture(scope="module")
def cr3bp_data_fixture():
    """
    Provides a real L1 point, psi, clmo, max_degree, and energy.
    """
    point = L1Point(mu=MU_EM)
    _ = point.position  # Ensures L1 position is calculated
    energy_val = point.energy 
    psi_arr, clmo_arr = init_index_tables(MAX_DEGREE_TEST)
    _ = center_manifold_real(point, psi_arr, clmo_arr, MAX_DEGREE_TEST)
    # Check that essential data was cached by precompute_cache
    poly_cm_cn_val = point.cache_get(('hamiltonian', MAX_DEGREE_TEST, 'center_manifold_complex'))
    if poly_cm_cn_val is None:
        pytest.fail("poly_cm ('center_manifold_complex') is None after precomputation.")
    
    poly_cm_rn_val = point.cache_get(('hamiltonian', MAX_DEGREE_TEST, 'center_manifold_real'))
    if poly_cm_rn_val is None:
        pytest.fail("poly_cm ('center_manifold_real') is None after precomputation.")

    poly_G_val = point.cache_get(('generating_functions', MAX_DEGREE_TEST))
    if poly_G_val is None:
        pytest.fail("Generating functions (poly_G_total) are None after precomputation.")

    return {
        "point": point,
        "psi": psi_arr,
        "clmo": clmo_arr,
        "max_degree": MAX_DEGREE_TEST,
        "energy_l1": energy_val, # Energy of the L1 point itself
        # Specific Hamiltonians are not returned here, tests will get them from point object
    }

def test_complexify_realify_round_trip():
    """Test that complexification and realification are inverse operations."""
    # Set random seed for reproducible tests
    np.random.seed(RANDOM_SEED)
    
    # Test with various real coordinate vectors
    test_real_coords = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),  # x-direction
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),  # y-direction
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),  # z-direction
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),  # px-direction
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),  # py-direction
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),  # pz-direction
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64),  # mixed small
        np.random.uniform(-1.0, 1.0, 6).astype(np.float64),          # random small
        np.random.uniform(-5.0, 5.0, 6).astype(np.float64),          # random larger
    ]
    
    for i, real_coords in enumerate(test_real_coords):
        # Real -> Complex -> Real round trip
        complex_coords = solve_real(real_coords)
        recovered_real_coords = solve_complex(complex_coords)
        
        # Check that we recover the original coordinates within tolerance
        np.testing.assert_allclose(
            recovered_real_coords, real_coords, 
            rtol=TOL_TEST, atol=TOL_TEST,
            err_msg=f"Real->Complex->Real round trip failed for test case {i}: "
                   f"input={real_coords}, recovered={recovered_real_coords}"
        )

def test_realify_complexify_round_trip():
    """Test that realification and complexification are inverse operations."""
    np.random.seed(RANDOM_SEED)
    
    # Test with various complex coordinate vectors
    test_complex_coords = [
        np.array([1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j], dtype=np.complex128),  # q1-direction
        np.array([0.0+0.0j, 1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j], dtype=np.complex128),  # q2-direction  
        np.array([0.0+0.0j, 0.0+0.0j, 1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j], dtype=np.complex128),  # q3-direction
        np.array([0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 1.0+0.0j, 0.0+0.0j, 0.0+0.0j], dtype=np.complex128),  # p1-direction
        np.array([0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 1.0+0.0j, 0.0+0.0j], dtype=np.complex128),  # p2-direction
        np.array([0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 1.0+0.0j], dtype=np.complex128),  # p3-direction
        np.array([0.1+0.1j, 0.2+0.2j, 0.3+0.3j, 0.4+0.4j, 0.5+0.5j, 0.6+0.6j], dtype=np.complex128),  # mixed small
        (np.random.uniform(-1.0, 1.0, 6) + 1j * np.random.uniform(-1.0, 1.0, 6)).astype(np.complex128),  # random small
        (np.random.uniform(-5.0, 5.0, 6) + 1j * np.random.uniform(-5.0, 5.0, 6)).astype(np.complex128),  # random larger
    ]
    
    for i, complex_coords in enumerate(test_complex_coords):
        # Complex -> Real -> Complex round trip
        real_coords = solve_complex(complex_coords)
        recovered_complex_coords = solve_real(real_coords)
        
        # Check that we recover the original coordinates within tolerance
        np.testing.assert_allclose(
            recovered_complex_coords, complex_coords,
            rtol=TOL_TEST, atol=TOL_TEST,
            err_msg=f"Complex->Real->Complex round trip failed for test case {i}: "
                   f"input={complex_coords}, recovered={recovered_complex_coords}"
        )

def test_coordinate_transforms_consistency():
    """Test mathematical consistency of the coordinate transformations."""

    sqrt2 = np.sqrt(2.0)
    
    # Test case: set y_rn = 1.0, all others = 0
    real_coords = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    complex_coords = solve_real(real_coords)
    
    # Expected: q2 = 1/√2, p2 = -i/√2 (from inverse transformation)
    expected_q2 = 1.0/sqrt2
    expected_p2 = 1j/sqrt2  # This matches the substitute_real transformation
    
    np.testing.assert_allclose(
        complex_coords[1].real, expected_q2, rtol=TOL_TEST, atol=TOL_TEST,
        err_msg=f"q2 real part incorrect: expected {expected_q2}, got {complex_coords[1].real}"
    )
    np.testing.assert_allclose(
        complex_coords[4], expected_p2, rtol=TOL_TEST, atol=TOL_TEST,
        err_msg=f"p2 incorrect: expected {expected_p2}, got {complex_coords[4]}"
    )
