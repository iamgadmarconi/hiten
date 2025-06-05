import numpy as np
import pytest
from numba.typed import List

from algorithms.center.coordinates import (_complexify_coordinates,
                                           _realify_coordinates)
from algorithms.center.manifold import center_manifold_real
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               encode_multiindex,
                                               init_index_tables)
from algorithms.center.polynomial.operations import polynomial_zero_list
from algorithms.center.transforms import complexify, phys2rn, realify
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

def test_complexify_realify_round_trip(cr3bp_data_fixture):
    """Test that complexification and realification are inverse operations."""
    data = cr3bp_data_fixture
    psi = data["psi"]
    clmo = data["clmo"]
    max_degree = data["max_degree"]
    
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
        complex_coords = _complexify_coordinates(real_coords, max_degree, psi, clmo)
        recovered_real_coords = _realify_coordinates(complex_coords, max_degree, psi, clmo)
        
        # Check that we recover the original coordinates within tolerance
        np.testing.assert_allclose(
            recovered_real_coords, real_coords, 
            rtol=TOL_TEST, atol=TOL_TEST,
            err_msg=f"Real->Complex->Real round trip failed for test case {i}: "
                   f"input={real_coords}, recovered={recovered_real_coords}"
        )


def test_realify_complexify_round_trip(cr3bp_data_fixture):
    """Test that realification and complexification are inverse operations."""
    data = cr3bp_data_fixture
    psi = data["psi"] 
    clmo = data["clmo"]
    max_degree = data["max_degree"]
    
    # Set random seed for reproducible tests
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
        real_coords = _realify_coordinates(complex_coords, max_degree, psi, clmo)
        recovered_complex_coords = _complexify_coordinates(real_coords, max_degree, psi, clmo)
        
        # Check that we recover the original coordinates within tolerance
        np.testing.assert_allclose(
            recovered_complex_coords, complex_coords,
            rtol=TOL_TEST, atol=TOL_TEST,
            err_msg=f"Complex->Real->Complex round trip failed for test case {i}: "
                   f"input={complex_coords}, recovered={recovered_complex_coords}"
        )


def test_coordinate_vs_polynomial_transforms(cr3bp_data_fixture):
    """Test that coordinate transforms match polynomial transforms for degree-1 terms."""
    data = cr3bp_data_fixture
    psi = data["psi"]
    clmo = data["clmo"]
    max_degree = data["max_degree"]
    
    # Set random seed for reproducible tests
    np.random.seed(RANDOM_SEED)
    
    # Create a test coordinate vector
    test_real_coords = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    
    # Method 1: Use coordinate transformation functions
    complex_coords_method1 = _complexify_coordinates(test_real_coords, max_degree, psi, clmo)
    
    # Method 2: Use polynomial transformation functions directly
    # Create polynomial with degree-1 terms only
    real_polys = polynomial_zero_list(max_degree, psi)
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    if len(real_polys) > 1:
        for i in range(6):
            if abs(test_real_coords[i]) > 1e-15:
                k = np.zeros(6, dtype=np.int64)
                k[i] = 1
                pos = encode_multiindex(k, 1, encode_dict_list)
                if 0 <= pos < real_polys[1].shape[0]:
                    real_polys[1][pos] = test_real_coords[i]
    
    # Transform using polynomial function
    complex_polys = complexify(real_polys, max_degree, psi, clmo)
    
    # Extract coordinates from polynomial
    complex_coords_method2 = np.zeros(6, dtype=np.complex128)
    if len(complex_polys) > 1:
        for i in range(6):
            k = np.zeros(6, dtype=np.int64)
            k[i] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            if 0 <= pos < complex_polys[1].shape[0]:
                complex_coords_method2[i] = complex_polys[1][pos]
    
    # Compare the two methods
    np.testing.assert_allclose(
        complex_coords_method1, complex_coords_method2,
        rtol=TOL_TEST, atol=TOL_TEST,
        err_msg=f"Coordinate transform doesn't match polynomial transform: "
               f"coord_method={complex_coords_method1}, poly_method={complex_coords_method2}"
    )
    
    # Test the reverse direction
    real_coords_method1 = _realify_coordinates(complex_coords_method1, max_degree, psi, clmo)
    
    # Use polynomial realify function
    real_polys_recovered = realify(complex_polys, max_degree, psi, clmo)
    
    # Extract coordinates
    real_coords_method2 = np.zeros(6, dtype=np.complex128)
    if len(real_polys_recovered) > 1:
        for i in range(6):
            k = np.zeros(6, dtype=np.int64)
            k[i] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            if 0 <= pos < real_polys_recovered[1].shape[0]:
                real_coords_method2[i] = real_polys_recovered[1][pos]
    
    # Compare the two methods
    np.testing.assert_allclose(
        real_coords_method1, real_coords_method2,
        rtol=TOL_TEST, atol=TOL_TEST,
        err_msg=f"Realify coordinate transform doesn't match polynomial transform: "
               f"coord_method={real_coords_method1}, poly_method={real_coords_method2}"
    )
    
    # Verify we get back to original coordinates (take real part since input was real)
    np.testing.assert_allclose(
        real_coords_method1.real, test_real_coords,
        rtol=TOL_TEST, atol=TOL_TEST,
        err_msg=f"Failed to recover original coordinates: "
               f"original={test_real_coords}, recovered={real_coords_method1.real}"
    )
    
    # Also verify that the imaginary parts are small for this real input case
    np.testing.assert_allclose(
        real_coords_method1.imag, np.zeros(6),
        rtol=TOL_TEST, atol=TOL_TEST,
        err_msg=f"Expected small imaginary parts for real input, got {real_coords_method1.imag}"
    )


def test_zero_coordinates_handling(cr3bp_data_fixture):
    """Test that zero coordinates are handled correctly."""
    data = cr3bp_data_fixture
    psi = data["psi"]
    clmo = data["clmo"]
    max_degree = data["max_degree"]
    
    # Test zero real coordinates
    zero_real = np.zeros(6, dtype=np.float64)
    complex_result = _complexify_coordinates(zero_real, max_degree, psi, clmo)
    real_result = _realify_coordinates(complex_result, max_degree, psi, clmo)
    
    np.testing.assert_allclose(
        real_result, zero_real,
        rtol=TOL_TEST, atol=TOL_TEST,
        err_msg="Zero real coordinates round trip failed"
    )
    
    # Test zero complex coordinates  
    zero_complex = np.zeros(6, dtype=np.complex128)
    real_result = _realify_coordinates(zero_complex, max_degree, psi, clmo)
    complex_result = _complexify_coordinates(real_result, max_degree, psi, clmo)
    
    np.testing.assert_allclose(
        complex_result, zero_complex,
        rtol=TOL_TEST, atol=TOL_TEST,
        err_msg="Zero complex coordinates round trip failed"
    )


def test_coordinate_transforms_consistency(cr3bp_data_fixture):
    """Test mathematical consistency of the coordinate transformations."""
    data = cr3bp_data_fixture
    psi = data["psi"]
    clmo = data["clmo"]
    max_degree = data["max_degree"]
    
    # Test specific transformations that should have known relationships
    # Based on the math in transforms.py:
    # y_rn = (q2 + i*p2)/√2, z_rn = (q3 + i*p3)/√2
    # py_rn = (i*q2 + p2)/√2, pz_rn = (i*q3 + p3)/√2
    # x_rn = q1, px_rn = p1
    
    sqrt2 = np.sqrt(2.0)
    
    # Test case: set y_rn = 1.0, all others = 0
    real_coords = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    complex_coords = _complexify_coordinates(real_coords, max_degree, psi, clmo)
    
    # Expected: q2 = 1/√2, p2 = -i/√2 (from inverse transformation)
    expected_q2 = 1.0/sqrt2
    expected_p2 = 1j/sqrt2  # This matches the realify transformation
    
    np.testing.assert_allclose(
        complex_coords[1].real, expected_q2, rtol=TOL_TEST, atol=TOL_TEST,
        err_msg=f"q2 real part incorrect: expected {expected_q2}, got {complex_coords[1].real}"
    )
    np.testing.assert_allclose(
        complex_coords[4], expected_p2, rtol=TOL_TEST, atol=TOL_TEST,
        err_msg=f"p2 incorrect: expected {expected_p2}, got {complex_coords[4]}"
    )
