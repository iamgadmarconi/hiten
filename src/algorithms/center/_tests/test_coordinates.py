import numpy as np
import pytest

from algorithms.center.coordinates import (_complexify_coordinates,
                                           _realify_coordinates)
from algorithms.center.lie import inverse_lie_transform, forward_lie_transform
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
        complex_coords = _complexify_coordinates(real_coords)
        recovered_real_coords = _realify_coordinates(complex_coords)
        
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
        real_coords = _realify_coordinates(complex_coords)
        recovered_complex_coords = _complexify_coordinates(real_coords)
        
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
    complex_coords = _complexify_coordinates(real_coords)
    
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

def test_full_roundtrip_transformation(cr3bp_data_fixture):
    """
    Test full roundtrip transformation sequence following your exact specification:
    1. real coordinates -> 2. _complexify_coordinates -> 3. inverse_lie_transform -> 
    4. _realify_coordinates -> 5. _complexify_coordinates -> 6. forward_lie_transform -> 
    7. _realify_coordinates
    
    Note: inverse_lie_transform expects 4D center manifold coordinates, so we'll
    extract the center manifold part (q2, p2, q3, p3) from the 6D complex coordinates.
    """
    # Get test data from fixture
    point = cr3bp_data_fixture["point"]
    psi = cr3bp_data_fixture["psi"]
    clmo = cr3bp_data_fixture["clmo"]
    max_degree = cr3bp_data_fixture["max_degree"]
    
    # Get required polynomials from cache
    poly_G_total = point.cache_get(('generating_functions', max_degree))
    if poly_G_total is None:
        pytest.fail("Generating functions not available in cache")
    
    # Set random seed for reproducible tests
    np.random.seed(RANDOM_SEED)
    
    # Test with small center manifold coordinates (since inverse_lie_transform expects 4D)
    test_cm_coords = [
        np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float64),    # q2 only
        np.array([0.0, 0.01, 0.0, 0.0], dtype=np.float64),    # p2 only  
        np.array([0.0, 0.0, 0.01, 0.0], dtype=np.float64),    # q3 only
        np.array([0.0, 0.0, 0.0, 0.01], dtype=np.float64),    # p3 only
        np.array([0.005, 0.005, 0.0, 0.0], dtype=np.float64), # q2, p2
        np.array([0.0, 0.0, 0.005, 0.005], dtype=np.float64), # q3, p3
        np.random.uniform(-0.02, 0.02, 4).astype(np.float64), # small random 4D
    ]
    
    for i, original_cm_coords in enumerate(test_cm_coords):
        try:
            # Step 1: real center manifold coordinates (4D starting point)
            coords_1 = original_cm_coords.copy()
            
            # Step 2: real -> complex (4D center manifold -> 4D complex)
            coords_2 = coords_1.astype(np.complex128)
            
            # Step 3: complex 4D center manifold -> inverse lie transform -> 6D complex
            coords_3 = inverse_lie_transform(
                coords_2, poly_G_total, psi, clmo, max_degree, TOL_TEST
            )
            
            # Step 4: complex 6D -> real 6D
            coords_4 = _realify_coordinates(coords_3)
            
            # Step 5: real 6D -> complex 6D  
            coords_5 = _complexify_coordinates(coords_4)
            
            # Step 6: complex 6D -> forward lie transform -> 4D center manifold
            coords_6 = forward_lie_transform(
                coords_5, poly_G_total, psi, clmo, max_degree, TOL_TEST
            )
            
            # Step 7: complex 4D -> real 4D (final result)
            final_cm_coords = coords_6.real  # Take real part of 4D complex result
            
            # Verify roundtrip: final should match original
            np.testing.assert_allclose(
                final_cm_coords, original_cm_coords,
                rtol=1e-8, atol=1e-10,  # Relaxed tolerance due to multiple transformations
                err_msg=f"Full roundtrip failed for test case {i}: "
                       f"original={original_cm_coords}, final={final_cm_coords}, "
                       f"difference={final_cm_coords - original_cm_coords}"
            )
            
        except Exception as e:
            pytest.fail(f"Roundtrip transformation failed for test case {i} "
                       f"with coordinates {original_cm_coords}: {str(e)}")
