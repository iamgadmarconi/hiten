import numpy as np
import pytest
from numba.typed import List

from algorithms.center.coordinates import (_cn2rn_coordinates,
                                           _complete_cm_coordinates,
                                           _rn2phys_coordinates)
from algorithms.center.lie import (_apply_inverse_lie_transforms,
                                   _apply_single_inverse_generator)
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               encode_multiindex,
                                               init_index_tables)
from algorithms.center.polynomial.operations import polynomial_zero_list
from algorithms.center.transforms import cn2rn, phys2rn, rn2cn
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
    
    try:
        # This should compute and cache all necessary Hamiltonian forms and generating functions
        point.precompute_cache([MAX_DEGREE_TEST], {MAX_DEGREE_TEST: psi_arr}, {MAX_DEGREE_TEST: clmo_arr})
    except Exception as e:
        pytest.fail(f"Failed during point.precompute_cache in fixture: {e}")
    
    # Check that essential data was cached by precompute_cache
    poly_cm_cn_val = point.get_cached_hamiltonian(MAX_DEGREE_TEST, "center_manifold_cn")
    if poly_cm_cn_val is None:
        pytest.fail("poly_cm ('center_manifold_cn') is None after precomputation.")
    
    poly_cm_rn_val = point.get_cached_hamiltonian(MAX_DEGREE_TEST, "center_manifold_rn")
    if poly_cm_rn_val is None:
        pytest.fail("poly_cm ('center_manifold_rn') is None after precomputation.")

    poly_G_val = point.get_cached_generating_functions(MAX_DEGREE_TEST)
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

def _rn2cn_coordinates(rn_coords: np.ndarray, max_degree: int, psi: np.ndarray, clmo: list) -> np.ndarray:
    """Helper: Converts real normal vector to complex normal vector using rn2cn transform."""
    rn_polys = polynomial_zero_list(max_degree, psi)
    encode_dict_list = _create_encode_dict_from_clmo(List(clmo))
    if len(rn_polys) > 1:
        for i in range(6):
            if abs(rn_coords[i]) > 1e-15:
                k = np.zeros(6, dtype=np.int64); k[i] = 1
                pos = encode_multiindex(k, 1, encode_dict_list)
                if 0 <= pos < rn_polys[1].shape[0]: rn_polys[1][pos] = rn_coords[i]
    cn_polys = rn2cn(rn_polys, max_degree, psi, clmo)
    cn_coords_rt = np.zeros(6, dtype=np.complex128)
    if len(cn_polys) > 1:
        for i in range(6):
            k = np.zeros(6, dtype=np.int64); k[i] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            if 0 <= pos < cn_polys[1].shape[0]: cn_coords_rt[i] = cn_polys[1][pos]
    return cn_coords_rt

def _phys2rn_coordinates(phys_coords: np.ndarray, point: L1Point, max_degree: int, psi: np.ndarray, clmo: list) -> np.ndarray:
    """Helper: Converts physical vector to real normal vector using phys2rn transform."""
    phys_polys = polynomial_zero_list(max_degree, psi)
    encode_dict_list = _create_encode_dict_from_clmo(List(clmo))
    if len(phys_polys) > 1:
        for i in range(6):
            if abs(phys_coords[i]) > 1e-15:
                k = np.zeros(6, dtype=np.int64); k[i] = 1
                pos = encode_multiindex(k, 1, encode_dict_list)
                if 0 <= pos < phys_polys[1].shape[0]: phys_polys[1][pos] = phys_coords[i]
    rn_polys = phys2rn(point, phys_polys, max_degree, psi, clmo)
    rn_coords_rt = np.zeros(6, dtype=np.float64)
    if len(rn_polys) > 1:
        for i in range(6):
            k = np.zeros(6, dtype=np.int64); k[i] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            if 0 <= pos < rn_polys[1].shape[0]: rn_coords_rt[i] = rn_polys[1][pos].real
    return rn_coords_rt

def test_coordinate_round_trip_rn_to_cn(cr3bp_data_fixture):
    """Test RN → CN → RN round trip (this should work)."""
    data = cr3bp_data_fixture
    psi = data["psi"]
    clmo = List(data["clmo"])
    max_degree = data["max_degree"]
    
    rn_coords_real = np.array([0.1, 0.05, 0.02, 0.01, 0.03, 0.01], dtype=np.float64)
    
    # Using the helper for rn -> cn vector transformation
    cn_coords = _rn2cn_coordinates(rn_coords_real, max_degree, psi, clmo)
    # Using the main module's function for cn -> rn vector transformation
    rn_coords_back = _cn2rn_coordinates(cn_coords, max_degree, psi, clmo)
    
    np.testing.assert_allclose(rn_coords_real, rn_coords_back, rtol=1e-12, atol=1e-12)

def test_coordinate_round_trip_physical(cr3bp_data_fixture):
    """Test Physical → RN → CN → RN → Physical round trip."""
    data = cr3bp_data_fixture
    point = data["point"]
    psi = data["psi"]
    clmo = List(data["clmo"])
    max_degree = data["max_degree"]
    
    phys_coords = np.array([0.01, 0.005, 0.002, 0.001, 0.003, 0.001], dtype=np.float64)
    
    # Using helpers and main module functions
    rn_coords = _phys2rn_coordinates(phys_coords, point, max_degree, psi, clmo)
    cn_coords = _rn2cn_coordinates(rn_coords, max_degree, psi, clmo)
    rn_coords_back = _cn2rn_coordinates(cn_coords, max_degree, psi, clmo)
    phys_coords_back = _rn2phys_coordinates(rn_coords_back, point, max_degree, psi, clmo)
    
    np.testing.assert_allclose(phys_coords, phys_coords_back, rtol=1e-10, atol=1e-10)

def test_transformation_matrices_are_inverses(cr3bp_data_fixture):
    """Test that the CN↔RN transformation matrices are proper inverses at polynomial level."""
    data = cr3bp_data_fixture
    psi = data["psi"]
    clmo = List(data["clmo"])
    max_degree = data["max_degree"]
    
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    for coord_idx in range(6):
        poly_coord = polynomial_zero_list(max_degree, psi)
        if max_degree >= 1:
            k = np.zeros(6, dtype=np.int64)
            k[coord_idx] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            poly_coord[1][pos] = 1.0 
        
        poly_cn = rn2cn(poly_coord, max_degree, psi, clmo) # rn_poly to cn_poly
        poly_rn_back = cn2rn(poly_cn, max_degree, psi, clmo) # cn_poly to rn_poly
        
        for deg in range(min(2, max_degree + 1)): 
            np.testing.assert_allclose(
                poly_coord[deg], poly_rn_back[deg], 
                rtol=1e-12, atol=1e-15,
                err_msg=f"Polynomial RN→CN→RN round trip failed for coord {coord_idx}, degree {deg}"
            )

def test_coordinate_mapping():
    """Debug the coordinate mapping in inverse Lie transforms."""
    
    # Test input - this should be [q2, p2, q3, p3]
    cm_coords_4d = np.array([0.01, 0.005, 0.0, 0.02], dtype=np.complex128)
    
    # How it's mapped to 6D
    coords_6d = np.zeros(6, dtype=np.complex128)
    coords_6d[1] = cm_coords_4d[0]  # q2
    coords_6d[2] = cm_coords_4d[2]  # q3  
    coords_6d[4] = cm_coords_4d[1]  # p2
    coords_6d[5] = cm_coords_4d[3]  # p3
    
    print(f"CM coords [q2,p2,q3,p3]: {cm_coords_4d}")
    print(f"6D coords [q1,q2,q3,p1,p2,p3]: {coords_6d}")
    
    # Key question: Does p3 (index 3 in cm_coords) map to coords_6d[5]?
    print(f"p3 value: {cm_coords_4d[3]} → coords_6d[5]: {coords_6d[5]}")
    
    return coords_6d

def test_inverse_lie_preserve_p3(cr3bp_data_fixture):
    """Check if inverse Lie transforms are preserving the p3 coordinate."""
    
    data = cr3bp_data_fixture
    point = data["point"]
    psi = data["psi"]
    clmo = List(data["clmo"])
    max_degree = data["max_degree"]

    # Start with simple test case
    cm_coords_4d = np.array([0.01, 0.005, 0.0, 0.02], dtype=np.complex128)
    
    coords_6d = np.zeros(6, dtype=np.complex128)
    coords_6d[1] = cm_coords_4d[0]  # q2
    coords_6d[2] = cm_coords_4d[2]  # q3
    coords_6d[4] = cm_coords_4d[1]  # p2  
    coords_6d[5] = cm_coords_4d[3]  # p3
    
    print(f"Before inverse Lie: coords_6d[5] (p3) = {coords_6d[5]}")
    
    # Apply inverse Lie transforms step by step
    poly_G = point.get_cached_generating_functions(max_degree)
    
    for degree in range(max_degree, 2, -1):
        if degree < len(poly_G) and np.any(poly_G[degree]):
            print(f"\nApplying inverse G{degree}")
            print(f"  Before: coords_6d[5] = {coords_6d[5]}")
            
            coords_6d = _apply_single_inverse_generator(
                coords_6d, poly_G[degree], degree, psi, clmo, 
                _create_encode_dict_from_clmo(clmo)
            )
            
            print(f"  After: coords_6d[5] = {coords_6d[5]}")
    
    print(f"\nFinal coords after all inverse Lie: {coords_6d}")
    return coords_6d


def test_coordinate_polynomial_conversion(cr3bp_data_fixture):
    """Debug the coordinate ↔ polynomial conversion."""
    
    # Test with known coordinates
    test_coords = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.complex128)
    
    print(f"Original coords: {test_coords}")
    
    data = cr3bp_data_fixture
    point = data["point"]
    psi = data["psi"]
    clmo = List(data["clmo"])
    max_degree = data["max_degree"]

    # Convert to polynomial
    poly = polynomial_zero_list(max_degree, psi)
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    if len(poly) > 1:
        for i in range(6):
            if abs(test_coords[i]) > 1e-15:
                k = np.zeros(6, dtype=np.int64)
                k[i] = 1
                pos = encode_multiindex(k, 1, encode_dict_list)
                if 0 <= pos < poly[1].shape[0]:
                    poly[1][pos] = test_coords[i]
                    print(f"Set poly[1][{pos}] = coords[{i}] = {test_coords[i]}")
    
    # Convert back to coordinates
    recovered_coords = np.zeros(6, dtype=np.complex128)
    if len(poly) > 1:
        for i in range(6):
            k = np.zeros(6, dtype=np.int64)
            k[i] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            if 0 <= pos < poly[1].shape[0]:
                recovered_coords[i] = poly[1][pos]
                print(f"Got coords[{i}] = poly[1][{pos}] = {recovered_coords[i]}")
    
    print(f"Recovered coords: {recovered_coords}")
    print(f"Match: {np.allclose(test_coords, recovered_coords)}")

def test_track_p3_through_pipeline(cr3bp_data_fixture):
    """Track where the p3 coordinate goes through the entire pipeline."""
    
    # Start with test data
    poincare_point = np.array([0.01, 0.005])
    energy = 0.1
    
    data = cr3bp_data_fixture
    point = data["point"]
    psi = data["psi"]
    clmo = List(data["clmo"])
    max_degree = data["max_degree"]
    
    print("=== Tracking p3 through pipeline ===")
    
    # Step 1: Complete coordinates
    poly_cm_cn = point.get_cached_hamiltonian(max_degree, "center_manifold_cn")
    cm_coords_4d = _complete_cm_coordinates(poly_cm_cn, poincare_point, energy, clmo)
    print(f"1. After completion: p3 = {cm_coords_4d[3]}")
    
    # Step 2: Map to 6D
    coords_6d = np.zeros(6, dtype=np.complex128)
    coords_6d[1] = cm_coords_4d[0]  # q2
    coords_6d[2] = cm_coords_4d[2]  # q3
    coords_6d[4] = cm_coords_4d[1]  # p2
    coords_6d[5] = cm_coords_4d[3]  # p3
    print(f"2. After 6D mapping: coords_6d[5] (p3) = {coords_6d[5]}")
    
    # Step 3: Apply inverse Lie transforms
    poly_G = point.get_cached_generating_functions(max_degree)
    cn_coords = _apply_inverse_lie_transforms(cm_coords_4d, poly_G, psi, clmo, max_degree, 1e-15)
    print(f"3. After inverse Lie: cn_coords[5] (p3) = {cn_coords[5]}")
    
    # Step 4: CN → RN conversion
    rn_coords = _cn2rn_coordinates(cn_coords, max_degree, psi, clmo)
    print(f"4. After CN→RN: rn_coords[5] (pz_rn) = {rn_coords[5]}")
    
    # Step 5: RN → Physical conversion  
    phys_coords = _rn2phys_coordinates(rn_coords, point, max_degree, psi, clmo)
    print(f"5. After RN→Phys: phys_coords[5] (PZ) = {phys_coords[5]}")
    
    print("\n=== Summary ===")
    print(f"Started with p3 = {cm_coords_4d[3]}")
    print(f"Ended with PZ = {phys_coords[5]}")
    print(f"Ratio: {phys_coords[5] / cm_coords_4d[3] if cm_coords_4d[3] != 0 else 'inf'}")