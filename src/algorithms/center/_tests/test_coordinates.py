import numpy as np
import pytest

from system.libration import L1Point 
from algorithms.center.coordinates import (
    _cn2rn_coordinates,
    _rn2phys_coordinates,
    _complete_cm_coordinates,
    _cm2phys_coordinates,
)
from algorithms.center.lie import _apply_inverse_lie_transforms
from algorithms.center.polynomial.base import (
    init_index_tables, 
    _create_encode_dict_from_clmo,
    encode_multiindex
)
from algorithms.center.polynomial.operations import polynomial_zero_list, polynomial_evaluate
from algorithms.center.transforms import cn2rn, rn2cn, phys2rn, rn2phys

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
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
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
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
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
    clmo = data["clmo"]
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
    clmo = data["clmo"]
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
    clmo = data["clmo"]
    max_degree = data["max_degree"]
    
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    for coord_idx in range(6):
        poly_coord = polynomial_zero_list(max_degree, psi)
        if max_degree >= 1:
            k = np.zeros(6, dtype=np.int64)
            k[coord_idx] = 1
            pos = encode_multiindex(k, 1, encode_dict_list)
            # Initialize with complex 1.0 for rn2cn if poly_coord represents CN initially
            # or float 1.0 if poly_coord represents RN initially.
            # Assuming poly_coord represents RN here as per test name rn2cn(poly_coord...)
            poly_coord[1][pos] = 1.0 
        
        # Transform RN poly → CN poly → RN poly
        # Here, poly_coord is an RN polynomial by construction of its coefficient.
        poly_cn = rn2cn(poly_coord, max_degree, psi, clmo) # rn_poly to cn_poly
        poly_rn_back = cn2rn(poly_cn, max_degree, psi, clmo) # cn_poly to rn_poly
        
        # Check up to degree 1, as higher degrees might have more complex interactions
        # not necessarily cancelling out if the input is just a single coord term.
        for deg in range(min(2, max_degree + 1)): 
            np.testing.assert_allclose(
                poly_coord[deg], poly_rn_back[deg], 
                rtol=1e-12, atol=1e-15,
                err_msg=f"Polynomial RN→CN→RN round trip failed for coord {coord_idx}, degree {deg}"
            )
