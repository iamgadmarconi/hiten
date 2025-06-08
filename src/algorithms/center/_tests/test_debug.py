import numpy as np
import pytest

from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.lie import lie_transform
from algorithms.center.polynomial.base import init_index_tables, _create_encode_dict_from_clmo
from algorithms.center.transforms import complexify, local2realmodal
from algorithms.center.polynomial.operations import polynomial_zero_list, polynomial_poisson_bracket
from system.libration import L1Point

# Test parameters
MU_EM = 0.0121505816  # Earth-Moon mass parameter
TOL_TEST = 1e-15


@pytest.fixture(scope="module")
def debug_setup():
    """Set up test data for diagnostics."""
    point = L1Point(mu=MU_EM)
    _ = point.position  # Ensure position is calculated
    
    max_degree = 8
    psi, clmo = init_index_tables(max_degree)
    
    # Build and normalize Hamiltonian
    H_phys = build_physical_hamiltonian(point, max_degree)
    H_rn = local2realmodal(point, H_phys, max_degree, psi, clmo)
    H_cn = complexify(H_rn, max_degree, psi, clmo)
    
    # Perform Lie transformation
    poly_trans, poly_G_total = lie_transform(point, H_cn, psi, clmo, max_degree)
    
    return {
        "point": point,
        "psi": psi,
        "clmo": clmo,
        "max_degree": max_degree,
        "poly_G_total": poly_G_total,
        "complex_6d_cm": np.array([ 0.+0.j,-0.47286937-0.02290062j, 0.+0.21637672j, 0.+0.j,-0.02290062-0.47286937j,0.21637672+0.j], dtype=np.complex128)
    }

def test_pb():
    psi, clmo = init_index_tables(1)
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    q2 = polynomial_zero_list(1, psi)
    p2 = polynomial_zero_list(1, psi)
    q2[1][1] = 1.0          # q2  is the second coordinate
    p2[1][4] = 1.0          # p2  is the fifth coordinate

    res = polynomial_poisson_bracket(q2, p2, 1, psi, clmo, encode_dict_list)
    assert res[0] == 1                     # constant term
