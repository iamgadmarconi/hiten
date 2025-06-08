import numpy as np
import pytest

from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.lie import lie_transform
from algorithms.center.polynomial.base import init_index_tables
from algorithms.center.transforms import complexify, local2realmodal
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

