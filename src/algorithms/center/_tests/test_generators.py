import numpy as np
import pytest

from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.lie import (_apply_series, _center2modal,
                                   evaluate_transform, lie_transform)
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               decode_multiindex,
                                               encode_multiindex,
                                               init_index_tables)
from algorithms.center.polynomial.operations import (
    polynomial_evaluate, polynomial_poisson_bracket, polynomial_zero_list)
from algorithms.center.transforms import substitute_complex, _local2realmodal
from algorithms.variables import N_VARS
from system.libration import L1Point, L2Point

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
    H_rn = _local2realmodal(point, H_phys, max_degree, psi, clmo)
    H_cn = substitute_complex(H_rn, max_degree, psi, clmo)
    
    # Perform Lie transformation
    poly_trans, poly_G_total, poly_elim_total = lie_transform(point, H_cn, psi, clmo, max_degree)
    
    return {
        "point": point,
        "psi": psi,
        "clmo": clmo,
        "max_degree": max_degree,
        "poly_G_total": poly_G_total,
        "poly_elim_total": poly_elim_total,
        "complex_6d_cm": np.array([ 0.+0.j,-0.47286937-0.02290062j, 0.+0.21637672j, 0.+0.j,-0.02290062-0.47286937j,0.21637672+0.j], dtype=np.complex128)
    }

def test_poly_G_total_shape(debug_setup):
    poly_G_total = debug_setup["poly_G_total"]
    psi = debug_setup["psi"]
    for n, G in enumerate(poly_G_total):
        if G is not None and G.any():
            assert G.ndim == 1 and len(G) == psi[N_VARS, n], f"bad shape at n={n}"

def test_poly_G_content(debug_setup):
    poly_G_total = debug_setup["poly_G_total"]
    max_degree = debug_setup["max_degree"]
    
    print("\nAnalyzing poly_G_total content:")
    print(f"Length of poly_G_total: {len(poly_G_total)}")
    print("-" * 50)
    
    for n in range(len(poly_G_total)):
        if poly_G_total[n] is None:
            print(f"Degree {n}: None")
        elif not poly_G_total[n].any():
            print(f"Degree {n}: All zeros (shape: {poly_G_total[n].shape})")
        else:
            num_nonzero = np.count_nonzero(poly_G_total[n])
            max_val = np.max(np.abs(poly_G_total[n]))
            print(f"Degree {n}: {num_nonzero} non-zero coefficients, max magnitude: {max_val:.2e}")
    
    # Also check what happens in _center2modal loop
    print("\n\nSimulating _center2modal loop behavior:")
    for max_deg in [4, 5, 6, 7, 8]:
        print(f"\nmax_degree = {max_deg}:")
        applied_degrees = []
        
        for n in range(3, max_deg + 1):
            if n >= len(poly_G_total):
                print(f"  Degree {n}: Skipped (n >= len(poly_G_total))")
            elif poly_G_total[n] is None:
                print(f"  Degree {n}: Skipped (None)")
            elif not poly_G_total[n].any():
                print(f"  Degree {n}: Skipped (all zeros)")
            else:
                print(f"  Degree {n}: Would be applied")
                applied_degrees.append(n)
        
        print(f"  Total degrees applied: {applied_degrees}")

def test_homological_equation(debug_setup):
    point = debug_setup["point"]
    poly_G_total = debug_setup["poly_G_total"]
    poly_elim_total = debug_setup["poly_elim_total"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    max_degree = debug_setup["max_degree"]
    
    # Get eigenvalues from the point
    linear_data = point.linear_data
    lam = linear_data.lambda1
    omega1 = linear_data.omega1
    omega2 = linear_data.omega2
    
    # Create encode dict for Poisson bracket computation
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    # Construct linear Hamiltonian: λq₁p₁ + iω₁q₂p₂ + iω₂q₃p₃
    # Variables: q₁=x₀, p₁=x₃, q₂=x₁, p₂=x₄, q₃=x₂, p₃=x₅
    linear_H = polynomial_zero_list(2, psi)  # degree 2 for quadratic terms
    
    # λq₁p₁ term: coefficient λ for x₀*x₃ monomial
    q1p1_multiindex = np.zeros(N_VARS, dtype=np.int64)
    q1p1_multiindex[0] = 1  # q₁
    q1p1_multiindex[3] = 1  # p₁
    q1p1_idx = encode_multiindex(q1p1_multiindex, 2, encode_dict_list)
    if q1p1_idx >= 0:
        linear_H[2][q1p1_idx] = lam
    
    # iω₁q₂p₂ term: coefficient iω₁ for x₁*x₄ monomial
    q2p2_multiindex = np.zeros(N_VARS, dtype=np.int64)
    q2p2_multiindex[1] = 1  # q₂
    q2p2_multiindex[4] = 1  # p₂
    q2p2_idx = encode_multiindex(q2p2_multiindex, 2, encode_dict_list)
    if q2p2_idx >= 0:
        linear_H[2][q2p2_idx] = 1j * omega1
    
    # iω₂q₃p₃ term: coefficient iω₂ for x₂*x₅ monomial
    q3p3_multiindex = np.zeros(N_VARS, dtype=np.int64)
    q3p3_multiindex[2] = 1  # q₃
    q3p3_multiindex[5] = 1  # p₃
    q3p3_idx = encode_multiindex(q3p3_multiindex, 2, encode_dict_list)
    if q3p3_idx >= 0:
        linear_H[2][q3p3_idx] = 1j * omega2
    
    # Test the homological equation for degrees 3-5 (where we have non-trivial G_n)
    tolerance = 1e-12  # Tolerance for numerical comparison
    
    for n in range(3, min(6, max_degree + 1)):
        if (n >= len(poly_G_total) or poly_G_total[n] is None or 
            n >= len(poly_elim_total) or poly_elim_total[n] is None):
            continue
            
        G_n = poly_G_total[n]
        P_elim_n = poly_elim_total[n]
        
        if not G_n.any() or not P_elim_n.any():  # Skip if all zeros
            continue
            
        # Create polynomial representation for G_n
        G_n_poly = polynomial_zero_list(n, psi)
        G_n_poly[n][:] = G_n[:]
        
        # Compute Poisson bracket {linear_H, G_n}
        pb_result = polynomial_poisson_bracket(linear_H, G_n_poly, max(2, n), psi, clmo, encode_dict_list)
        
        # Extract the degree n part of the Poisson bracket result
        if n < len(pb_result):
            pb_degree_n = pb_result[n]
            
            # Check the homological equation: {H₂, Gₙ} = -Pₙᵉˡⁱᵐ
            # This means: pb_degree_n + P_elim_n should be approximately zero
            difference = pb_degree_n + P_elim_n
            max_error = np.max(np.abs(difference))
            
            assert max_error < tolerance, (
                f"Homological equation violated at degree {n}: "
                f"max error = {max_error:.2e}, tolerance = {tolerance:.2e}"
            )
            
            # Also check that we have non-trivial terms (not just testing zeros)
            pb_magnitude = np.max(np.abs(pb_degree_n))
            elim_magnitude = np.max(np.abs(P_elim_n))
            
            assert pb_magnitude > tolerance and elim_magnitude > tolerance, (
                f"Trivial test at degree {n}: pb_magnitude={pb_magnitude:.2e}, "
                f"elim_magnitude={elim_magnitude:.2e}"
            )
