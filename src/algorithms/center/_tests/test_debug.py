import numpy as np
import pytest

from algorithms.center.coordinates import (solve_real,
                                           solve_complex)
from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.lie import lie_transform
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               decode_multiindex,
                                               init_index_tables)
from algorithms.center.polynomial.operations import polynomial_variable
from algorithms.center.transforms import substitute_complex, _local2realmodal, substitute_real
from system.libration import L1Point

MAX_DEG = 1  # We only need degree-1 polynomials for linear-consistency checks
PSI, CLMO = init_index_tables(MAX_DEG)
ENCODE_DICT_LIST = _create_encode_dict_from_clmo(CLMO)
VARIABLE_NAMES = ["q1", "q2", "q3", "p1", "p2", "p3"]


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
        "H_cn": H_cn,  # Add the complex normal form Hamiltonian
        "poly_G_total": poly_G_total,
        "poly_elim_total": poly_elim_total,
        "complex_6d_cm": np.array([ 0.+0.j,-0.47286937-0.02290062j, 0.+0.21637672j, 0.+0.j,-0.02290062-0.47286937j,0.21637672+0.j], dtype=np.complex128)
    }

def test_coordinate_transform_consistency():
    """substitute_real/substitute_complex (polynomial) must match coordinate versions for each variable."""
    for var_idx, var_name in enumerate(VARIABLE_NAMES):
        # 1. Build degree-1 polynomial for the single variable
        poly_var = polynomial_variable(var_idx, MAX_DEG, PSI, CLMO, ENCODE_DICT_LIST)

        # 2. Build the corresponding coordinate vector
        coord_var = np.zeros(6, dtype=np.complex128)
        coord_var[var_idx] = 1.0

        print(f"\n Variable {var_name} (index {var_idx})")
        print("-" * 40)
        print(f"Original polynomial coeffs (degree1): {poly_var[1]}")
        print(f"Coordinate vector: {coord_var}")

        poly_realified = substitute_real(poly_var, MAX_DEG, PSI, CLMO)
        coord_realified = solve_complex(coord_var)
        print("\n  REALIFY comparison:")
        print(f"   Polynomial result: {poly_realified[1]}")
        print(f"   Coordinate result: {coord_realified}")
        assert np.allclose(
            poly_realified[1], coord_realified, atol=1e-14
        ), f"substitute_real mismatch for variable {var_name}"

        poly_complexified = substitute_complex(poly_var, MAX_DEG, PSI, CLMO)
        coord_complexified = solve_real(coord_var)
        print("\n  COMPLEXIFY comparison:")
        print(f"   Polynomial result: {poly_complexified[1]}")
        print(f"   Coordinate result: {coord_complexified}")
        assert np.allclose(
            poly_complexified[1], coord_complexified, atol=1e-14
        ), f"substitute_complex mismatch for variable {var_name}"
        print("-" * 40)

def test_inspect_hamiltonian_before_lie_transform(debug_setup):
    """Inspect and display the Hamiltonian coefficients before Lie transformation (poly_cn)."""
    H_cn = debug_setup["H_cn"]
    clmo = debug_setup["clmo"]
    
    print("\nInspecting Hamiltonian Before Lie Transform (Complex Normal Form):")
    print("="*70)
    
    # Display degrees 2, 3, and 4 as requested
    for degree in [2, 3, 4]:
        if degree >= len(H_cn) or H_cn[degree] is None:
            print(f"\nH{degree}: Not available")
            continue
            
        H_n = H_cn[degree]
        if H_n.size == 0 or not H_n.any():
            print(f"\nH{degree}: All coefficients are zero")
            continue
        
        print(f"\nH{degree} monomial structure:")
        print("-" * 50)
        
        # Count non-zero terms
        nonzero_count = 0
        
        # Go through all coefficients
        for idx, coeff in enumerate(H_n):
            if abs(coeff) < 1e-15:  # Skip essentially zero coefficients
                continue
                
            nonzero_count += 1
            
            # Decode the multi-index
            k = decode_multiindex(idx, degree, clmo)
            q1, q2, q3, p1, p2, p3 = k
            
            # Format coefficient similar to the image
            if abs(coeff.imag) < 1e-15:
                # Real coefficient
                coeff_str = f"{coeff.real:.5f}"
            elif abs(coeff.real) < 1e-15:
                # Pure imaginary coefficient
                if coeff.imag > 0:
                    coeff_str = f"{coeff.imag:.5f} i"
                else:
                    coeff_str = f"- {abs(coeff.imag):.5f} i"
            else:
                # Complex coefficient
                if coeff.imag >= 0:
                    coeff_str = f"{coeff.real:.5f} + {coeff.imag:.5f} i"
                else:
                    coeff_str = f"{coeff.real:.5f} - {abs(coeff.imag):.5f} i"
            
            # Display in format similar to image: hh[2][q1,q2,q3,p1,p2,p3] = coefficient
            print(f"hh[{degree}][{q1},{q2},{q3},{p1},{p2},{p3}] = {coeff_str}")
        
        print(f"\nTotal non-zero terms in H{degree}: {nonzero_count}")
    
    print("\n" + "="*70)

def test_inspect_generator_monomials(debug_setup):
    """Inspect and display the monomial structure of generators G3 and G4."""
    poly_G_total = debug_setup["poly_G_total"]
    clmo = debug_setup["clmo"]
    
    print("\nInspecting Generator Monomial Structure:")
    print("="*70)
    
    # Inspect G3 and G4 specifically
    for degree in [3, 4]:
        if degree >= len(poly_G_total) or poly_G_total[degree] is None:
            print(f"\nG{degree}: Not available")
            continue
            
        G_n = poly_G_total[degree]
        if not G_n.any():
            print(f"\nG{degree}: All coefficients are zero")
            continue
        
        print(f"\nG{degree} monomial structure:")
        print("-" * 50)
        
        # Count non-zero terms
        nonzero_count = 0
        
        # Go through all coefficients
        for idx, coeff in enumerate(G_n):
            if abs(coeff) < 1e-15:  # Skip essentially zero coefficients
                continue
                
            nonzero_count += 1
            
            # Decode the multi-index
            k = decode_multiindex(idx, degree, clmo)
            q1, q2, q3, p1, p2, p3 = k
            
            # Format coefficient similar to the image
            if abs(coeff.imag) < 1e-15:
                # Real coefficient
                coeff_str = f"{coeff.real:.6f}"
            elif abs(coeff.real) < 1e-15:
                # Pure imaginary coefficient
                coeff_str = f"{coeff.imag:.6f} i"
            else:
                # Complex coefficient
                coeff_str = f"{coeff.real:.6f} {coeff.imag:+.6f} i"
            
            # Display in format similar to image: g[3][q1,q2,q3,p1,p2,p3] = coefficient
            print(f"g[{degree}][{q1},{q2},{q3},{p1},{p2},{p3}] = {coeff_str}")
        
        print(f"\nTotal non-zero terms in G{degree}: {nonzero_count}")
    
    print("\n" + "="*70)
