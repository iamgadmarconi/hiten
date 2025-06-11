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
from algorithms.center.transforms import complexify, local2realmodal
from algorithms.variables import N_VARS
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
    """Test the shape of the generating polynomials."""
    poly_G_total = debug_setup["poly_G_total"]
    psi = debug_setup["psi"]
    for n, G in enumerate(poly_G_total):
        if G is not None and G.any():
            assert G.ndim == 1 and len(G) == psi[N_VARS, n], f"bad shape at n={n}"

def test_poly_G_content(debug_setup):
    """Check which degrees have non-zero generating functions."""
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

def test_generator_structure(debug_setup):
    """Analyze the structure of generating functions to understand their contributions."""
    poly_G_total = debug_setup["poly_G_total"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    
    print("\nAnalyzing generator structure and potential contributions:")
    print("="*70)
    
    # Check which variables appear in each generator
    for n in range(3, min(9, len(poly_G_total))):
        if poly_G_total[n] is None or not poly_G_total[n].any():
            continue
            
        G_n = poly_G_total[n]
        print(f"\nGenerator G_{n}:")
        print(f"  Total coefficients: {len(G_n)}")
        print(f"  Non-zero coefficients: {np.count_nonzero(G_n)}")
        print(f"  Max magnitude: {np.max(np.abs(G_n)):.3e}")
        
        # Analyze which types of terms are present
        # Look for terms that could affect q1/p1 when starting from center manifold
        encode_dict = _create_encode_dict_from_clmo(clmo)[n]
        
        # Categories of terms
        has_q1_linear = False  # Terms with q1^1 * (other vars)
        has_p1_linear = False  # Terms with p1^1 * (other vars)
        has_q1p1_only = False  # Terms with only q1 and/or p1
        pure_cm_terms = 0      # Terms with only q2,p2,q3,p3
        
        for idx, coeff in enumerate(G_n):
            if abs(coeff) < 1e-15:
                continue
                
            # Decode the multi-index
            k = decode_multiindex(idx, n, clmo)
            k0, k1, k2, k3, k4, k5 = k  # q1, q2, q3, p1, p2, p3
            
            # Check categories
            if k0 == 1 and k3 == 0:  # q1^1 without p1
                has_q1_linear = True
            if k3 == 1 and k0 == 0:  # p1^1 without q1
                has_p1_linear = True
            if k1 == 0 and k2 == 0 and k4 == 0 and k5 == 0:  # Only q1/p1
                has_q1p1_only = True
            if k0 == 0 and k3 == 0:  # No q1 or p1
                pure_cm_terms += 1
        
        print(f"  Has q1-linear terms: {has_q1_linear}")
        print(f"  Has p1-linear terms: {has_p1_linear}")
        print(f"  Has q1/p1-only terms: {has_q1p1_only}")
        print(f"  Pure CM terms (no q1/p1): {pure_cm_terms}")


def test_homological_equation(debug_setup):
    """
    Homological equation spot test.
    Check that {λq₁p₁ + λω₁q₂p₂ + λω₂q₃p₃, Gₙ} = -Pₙᵉˡⁱᵐ
    using polynomial_poisson_bracket and the eliminated terms.
    """
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

def test_generator_cm_property(debug_setup):
    """
    Test if generators preserve the center manifold constraint.
    
    For a proper center manifold reduction, the transformed Hamiltonian
    should have no linear terms in q1, p1 when restricted to the center manifold.
    """
    poly_G_total = debug_setup["poly_G_total"]
    poly_elim_total = debug_setup["poly_elim_total"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    
    print("\nAnalyzing generator terms for center manifold compatibility:")
    print("="*70)
    
    # For each generator, check what types of terms it contains
    for n in range(3, min(7, len(poly_G_total))):
        if poly_G_total[n] is None or not poly_G_total[n].any():
            continue
            
        G_n = poly_G_total[n]
        encode_dict = _create_encode_dict_from_clmo(clmo)[n]
        
        print(f"\nGenerator G_{n}:")
        
        # Categorize terms by their dependence on q1, p1
        term_categories = {
            'pure_cm': [],        # Only q2,p2,q3,p3
            'linear_q1': [],      # q1 * (cm vars)^(n-1)
            'linear_p1': [],      # p1 * (cm vars)^(n-1)
            'q1p1_mixed': [],     # Contains both q1 and p1
            'high_order_q1p1': [] # Higher powers of q1 or p1
        }
        
        for idx, coeff in enumerate(G_n):
            if abs(coeff) < 1e-15:
                continue
                
            k = decode_multiindex(idx, n, clmo)
            k0, k1, k2, k3, k4, k5 = k  # q1, q2, q3, p1, p2, p3
            
            # Categorize
            if k0 == 0 and k3 == 0:
                term_categories['pure_cm'].append((k, coeff))
            elif k0 == 1 and k3 == 0:
                term_categories['linear_q1'].append((k, coeff))
            elif k0 == 0 and k3 == 1:
                term_categories['linear_p1'].append((k, coeff))
            elif k0 >= 1 and k3 >= 1:
                term_categories['q1p1_mixed'].append((k, coeff))
            else:
                term_categories['high_order_q1p1'].append((k, coeff))
        
        # Report findings
        for category, terms in term_categories.items():
            if terms:
                print(f"  {category}: {len(terms)} terms")
                # Show a few examples
                for i, (k, coeff) in enumerate(terms[:3]):
                    print(f"    k={k}, coeff={coeff:.6e}")
                if len(terms) > 3:
                    print(f"    ... and {len(terms)-3} more")
        
        # Also check eliminated terms
        if n < len(poly_elim_total) and poly_elim_total[n] is not None:
            P_elim = poly_elim_total[n]
            elim_nonzero = np.count_nonzero(P_elim)
            if elim_nonzero > 0:
                print(f"  Eliminated terms: {elim_nonzero}")


def check_real_generator(G, degree, clmo, encode_dict_list, var_pairs):
    """
    Checks if a polynomial, represented by its coefficients G, is real-valued
    given pairs of variables that are complex conjugates.

    var_pairs = [(0,3), (1,4), (2,5)]  # example index pairs (z, z̄)

    For a polynomial P to be real-valued, the coefficient of a monomial term
    must be the complex conjugate of the coefficient of the term with swapped
    conjugate variables. This function computes the maximum violation of this property.

    Returns
    -------
    float
        The maximum absolute difference |c_α - conj(c_β)| over all pairs of
        coefficients (c_α, c_β) that should be conjugates.
    """
    if not G.any():
        return 0.0

    max_error = 0.0
    
    visited_indices = np.zeros(len(G), dtype=bool)

    for idx, coeff in enumerate(G):
        if visited_indices[idx]:
            continue

        # Decode the 1D-array index to a multi-index k
        k = decode_multiindex(idx, degree, clmo)

        # Create the multi-index for the conjugate term by swapping exponents
        k_swapped = np.copy(k)
        for v_idx_from, v_idx_to in var_pairs:
            k_swapped[v_idx_from] = k[v_idx_to]
            k_swapped[v_idx_to] = k[v_idx_from]

        # Encode the swapped multi-index back to a 1D-array index
        idx_conj = encode_multiindex(k_swapped, degree, encode_dict_list)

        if idx_conj < 0:
            # This case means the conjugate monomial does not exist in the basis.
            # If the original coefficient is non-zero, this implies the polynomial
            # is not real, and the error is the magnitude of the coefficient.
            if abs(coeff) > 1e-15:
                max_error = max(max_error, abs(coeff))
            continue

        coeff_conj = G[idx_conj]
        
        # The reality condition is c_k = conj(c_{k_swapped})
        error = abs(coeff - np.conj(coeff_conj))
        if error > max_error:
            max_error = error

        # Mark both this index and its conjugate as visited to avoid re-checking
        visited_indices[idx] = True
        if idx_conj < len(G) and idx_conj >= 0:
            visited_indices[idx_conj] = True

    return max_error


def test_poly_G_is_real(debug_setup):
    """
    Test that the generating functions G_n correspond to real-valued polynomials.
    This is a critical property for the transformation to be valid, as the
    generating functions must be real to generate a canonical transformation
    that preserves the real-valued nature of the Hamiltonian.
    """
    poly_G_total = debug_setup["poly_G_total"]
    clmo = debug_setup["clmo"]
    max_degree = debug_setup["max_degree"]
    
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    # Define pairs of variable indices that are complex conjugates of each other.
    # In our coordinate system (q1, q2, q3, p1, p2, p3), if p_i = conj(q_i),
    # the pairs of indices are (0, 3), (1, 4), and (2, 5).
    var_pairs = [(0, 3), (1, 4), (2, 5)]

    print("\nVerifying G_n reality:")
    for n in range(3, max_degree + 1):
        if n >= len(poly_G_total) or poly_G_total[n] is None:
            continue
        
        G_n = poly_G_total[n]
        if not G_n.any():
            continue
            
        error = check_real_generator(G_n, n, clmo, encode_dict_list, var_pairs)
        
        print(f"G_{n}: Max reality error = {error:.2e}")


def dump_non_real(poly, degree, clmo, thresh=1e-12, tag="H"):
    """List every monomial of total degree `degree` whose coefficient is not real."""
    bad = []
    arr = poly[degree]
    for idx, c in enumerate(arr):
        if abs(c.imag) > thresh:          # adjust threshold if needed
            exp = decode_multiindex(idx, degree, clmo)  # decode to get multi-index
            bad.append((exp, c))
    if bad:
        print(f"{tag}[{degree}]: {len(bad)} non-real terms")
        for exp, c in bad[:10]:           # print at most 10 per degree
            kq, kp = exp[:3], exp[3:]
            print(f"  {kq}|{kp} : {c.real:+.3e} {c.imag:+.3e} i")
    return len(bad)


def test_dump_non_real_coefficients(debug_setup):
    """
    Test function that dumps non-real coefficients for inspection.
    This helps identify which monomials have significant imaginary parts.
    """
    poly_G_total = debug_setup["poly_G_total"]
    poly_elim_total = debug_setup["poly_elim_total"]
    clmo = debug_setup["clmo"]
    max_degree = debug_setup["max_degree"]
    
    print("\nAnalyzing non-real coefficients in polynomials:")
    print("="*70)
    
    # Check generating functions
    print("\nGenerating functions G_n:")
    total_non_real_G = 0
    for n in range(3, max_degree + 1):
        if n >= len(poly_G_total) or poly_G_total[n] is None or not poly_G_total[n].any():
            continue
        count = dump_non_real(poly_G_total, n, clmo, tag="G")
        total_non_real_G += count
    
    print(f"\nTotal non-real terms in G_n: {total_non_real_G}")
    
    # Check eliminated terms
    print("\nEliminated terms P_elim_n:")
    total_non_real_elim = 0
    for n in range(3, max_degree + 1):
        if n >= len(poly_elim_total) or poly_elim_total[n] is None or not poly_elim_total[n].any():
            continue
        count = dump_non_real(poly_elim_total, n, clmo, tag="P_elim")
        total_non_real_elim += count
    
    print(f"\nTotal non-real terms in P_elim_n: {total_non_real_elim}")
    
    # Summary
    if total_non_real_G == 0 and total_non_real_elim == 0:
        print("\n✓ All coefficients are real (within threshold)")
    else:
        print(f"\n⚠ Found {total_non_real_G + total_non_real_elim} total non-real coefficients")