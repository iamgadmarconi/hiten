import math
from typing import List

import numpy as np
import pytest
import sympy as sp
from numpy.linalg import norm

from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.lie import (_apply_lie_transform,
                                   _center2modal, 
                                   _get_homogeneous_terms,
                                   _select_terms_for_elimination,
                                   _solve_homological_equation,
                                   evaluate_transform,
                                   lie_transform)
from algorithms.center.manifold import center_manifold_real
from algorithms.center.polynomial.algebra import _poly_poisson
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               decode_multiindex,
                                               encode_multiindex,
                                               init_index_tables, make_poly)
from algorithms.center.polynomial.conversion import sympy2poly
from algorithms.center.polynomial.operations import (polynomial_differentiate,
                                                     polynomial_evaluate,
                                                     polynomial_poisson_bracket,
                                                     polynomial_variables_list,
                                                     polynomial_zero_list)
from algorithms.center.transforms import complexify, local2realmodal
from algorithms.variables import N_VARS
from system.libration import L1Point

MU_EM = 0.0121505816  # Earth-Moon mass parameter (example)
MAX_DEGREE_TEST = 6   
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

@pytest.fixture
def cn_hamiltonian_data(request):
    max_deg = request.param

    # psi table needs to be large enough for n_missing in _get_homogeneous_terms tests
    psi_init_deg = max_deg + 2
    psi, clmo = init_index_tables(psi_init_deg)
    encode_dict = _create_encode_dict_from_clmo(clmo)

    # Use a standard mu value (e.g., Earth-Moon L1)
    mu_earth_moon = 0.012150585609624
    point = L1Point(mu=mu_earth_moon)

    # The Hamiltonian itself is constructed up to max_deg.
    # The psi and clmo (initialized for psi_init_deg) are suitable as psi_init_deg >= max_deg.
    H_phys = build_physical_hamiltonian(point, max_deg)
    H_rn = local2realmodal(point, H_phys, max_deg, psi, clmo)
    H_coeffs = complexify(H_rn, max_deg, psi, clmo)

    return H_coeffs, psi, clmo, encode_dict, max_deg


@pytest.mark.parametrize("cn_hamiltonian_data", [2, 3, 4, 6], indirect=True)
def test_get_homogeneous_terms_when_n_is_within_H_coeffs(cn_hamiltonian_data):
    H_coeffs, psi, clmo, encode_dict, max_deg = cn_hamiltonian_data

    n = 3  # Test for degree 3 terms
    if n > max_deg:
        # This block handles cases where n=3 but max_deg < 3 (e.g., max_deg=2).
        # If H_coeffs does not have H_coeffs[n], _get_homogeneous_terms should return zeros.
        Hn = _get_homogeneous_terms(H_coeffs, n, psi)
        assert np.all(Hn == 0), f"vector for n={n} (n > max_deg={max_deg}) is not zero"
        assert len(Hn) == psi[6, n], f"wrong length for zero vector for n={n} (n > max_deg={max_deg})"
    else:  # n <= max_deg (e.g. max_deg = 3, 4, or 6)
        Hn = _get_homogeneous_terms(H_coeffs, n, psi)
        expected_Hn = H_coeffs[n]
        assert np.array_equal(Hn, expected_Hn), "returned vector is not H_n"
        # must be a *copy*, not the original reference
        if Hn.size > 0:
            original_coeff_val = H_coeffs[n][0] # Save original value from H_coeffs
            Hn[0] += 1.0 # Modify the supposed copy
            assert H_coeffs[n][0] == original_coeff_val, "Original H_coeffs was modified!"
            assert Hn[0] != original_coeff_val, "Copy was not modified or not a proper copy."
        elif expected_Hn.size == 0:
            # Both are empty, this is fine. No copy modification to test on Hn[0].
            pass
        # No 'else' here, as an empty Hn and non-empty expected_Hn would be caught by np.array_equal


@pytest.mark.parametrize("cn_hamiltonian_data", [2, 3, 4, 6], indirect=True)
def test_get_homogeneous_terms_when_n_is_beyond_H_coeffs_degree(cn_hamiltonian_data):
    H_coeffs, psi, clmo, encode_dict, max_deg = cn_hamiltonian_data

    # H_coeffs extends up to max_deg. We test for a degree n_missing > max_deg.
    # but still within psi_init_deg (max_deg + 2)
    n_missing = max_deg + 1

    Hn_zero = _get_homogeneous_terms(H_coeffs, n_missing, psi)
    assert np.all(Hn_zero == 0), "vector for missing degree is not zero"
    # The length of Hn_zero should correspond to psi[6, n_missing]
    assert len(Hn_zero) == psi[6, n_missing], "wrong length for zero vector"


@pytest.mark.parametrize("cn_hamiltonian_data", [2, 3, 4, 6], indirect=True)
def test_get_homogeneous_terms_when_n_is_at_psi_table_edge(cn_hamiltonian_data):
    H_coeffs, psi, clmo, encode_dict, max_deg = cn_hamiltonian_data

    # This case tests access at psi_init_deg = max_deg + 2.
    n_at_psi_edge = max_deg + 2

    Hn_zero_psi_edge = _get_homogeneous_terms(H_coeffs, n_at_psi_edge, psi)
    assert np.all(Hn_zero_psi_edge == 0), "vector for missing degree (psi edge) is not zero"
    assert len(Hn_zero_psi_edge) == psi[6, n_at_psi_edge], "wrong length for zero vector (psi edge)"


@pytest.mark.parametrize("n", [3, 4, 6])
def test_select_terms_for_elimination(n):
    max_deg = n                       # lookup tables big enough
    psi, clmo = init_index_tables(max_deg)

    size = psi[6, n]
    rng  = np.random.default_rng(0)

    # random complex coefficients in [-1,1] + i[-1,1]
    Hn_orig = (rng.uniform(-1, 1, size) + 1j*rng.uniform(-1, 1, size)).astype(np.complex128)
    # Create a copy for checking if the original input is mutated
    Hn_for_mutation_check = Hn_orig.copy()

    # ---------- routine under test ---------------------------------------
    # The function _select_terms_for_elimination is expected to return a new array
    # where terms with k[0]==k[3] ("good" terms) are zeroed out,
    # and terms with k[0]!=k[3] ("bad" terms, for elimination) are preserved.
    got = _select_terms_for_elimination(Hn_orig, n, clmo)

    # ---------- verification -------------------------------------------
    # Verify basic properties of the output array
    assert isinstance(got, np.ndarray), "Output should be a numpy array."
    assert got.shape == Hn_orig.shape, \
        f"Output shape {got.shape} does not match input shape {Hn_orig.shape}."
    assert got.dtype == Hn_orig.dtype, \
        f"Output dtype {got.dtype} does not match input dtype {Hn_orig.dtype}."

    # Verify each term's value in the output based on its multi-index property
    for pos in range(size):
        k = decode_multiindex(pos, n, clmo)
        original_value_at_pos = Hn_orig[pos]

        if k[0] == k[3]:  # "Good" term (q1_exponent == p1_exponent)
                          # These terms are not for elimination by Gn, so the function
                          # _select_terms_for_elimination (which selects terms *to be* eliminated)
                          # should output zero for them.
            assert got[pos] == 0j, \
                f"For n={n}, pos={pos} (k={k} where k[0]==k[3]), Hn_orig[{pos}]={original_value_at_pos}. " \
                f"Expected got[{pos}]=0j, but got {got[pos]}."
        else:  # "Bad" term (q1_exponent != p1_exponent)
               # These terms are for elimination by Gn, so the function
               # _select_terms_for_elimination should preserve/select them.
            assert got[pos] == original_value_at_pos, \
                f"For n={n}, pos={pos} (k={k} where k[0]!=k[3]), Hn_orig[{pos}]={original_value_at_pos}. " \
                f"Expected got[{pos}]={original_value_at_pos}, but got {got[pos]}."

    # Make sure the input Hn_orig was not mutated in-place
    assert np.array_equal(Hn_orig, Hn_for_mutation_check), \
        "Input Hn_orig was mutated by _select_terms_for_elimination. " \
        "The original Hn should remain unchanged as it might be used elsewhere."


@pytest.mark.parametrize("n", [2, 3, 4, 6, 9])
def test_homological_property(n):
    max_deg = n
    psi, clmo = init_index_tables(max_deg)
    encode_dict = _create_encode_dict_from_clmo(clmo)

    # pick arbitrary non-resonant frequencies
    lam, w1, w2 = 3.1, 2.4, 2.2
    eta = np.array([lam, 1j*w1, 1j*w2], dtype=np.complex128)

    # ---- fake degree-n polynomial with random 'bad' terms only ------------
    size = psi[6, n]
    rng  = np.random.default_rng(1234)
    Hn_bad = np.zeros(size, dtype=np.complex128)
    for pos in range(size):
        k = decode_multiindex(pos, n, clmo)
        if k[0] != k[3]:                     # k_q1 ≠ k_p1 → bad
            Hn_bad[pos] = rng.normal() + 1j*rng.normal()

    # ---- call the solver ---------------------------------------------------
    Gn = _solve_homological_equation(Hn_bad, n, eta, clmo)

    # ---- compute {H2,Gn} using Poisson bracket code -----------------------
    # Build H2 in coefficient-list format (degree 2)
    H2_list = polynomial_zero_list(max_deg, psi)
    idx = encode_multiindex((1,0,0,1,0,0), 2, encode_dict)   # q1 p1
    H2_list[2][idx] = lam
    idx = encode_multiindex((0,1,0,0,1,0), 2, encode_dict)   # q2 p2
    H2_list[2][idx] = 1j*w1
    idx = encode_multiindex((0,0,1,0,0,1), 2, encode_dict)   # q3 p3
    H2_list[2][idx] = 1j*w2

    # bracket restricted to degree n because both inputs are homogeneous
    # PB = poisson_bracket_degree2(H2[2], Gn, n, psi, clmo) # Old line
    
    # Use _poly_poisson for homogeneous inputs H2_list[2] (degree 2) and Gn (degree n)
    # Result is homogeneous of degree 2 + n - 2 = n
    PB_coeffs = _poly_poisson(H2_list[2], 2, Gn, n, psi, clmo, encode_dict)

    # ---- identity check ----------------------------------------------------
    # PB_coeffs must equal -Hn_bad *exactly* (same vector)
    assert np.allclose(PB_coeffs, -Hn_bad, atol=1e-14, rtol=1e-14)

    # bonus: Gn has zero on every "good" index
    for pos, g in enumerate(Gn):
        k = decode_multiindex(pos, n, clmo)
        if k[0] == k[3]:
            assert g == 0


# Define parameter sets for the test
test_params = [
    pytest.param("base_degG3_Nmax4_realH", 3, (2,0,0,0,1,0), 0.7, 1.3, 4, id="Base_degG3_Nmax4_realH"),
    pytest.param("high_degG5_Nmax8_realH", 5, (4,0,0,0,1,0), 0.7, 1.3, 8, id="High_degG5_Nmax8_realH"), # N_max=8 for {{H,G},G}
    pytest.param("Nmax6_degG4_realH", 4, (3,0,0,0,1,0), 0.7, 1.3, 6, id="Nmax6_degG4_realH_Term2_deg6"), # deg(H)=2, deg(G)=4 -> {{H,G},G} is deg 6
    pytest.param("complexH_degG3_Nmax4", 3, (2,0,0,0,1,0), 0.7, 1.3+0.5j, 4, id="ComplexH_degG3_Nmax4"),
    pytest.param("degG2_Nmax4_realH", 2, (1,0,0,0,1,0), 0.7, 1.3, 4, id="Low_degG2_Nmax4_realH_K_is_1"), # K = max(1, deg_G-1) = max(1,1)=1
]

@pytest.mark.parametrize(
    "test_name, G_deg_actual, G_exps, G_coeff_val, H_coeff_val, N_max_test",
    test_params
)
def test_apply_lie_transform(test_name, G_deg_actual, G_exps, G_coeff_val, H_coeff_val, N_max_test):
    psi, clmo = init_index_tables(N_max_test)
    encode_dict = _create_encode_dict_from_clmo(clmo)

    H_deg_actual = 2
    H_exps_tuple = (0,1,0,0,1,0)
    H_exps_np = np.array(H_exps_tuple, dtype=np.int64)
    H_coeffs_list = polynomial_zero_list(N_max_test, psi)
    idx_H = encode_multiindex(H_exps_np, H_deg_actual, encode_dict)
    if H_deg_actual <= N_max_test: # Ensure degree is within bounds of the list
        H_coeffs_list[H_deg_actual][idx_H] = H_coeff_val

    _ = polynomial_zero_list(N_max_test, psi) # G_n is just one component

    G_n_array = make_poly(G_deg_actual, psi)

    G_exps_np = np.array(G_exps, dtype=np.int64)
    idx_G = encode_multiindex(G_exps_np, G_deg_actual, encode_dict)
    G_n_array[idx_G] = G_coeff_val
    
    # Call the function under test
    H1_transformed_coeffs = _apply_lie_transform(H_coeffs_list, G_n_array, G_deg_actual, N_max_test, psi, clmo, encode_dict, tol=1e-15)

    # --- SymPy Reference Calculation ---
    q1,q2,q3,p1,p2,p3 = sp.symbols('q1 q2 q3 p1 p2 p3')
    coords = (q1,q2,q3,p1,p2,p3)

    # Construct Hsym
    Hsym = sp.sympify(H_coeff_val) 
    for i, exp_val in enumerate(H_exps_tuple):
        if exp_val > 0:
            Hsym *= coords[i]**exp_val

    # Construct Gsym
    Gsym = sp.sympify(G_coeff_val)
    for i, exp_val in enumerate(G_exps):
        if exp_val > 0:
            Gsym *= coords[i]**exp_val
    
    def sympy_poisson_bracket(f, g, variables_tuple):
        q_vars = variables_tuple[:len(variables_tuple)//2]
        p_vars = variables_tuple[len(variables_tuple)//2:]
        bracket = sp.S.Zero
        for i_pb in range(len(q_vars)): # Renamed loop var to avoid conflict
            bracket += (sp.diff(f, q_vars[i_pb]) * sp.diff(g, p_vars[i_pb]) -
                        sp.diff(f, p_vars[i_pb]) * sp.diff(g, q_vars[i_pb]))
        return sp.expand(bracket)

    K_series = max(1, G_deg_actual - 1)
    
    current_ad_term_sym = Hsym 
    Href_sym_calc = Hsym

    if K_series > 0 :
        for k_val in range(1, K_series + 1):
            current_ad_term_sym = sympy_poisson_bracket(current_ad_term_sym, Gsym, coords)
            Href_sym_calc += current_ad_term_sym / math.factorial(k_val)

    Href_poly = sympy2poly(Href_sym_calc, list(coords), psi, clmo, encode_dict)

    # --- Comparison ---
    length_error_msg = f"Test '{test_name}': Output H1_transformed_coeffs has unexpected length {len(H1_transformed_coeffs)}, expected {N_max_test + 1}"
    assert len(H1_transformed_coeffs) == N_max_test + 1, length_error_msg

    for d in range(N_max_test + 1):
        coeffs_from_lie_transform = H1_transformed_coeffs[d]
        
        if d < len(Href_poly):
            coeffs_from_sympy_ref = Href_poly[d]
        else:
            expected_size = psi[N_VARS, d] if d < psi.shape[1] else 0 
            if expected_size < 0: expected_size = 0 
            coeffs_from_sympy_ref = np.zeros(expected_size, dtype=np.complex128)

        if coeffs_from_lie_transform.ndim == 0 and coeffs_from_lie_transform.size == 1:
             coeffs_from_lie_transform = coeffs_from_lie_transform.reshape(1)
        if coeffs_from_sympy_ref.ndim == 0 and coeffs_from_sympy_ref.size == 1:
             coeffs_from_sympy_ref = coeffs_from_sympy_ref.reshape(1)
        
        mismatch_msg = (
            f"Test '{test_name}': Mismatch at degree {d}.\n"
            f"Computed (Lie): {coeffs_from_lie_transform}\n"
            f"Expected (SymPy): {coeffs_from_sympy_ref}\n"
            f"Sympy Href: {Href_sym_calc}"
        )
        assert np.allclose(coeffs_from_lie_transform, coeffs_from_sympy_ref, atol=1e-14, rtol=1e-14), \
            mismatch_msg


@pytest.mark.parametrize("cn_hamiltonian_data", [2, 3, 4, 6], indirect=True)
def test_lie_transform_removes_bad_terms(cn_hamiltonian_data):
    H_coeffs, psi, clmo, _, max_deg = cn_hamiltonian_data
    mu_earth_moon = 0.012150585609624
    point = L1Point(mu=mu_earth_moon)
    H_out, _ = lie_transform(point, H_coeffs, psi, clmo, max_deg)

    # Use a tolerance appropriate for accumulated floating-point errors
    tolerance = 1e-15
    
    for n in range(3, max_deg + 1):
        bad = _select_terms_for_elimination(H_out[n], n, clmo)
        max_bad_coeff = np.max(np.abs(bad)) if bad.size > 0 else 0.0
        assert max_bad_coeff < tolerance, (
            f"Bad monomials not sufficiently eliminated at degree {n}. "
            f"Max coefficient: {max_bad_coeff:.2e}, tolerance: {tolerance:.2e}. "
            f"Non-zero positions: {np.where(np.abs(bad) >= tolerance)}")

    assert np.allclose(H_out[2], H_coeffs[2], atol=0, rtol=0)


def test_center2modal_simple_symplectic_check(cr3bp_data_fixture):
    """
    Simple diagnostic test to understand the symplectic failure.
    """
    point = cr3bp_data_fixture["point"]
    psi = cr3bp_data_fixture["psi"]
    clmo = cr3bp_data_fixture["clmo"]
    max_deg = 3  # Use low degree for debugging
    
    # Get generating functions
    poly_G_total = point.cache_get(('generating_functions', max_deg))
    if poly_G_total is None:
        psi, clmo = init_index_tables(max_deg)
        _ = center_manifold_real(point, psi, clmo, max_deg)
        poly_G_total = point.cache_get(('generating_functions', max_deg))
    
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    # Get expansions
    expansions = _center2modal(poly_G_total, max_deg, psi, clmo, inverse=True)
    
    # Test a simple case: {q0, p0} should be 1
    q0_poly = expansions[0]  # q0 expansion
    p0_poly = expansions[3]  # p0 expansion
    
    bracket_result = polynomial_poisson_bracket(
        q0_poly, p0_poly, max_deg, psi, clmo, encode_dict_list
    )
    
    # This should be the constant polynomial 1
    print(f"Q0-P0 bracket degree 0 coeffs: {bracket_result[0] if len(bracket_result) > 0 else 'empty'}")
    if len(bracket_result) > 1:
        print(f"Q0-P0 bracket degree 1 coeffs: {bracket_result[1]}")
    if len(bracket_result) > 2:
        print(f"Q0-P0 bracket degree 2 coeffs: {bracket_result[2]}")
    if len(bracket_result) > 3:
        print(f"Q0-P0 bracket degree 3 coeffs: {bracket_result[3]}")
    
    # Test a simple case: {q0, q1} should be 0
    q1_poly = expansions[1]  # q1 expansion
    
    bracket_result_qq = polynomial_poisson_bracket(
        q0_poly, q1_poly, max_deg, psi, clmo, encode_dict_list
    )
    
    print(f"Q0-Q1 bracket degree 0 coeffs: {bracket_result_qq[0] if len(bracket_result_qq) > 0 else 'empty'}")
    if len(bracket_result_qq) > 1:
        print(f"Q0-Q1 bracket degree 1 coeffs: {bracket_result_qq[1]}")
    if len(bracket_result_qq) > 2:
        print(f"Q0-Q1 bracket degree 2 coeffs: {bracket_result_qq[2]}")
    if len(bracket_result_qq) > 3:
        print(f"Q0-Q1 bracket degree 3 coeffs: {bracket_result_qq[3]}")
    
    # Let's also check what the q0 and q1 expansions look like
    print("Q0 expansion:")
    for deg in range(len(q0_poly)):
        if q0_poly[deg].size > 0 and np.any(q0_poly[deg] != 0):
            print(f"  Degree {deg}: {q0_poly[deg]}")
    
    print("Q1 expansion:")
    for deg in range(len(q1_poly)):
        if q1_poly[deg].size > 0 and np.any(q1_poly[deg] != 0):
            print(f"  Degree {deg}: {q1_poly[deg]}")


@pytest.mark.parametrize("max_deg", [3, 4, 5])
def test_center2modal_symplectic_property(cr3bp_data_fixture, max_deg):
    """
    Test that the _center2modal transformation is symplectic by verifying
    that it preserves canonical Poisson bracket relations.
    
    A transformation is symplectic if the transformed coordinates satisfy:
    {Q_i, Q_j} = 0, {P_i, P_j} = 0, {Q_i, P_j} = δ_ij
    where {,} denotes the Poisson bracket and δ_ij is the Kronecker delta.
    """
    point = cr3bp_data_fixture["point"]
    psi = cr3bp_data_fixture["psi"]
    clmo = cr3bp_data_fixture["clmo"]
    
    # Initialize tables for this test's max_deg if different from fixture
    if max_deg != cr3bp_data_fixture["max_degree"]:
        psi, clmo = init_index_tables(max_deg)
        _ = center_manifold_real(point, psi, clmo, max_deg)
    
    # Get the generating functions from the point's cache
    poly_G_total = point.cache_get(('generating_functions', max_deg))
    if poly_G_total is None:
        pytest.fail("Generating functions not found in cache")
    
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    # Apply the inverse Lie transformation to get coordinate expansions
    # These represent the transformation from center manifold coords to modal coords
    expansions = _center2modal(poly_G_total, max_deg, psi, clmo, inverse=True)
    
    # Create zero and identity polynomials for comparison
    zero_poly = polynomial_zero_list(max_deg, psi)
    identity_poly = polynomial_zero_list(max_deg, psi)
    if len(identity_poly) > 0 and identity_poly[0].size > 0:
        identity_poly[0][0] = 1.0  # Constant polynomial equal to 1
    
    tolerance = 1e-10
    
    # Test canonical Poisson bracket relations
    for i in range(6):
        for j in range(6):
            # Compute Poisson bracket of transformed coordinates
            bracket_result = polynomial_poisson_bracket(
                expansions[i], expansions[j], max_deg, psi, clmo, encode_dict_list
            )
            
            if i < 3 and j < 3:
                # {Q_i, Q_j} should be 0 (position-position brackets)
                _assert_polynomial_close_to_zero(
                    bracket_result, tolerance,
                    f"Position-position bracket {{Q{i}, Q{j}}} should be zero"
                )
            elif i >= 3 and j >= 3:
                # {P_i, P_j} should be 0 (momentum-momentum brackets)
                _assert_polynomial_close_to_zero(
                    bracket_result, tolerance,
                    f"Momentum-momentum bracket {{P{i-3}, P{j-3}}} should be zero"
                )
            elif i < 3 and j >= 3:
                # {Q_i, P_j} should be δ_ij (mixed brackets)
                if i == j - 3:
                    # Should be 1 (Kronecker delta = 1)
                    _assert_polynomial_close_to_constant(
                        bracket_result, identity_poly, tolerance,
                        f"Mixed bracket {{Q{i}, P{j-3}}} should be 1"
                    )
                else:
                    # Should be 0 (Kronecker delta = 0)
                    _assert_polynomial_close_to_zero(
                        bracket_result, tolerance,
                        f"Mixed bracket {{Q{i}, P{j-3}}} should be zero"
                    )
            elif i >= 3 and j < 3:
                # {P_i, Q_j} should be -δ_ij (antisymmetry of Poisson bracket)
                if i - 3 == j:
                    # Should be -1
                    minus_identity_poly = polynomial_zero_list(max_deg, psi)
                    if len(minus_identity_poly) > 0 and minus_identity_poly[0].size > 0:
                        minus_identity_poly[0][0] = -1.0
                    _assert_polynomial_close_to_constant(
                        bracket_result, minus_identity_poly, tolerance,
                        f"Mixed bracket {{P{i-3}, Q{j}}} should be -1"
                    )
                else:
                    # Should be 0
                    _assert_polynomial_close_to_zero(
                        bracket_result, tolerance,
                        f"Mixed bracket {{P{i-3}, Q{j}}} should be zero"
                    )


def test_center2modal_jacobian_symplectic_property(cr3bp_data_fixture):
    """
    Test that the Jacobian of the _center2modal transformation is symplectic
    by evaluating it at specific points and checking the matrix condition M^T J M = J.
    """
    point = cr3bp_data_fixture["point"]
    psi = cr3bp_data_fixture["psi"]
    clmo = cr3bp_data_fixture["clmo"]
    max_deg = cr3bp_data_fixture["max_degree"]
    
    # Get generating functions
    poly_G_total = point.cache_get(('generating_functions', max_deg))
    if poly_G_total is None:
        pytest.fail("Generating functions not found in cache")
    
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    # Get coordinate expansions
    expansions = _center2modal(poly_G_total, max_deg, psi, clmo, inverse=True)
    
    # Test at several points in the center manifold
    np.random.seed(RANDOM_SEED)
    test_points = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.complex128),  # Origin
        np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.complex128),  # Small displacement in q1
        np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.complex128),  # Small displacement in q2
        np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0], dtype=np.complex128),  # Small displacement in q3
    ]
    
    # Add some random test points with small magnitudes
    for _ in range(3):
        random_point = 0.05 * (np.random.randn(6) + 1j * np.random.randn(6))
        test_points.append(random_point.astype(np.complex128))
    
    for k, test_point in enumerate(test_points):
        # Compute Jacobian matrix numerically
        jacobian_matrix = _compute_transformation_jacobian(
            expansions, test_point, psi, clmo, encode_dict_list
        )
        
        # Check if Jacobian is symplectic
        is_symplectic_result = _is_complex_symplectic(jacobian_matrix)
        
        assert is_symplectic_result, (
            f"Jacobian matrix at test point {k} is not symplectic. "
            f"Point: {test_point}, Max eigenvalue of M^T J M - J: "
            f"{_symplectic_error(jacobian_matrix):.2e}"
        )


def _assert_polynomial_close_to_zero(poly_p: List[np.ndarray], tol: float, msg: str):
    """Assert that all coefficients of a polynomial are close to zero."""
    for deg in range(len(poly_p)):
        if poly_p[deg].size > 0:
            max_coeff = np.max(np.abs(poly_p[deg]))
            assert max_coeff < tol, f"{msg}. Max coefficient at degree {deg}: {max_coeff:.2e}"


def _assert_polynomial_close_to_constant(
    poly_p: List[np.ndarray], 
    expected_poly: List[np.ndarray], 
    tol: float, 
    msg: str
):
    """Assert that a polynomial is close to an expected constant polynomial."""
    assert len(poly_p) == len(expected_poly), f"{msg}. Length mismatch."
    
    for deg in range(len(poly_p)):
        if poly_p[deg].size > 0 or expected_poly[deg].size > 0:
            if poly_p[deg].shape != expected_poly[deg].shape:
                # Handle shape mismatch by reshaping if sizes match
                if poly_p[deg].size == expected_poly[deg].size:
                    poly_p_reshaped = poly_p[deg].reshape(expected_poly[deg].shape)
                    diff = np.abs(poly_p_reshaped - expected_poly[deg])
                else:
                    assert False, f"{msg}. Shape/size mismatch at degree {deg}"
            else:
                diff = np.abs(poly_p[deg] - expected_poly[deg])
            
            max_diff = np.max(diff)
            assert max_diff < tol, f"{msg}. Max difference at degree {deg}: {max_diff:.2e}"


def _compute_transformation_jacobian(
    expansions: List[List[np.ndarray]], 
    point: np.ndarray, 
    psi: np.ndarray, 
    clmo: List[np.ndarray],
    encode_dict_list: List[dict],
    h: float = 1e-8
) -> np.ndarray:
    """
    Compute the Jacobian matrix of the transformation numerically using finite differences.
    
    Parameters
    ----------
    expansions : List[List[np.ndarray]]
        Six polynomial expansions from _center2modal
    point : np.ndarray
        Point at which to evaluate the Jacobian
    psi, clmo, encode_dict_list : arrays
        Polynomial indexing structures
    h : float
        Step size for finite differences
        
    Returns
    -------
    np.ndarray
        6x6 complex Jacobian matrix
    """
    jacobian = np.zeros((6, 6), dtype=np.complex128)
    
    # Evaluate transformation at the base point
    base_result = evaluate_transform(expansions, point, clmo)
    
    # Compute partial derivatives using finite differences
    for j in range(6):
        # Create perturbed point
        point_plus = point.copy()
        point_plus[j] += h
        
        # Evaluate transformation at perturbed point
        perturbed_result = evaluate_transform(expansions, point_plus, clmo)
        
        # Compute finite difference approximation
        jacobian[:, j] = (perturbed_result - base_result) / h
    
    return jacobian


def _is_complex_symplectic(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a complex 6x6 matrix is symplectic by verifying M^T J M = J.
    
    Parameters
    ----------
    matrix : np.ndarray
        6x6 complex matrix to test
    tol : float
        Tolerance for the test
        
    Returns
    -------
    bool
        True if the matrix is symplectic within the given tolerance
    """
    # Standard symplectic matrix J
    J = np.zeros((6, 6), dtype=np.complex128)
    n = 3  # 3 degrees of freedom
    for i in range(n):
        J[i, i+n] = 1.0 + 0j
        J[i+n, i] = -1.0 + 0j
    
    # Calculate M^T J M
    M_T_J_M = matrix.conj().T @ J @ matrix
    
    # Check if M^T J M = J
    return np.allclose(M_T_J_M, J, atol=tol)


def _symplectic_error(matrix: np.ndarray) -> float:
    """
    Compute the symplectic error ||M^T J M - J||_∞ for a matrix M.
    
    Parameters
    ----------
    matrix : np.ndarray
        6x6 matrix to test
        
    Returns
    -------
    float
        Maximum absolute value of the elements of M^T J M - J
    """
    # Standard symplectic matrix J
    J = np.zeros((6, 6), dtype=np.complex128)
    n = 3  # 3 degrees of freedom
    for i in range(n):
        J[i, i+n] = 1.0 + 0j
        J[i+n, i] = -1.0 + 0j
    
    # Calculate M^T J M - J
    M_T_J_M = matrix.conj().T @ J @ matrix
    error_matrix = M_T_J_M - J
    
    return np.max(np.abs(error_matrix))


