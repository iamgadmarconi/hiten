import math

import numpy as np
import pytest
import sympy as sp

from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.lie import (_apply_lie_transform,
                                   _get_homogeneous_terms,
                                   _select_terms_for_elimination,
                                   _solve_homological_equation, lie_transform)
from algorithms.center.polynomial.algebra import _poly_poisson
from algorithms.center.polynomial.base import (decode_multiindex,
                                               encode_multiindex,
                                               init_index_tables)
from algorithms.center.polynomial.conversion import sympy2poly
from algorithms.center.polynomial.operations import polynomial_zero_list
from algorithms.center.transforms import phys2rn, rn2cn
from algorithms.variables import N_VARS
from system.libration import L1Point


@pytest.fixture
def cn_hamiltonian_data(request):
    max_deg = request.param

    # psi table needs to be large enough for n_missing in _get_homogeneous_terms tests
    psi_init_deg = max_deg + 2
    psi, clmo = init_index_tables(psi_init_deg)

    # Use a standard mu value (e.g., Earth-Moon L1)
    mu_earth_moon = 0.012150585609624
    point = L1Point(mu=mu_earth_moon)

    # The Hamiltonian itself is constructed up to max_deg.
    # The psi and clmo (initialized for psi_init_deg) are suitable as psi_init_deg >= max_deg.
    H_phys = build_physical_hamiltonian(point, max_deg)
    H_rn = phys2rn(point, H_phys, max_deg, psi, clmo)
    H_coeffs = rn2cn(H_rn, max_deg, psi, clmo)

    return H_coeffs, psi, clmo, max_deg


@pytest.mark.parametrize("cn_hamiltonian_data", [2, 3, 4, 6], indirect=True)
def test_get_homogeneous_terms_when_n_is_within_H_coeffs(cn_hamiltonian_data):
    H_coeffs, psi, clmo, max_deg = cn_hamiltonian_data

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
    H_coeffs, psi, clmo, max_deg = cn_hamiltonian_data

    # H_coeffs extends up to max_deg. We test for a degree n_missing > max_deg.
    # but still within psi_init_deg (max_deg + 2)
    n_missing = max_deg + 1

    Hn_zero = _get_homogeneous_terms(H_coeffs, n_missing, psi)
    assert np.all(Hn_zero == 0), "vector for missing degree is not zero"
    # The length of Hn_zero should correspond to psi[6, n_missing]
    assert len(Hn_zero) == psi[6, n_missing], "wrong length for zero vector"


@pytest.mark.parametrize("cn_hamiltonian_data", [2, 3, 4, 6], indirect=True)
def test_get_homogeneous_terms_when_n_is_at_psi_table_edge(cn_hamiltonian_data):
    H_coeffs, psi, clmo, max_deg = cn_hamiltonian_data

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
    H2 = polynomial_zero_list(max_deg, psi)
    idx = encode_multiindex((1,0,0,1,0,0), 2, psi, clmo)   # q1 p1
    H2[2][idx] = lam
    idx = encode_multiindex((0,1,0,0,1,0), 2, psi, clmo)   # q2 p2
    H2[2][idx] = 1j*w1
    idx = encode_multiindex((0,0,1,0,0,1), 2, psi, clmo)   # q3 p3
    H2[2][idx] = 1j*w2

    # bracket restricted to degree n because both inputs are homogeneous
    # PB = poisson_bracket_degree2(H2[2], Gn, n, psi, clmo) # Old line
    
    # Use _poly_poisson for homogeneous inputs H2[2] (degree 2) and Gn (degree n)
    # Result is homogeneous of degree 2 + n - 2 = n
    PB_coeffs = _poly_poisson(H2[2], 2, Gn, n, psi, clmo)

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

    # --- Setup H_coeffs_list ---
    # H is always c_H * q2*p2 (degree 2)
    H_deg_actual = 2
    H_exps = (0,1,0,0,1,0) 
    H_coeffs_list = polynomial_zero_list(N_max_test, psi)
    idx_H = encode_multiindex(H_exps, H_deg_actual, psi, clmo)
    if H_deg_actual <= N_max_test: # Ensure degree is within bounds of the list
        H_coeffs_list[H_deg_actual][idx_H] = H_coeff_val

    # --- Setup G_coeffs_list ---
    # G is c_G * q1^A * p2 (or similar based on G_exps)
    G_coeffs_list = polynomial_zero_list(N_max_test, psi) # G_n is just one component
    # The G_n passed to _apply_lie_transform is a single ndarray, not a list.
    # So, G_coeffs_list itself is not directly used but helps create G_n_array.
    G_n_array = polynomial_zero_list(G_deg_actual, psi)[G_deg_actual] # Get a correctly sized array for G_deg_actual
    
    idx_G = encode_multiindex(G_exps, G_deg_actual, psi, clmo)
    G_n_array[idx_G] = G_coeff_val
    
    # Call the function under test
    H1_transformed_coeffs = _apply_lie_transform(H_coeffs_list, G_n_array, G_deg_actual, N_max_test, psi, clmo, tol=1e-15)

    # --- SymPy Reference Calculation ---
    q1,q2,q3,p1,p2,p3 = sp.symbols('q1 q2 q3 p1 p2 p3')
    coords = (q1,q2,q3,p1,p2,p3)

    # Construct Hsym
    Hsym = sp.sympify(H_coeff_val) # Handles complex numbers correctly
    for i, exp_val in enumerate(H_exps):
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

    # Calculate Lie series: H_ref = sum_{k=0 to K_series} Ad_G^k(H) / k!
    # K_series matches K = max(1, deg_G - 1) from _apply_lie_transform
    K_series = max(1, G_deg_actual - 1)
    
    current_ad_term_sym = Hsym 
    Href_sym_calc = Hsym # Term for k=0

    if K_series > 0 : # Only proceed if there are bracket terms to add
        for k_val in range(1, K_series + 1):
            current_ad_term_sym = sympy_poisson_bracket(current_ad_term_sym, Gsym, coords)
            Href_sym_calc += current_ad_term_sym / math.factorial(k_val)
    
    # Convert the SymPy reference Href_sym_calc to our polynomial coefficient list format
    # The list(coords) is important as sympy2poly expects a Python list of symbols.
    # psi and clmo should be the ones initialized with N_max_test.
    Href_poly = sympy2poly(Href_sym_calc, list(coords), psi, clmo)

    # --- Comparison ---
    length_error_msg = f"Test '{test_name}': Output H1_transformed_coeffs has unexpected length {len(H1_transformed_coeffs)}, expected {N_max_test + 1}"
    assert len(H1_transformed_coeffs) == N_max_test + 1, length_error_msg

    for d in range(N_max_test + 1):
        coeffs_from_lie_transform = H1_transformed_coeffs[d]
        
        if d < len(Href_poly):
            coeffs_from_sympy_ref = Href_poly[d]
        else:
            # If Href_poly doesn't have this degree, all coeffs are zero.
            expected_size = psi[N_VARS, d] if d < psi.shape[1] else 0 
            if expected_size < 0: expected_size = 0 
            coeffs_from_sympy_ref = np.zeros(expected_size, dtype=np.complex128)

        # Reshape scalar-like 0-dim arrays that might come from make_poly for degree 0 if not careful
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
    H_coeffs, psi, clmo, max_deg = cn_hamiltonian_data
    mu_earth_moon = 0.012150585609624
    point = L1Point(mu=mu_earth_moon)
    H_out, G_total = lie_transform(point, H_coeffs, psi, clmo, max_deg)

    # property: no bad monomials remain in any degree ≥3
    for n in range(3, max_deg + 1):
        bad = _select_terms_for_elimination(H_out[n], n, clmo)
        assert not bad.any(), (
            f"Bad monomials not eliminated at degree {n}: {np.where(bad!=0)}")

    # quadratic part should be exactly what we started with
    assert np.allclose(H_out[2], H_coeffs[2], atol=0, rtol=0)
