import math

import numpy as np
import pytest
import sympy as sp

from hiten.system.center import CenterManifold
from hiten.algorithms.center.hamiltonian import build_physical_hamiltonian
from hiten.algorithms.center.lie import (_apply_coord_transform,
                                   _apply_poly_transform, _evaluate_transform,
                                   _get_homogeneous_terms, _lie_expansion,
                                   _lie_transform,
                                   _select_terms_for_elimination,
                                   _solve_homological_equation)
from hiten.algorithms.polynomial.algebra import _poly_poisson
from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                               decode_multiindex,
                                               encode_multiindex, make_poly)
from hiten.algorithms.polynomial.conversion import sympy2poly
from hiten.algorithms.polynomial.operations import (
    polynomial_evaluate, polynomial_poisson_bracket, polynomial_zero_list)
from hiten.algorithms.center.transforms import _local2realmodal, substitute_complex
from hiten.utils.config import N_VARS
from hiten.system.base import System, systemConfig
from hiten.system.body import Body
from hiten.utils.constants import Constants

TEST_L_POINT_IDX = 1
TEST_MAX_DEG = 6


@pytest.fixture(scope="module")
def lie_test_setup():
    Earth = Body("Earth", Constants.bodies["earth"]["mass"], Constants.bodies["earth"]["radius"], "blue")
    Moon = Body("Moon", Constants.bodies["moon"]["mass"], Constants.bodies["moon"]["radius"], "gray", Earth)
    distance = Constants.get_orbital_distance("earth", "moon")
    system = System(systemConfig(Earth, Moon, distance))
    libration_point = system.get_libration_point(TEST_L_POINT_IDX)

    cm = CenterManifold(libration_point, TEST_MAX_DEG)
    cm.compute()
    return cm


def test_get_homogeneous_terms(lie_test_setup):
    cm = lie_test_setup
    H_coeffs = cm.cache_get(('hamiltonian', TEST_MAX_DEG, 'center_manifold_real'))
    psi = cm.psi

    n = 3  # Test for degree 3 terms
    if n > TEST_MAX_DEG:

        Hn = _get_homogeneous_terms(H_coeffs, n, psi)
        assert np.all(Hn == 0), f"vector for n={n} (n > max_deg={TEST_MAX_DEG}) is not zero"
        assert len(Hn) == psi[6, n], f"wrong length for zero vector for n={n} (n > max_deg={TEST_MAX_DEG})"
    else:  # n <= max_deg (e.g. max_deg = 3, 4, or 6)
        Hn = _get_homogeneous_terms(H_coeffs, n, psi)
        expected_Hn = H_coeffs[n]
        assert np.array_equal(Hn, expected_Hn), "returned vector is not H_n"
        if Hn.size > 0:
            original_coeff_val = H_coeffs[n][0]
            Hn[0] += 1.0
            assert H_coeffs[n][0] == original_coeff_val, "Original H_coeffs was modified!"
            assert Hn[0] != original_coeff_val, "Copy was not modified or not a proper copy."
        elif expected_Hn.size == 0:
            pass


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("n", [3, 4, 6])
def test_select_terms_for_elimination(seed, n, lie_test_setup):
    cm = lie_test_setup
    psi = cm.psi
    clmo = cm.clmo
    size = psi[6, n]
    rng  = np.random.default_rng(seed)

    Hn_orig = (rng.uniform(-1, 1, size) + 1j*rng.uniform(-1, 1, size)).astype(np.complex128)
    Hn_for_mutation_check = Hn_orig.copy()

    got = _select_terms_for_elimination(Hn_orig, n, clmo)

    assert isinstance(got, np.ndarray), "Output should be a numpy array."
    assert got.shape == Hn_orig.shape, \
        f"Output shape {got.shape} does not match input shape {Hn_orig.shape}."
    assert got.dtype == Hn_orig.dtype, \
        f"Output dtype {got.dtype} does not match input dtype {Hn_orig.dtype}."

    for pos in range(size):
        k = decode_multiindex(pos, n, clmo)
        original_value_at_pos = Hn_orig[pos]

        if k[0] == k[3]:
            assert got[pos] == 0j, \
                f"For n={n}, pos={pos} (k={k} where k[0]==k[3]), Hn_orig[{pos}]={original_value_at_pos}. " \
                f"Expected got[{pos}]=0j, but got {got[pos]}."
        else:
            assert got[pos] == original_value_at_pos, \
                f"For n={n}, pos={pos} (k={k} where k[0]!=k[3]), Hn_orig[{pos}]={original_value_at_pos}. " \
                f"Expected got[{pos}]={original_value_at_pos}, but got {got[pos]}."

    assert np.array_equal(Hn_orig, Hn_for_mutation_check), \
        "Input Hn_orig was mutated by _select_terms_for_elimination. " \
        "The original Hn should remain unchanged as it might be used elsewhere."


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_homological_property(n, lie_test_setup):
    cm = lie_test_setup
    psi = cm.psi
    clmo = cm.clmo
    encode_dict = cm.encode_dict_list

    lam, w1, w2 = 3.1, 2.4, 2.2
    eta = np.array([lam, 1j*w1, 1j*w2], dtype=np.complex128)

    size = psi[6, n]
    rng  = np.random.default_rng(1234)
    Hn_bad = np.zeros(size, dtype=np.complex128)
    for pos in range(size):
        k = decode_multiindex(pos, n, clmo)
        if k[0] != k[3]:
            Hn_bad[pos] = rng.normal() + 1j*rng.normal()

    Gn = _solve_homological_equation(Hn_bad, n, eta, clmo)

    H2_list = polynomial_zero_list(TEST_MAX_DEG, psi)
    idx = encode_multiindex((1,0,0,1,0,0), 2, encode_dict) # q1 p1
    H2_list[2][idx] = lam
    idx = encode_multiindex((0,1,0,0,1,0), 2, encode_dict) # q2 p2
    H2_list[2][idx] = 1j*w1
    idx = encode_multiindex((0,0,1,0,0,1), 2, encode_dict)   # q3 p3
    H2_list[2][idx] = 1j*w2

    PB_coeffs = _poly_poisson(H2_list[2], 2, Gn, n, psi, clmo, encode_dict)

    assert np.allclose(PB_coeffs, -Hn_bad, atol=1e-14, rtol=1e-14)

    for pos, g in enumerate(Gn):
        k = decode_multiindex(pos, n, clmo)
        if k[0] == k[3]:
            assert g == 0


test_params = [
    pytest.param("base_degG3_Nmax4_realH", 3, (2,0,0,0,1,0), 0.7, 1.3, 4, id="Base_degG3_Nmax4_realH"),
    pytest.param("high_degG5_Nmax6_realH", 5, (4,0,0,0,1,0), 0.7, 1.3, 6, id="High_degG5_Nmax6_realH"), # Reduced N_max to stay within bounds
    pytest.param("Nmax6_degG4_realH", 4, (3,0,0,0,1,0), 0.7, 1.3, 6, id="Nmax6_degG4_realH_Term2_deg6"), # deg(H)=2, deg(G)=4 -> {{H,G},G} is deg 6
    pytest.param("complexH_degG3_Nmax4", 3, (2,0,0,0,1,0), 0.7, 1.3+0.5j, 4, id="ComplexH_degG3_Nmax4"),
    pytest.param("degG2_Nmax4_realH", 2, (1,0,0,0,1,0), 0.7, 1.3, 4, id="Low_degG2_Nmax4_realH_K_is_1"), # K = max(1, deg_G-1) = max(1,1)=1
]

@pytest.mark.parametrize(
    "test_name, G_deg_actual, G_exps, G_coeff_val, H_coeff_val, N_max_test",
    test_params
)
def test_apply_poly_transform(test_name, G_deg_actual, G_exps, G_coeff_val, H_coeff_val, N_max_test, lie_test_setup):
    cm = lie_test_setup
    psi = cm.psi
    clmo = cm.clmo
    encode_dict = cm.encode_dict_list

    H_deg_actual = 2
    H_exps_tuple = (0,1,0,0,1,0)
    H_exps_np = np.array(H_exps_tuple, dtype=np.int64)
    H_coeffs_list = polynomial_zero_list(N_max_test, psi)
    idx_H = encode_multiindex(H_exps_np, H_deg_actual, encode_dict)
    if H_deg_actual <= N_max_test:
        H_coeffs_list[H_deg_actual][idx_H] = H_coeff_val

    _ = polynomial_zero_list(N_max_test, psi)

    G_n_array = make_poly(G_deg_actual, psi)

    G_exps_np = np.array(G_exps, dtype=np.int64)
    idx_G = encode_multiindex(G_exps_np, G_deg_actual, encode_dict)
    G_n_array[idx_G] = G_coeff_val
    
    H1_transformed_coeffs = _apply_poly_transform(H_coeffs_list, G_n_array, G_deg_actual, N_max_test, psi, clmo, encode_dict, tol=1e-15)

    q1,q2,q3,p1,p2,p3 = sp.symbols('q1 q2 q3 p1 p2 p3')
    coords = (q1,q2,q3,p1,p2,p3)

    Hsym = sp.sympify(H_coeff_val) 
    for i, exp_val in enumerate(H_exps_tuple):
        if exp_val > 0:
            Hsym *= coords[i]**exp_val

    Gsym = sp.sympify(G_coeff_val)
    for i, exp_val in enumerate(G_exps):
        if exp_val > 0:
            Gsym *= coords[i]**exp_val
    
    def sympy_poisson_bracket(f, g, variables_tuple):
        q_vars = variables_tuple[:len(variables_tuple)//2]
        p_vars = variables_tuple[len(variables_tuple)//2:]
        bracket = sp.S.Zero
        for i_pb in range(len(q_vars)):
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


def test_lie_transform_removes_bad_terms(lie_test_setup):
    cm = lie_test_setup
    H_coeffs = cm.cache_get(('hamiltonian', TEST_MAX_DEG, 'center_manifold_real'))
    psi = cm.psi
    clmo = cm.clmo
    max_deg = TEST_MAX_DEG

    point = cm.point
    H_out, _, _ = _lie_transform(point, H_coeffs, psi, clmo, max_deg)

    tolerance = 1e-15
    
    for n in range(3, max_deg + 1):
        bad = _select_terms_for_elimination(H_out[n], n, clmo)
        max_bad_coeff = np.max(np.abs(bad)) if bad.size > 0 else 0.0
        assert max_bad_coeff < tolerance, (
            f"Bad monomials not sufficiently eliminated at degree {n}. "
            f"Max coefficient: {max_bad_coeff:.2e}, tolerance: {tolerance:.2e}. "
            f"Non-zero positions: {np.where(np.abs(bad) >= tolerance)}")

    assert np.allclose(H_out[2], H_coeffs[2], atol=0, rtol=0)


def test_lie_transform_on_center_manifold(lie_test_setup):
    cm = lie_test_setup
    point = cm.point
    psi = cm.psi
    clmo = cm.clmo
    max_degree = TEST_MAX_DEG
    
    H_phys = build_physical_hamiltonian(point, max_degree)
    H_rn = _local2realmodal(point, H_phys, max_degree, psi, clmo)
    H_cn = substitute_complex(H_rn, max_degree, psi, clmo)
    
    poly_trans, _, _ = _lie_transform(point, H_cn, psi, clmo, max_degree)
    
    print("\nAnalyzing transformed Hamiltonian on center manifold:")
    print("="*60)
    
    if len(poly_trans) > 1:
        H1 = poly_trans[1]
        print(f"Linear terms in transformed H: {np.count_nonzero(H1)} non-zero")
        for i, coeff in enumerate(H1):
            if abs(coeff) > 1e-15:
                print(f"  x_{i}: {coeff:.6e}")
    
    if len(poly_trans) > 2:
        H2 = poly_trans[2]

        cross_terms = []
        for idx, coeff in enumerate(H2):
            if abs(coeff) > 1e-15:
                k = decode_multiindex(idx, 2, clmo)
                has_q1p1 = (k[0] > 0 or k[3] > 0)
                has_cm = (k[1] > 0 or k[2] > 0 or k[4] > 0 or k[5] > 0)
                if has_q1p1 and has_cm:
                    cross_terms.append((k, coeff))
        
        print(f"\nQuadratic cross-terms (q1/p1 x CM): {len(cross_terms)}")
        for k, coeff in cross_terms[:5]:
            vars_str = []
            var_names = ['q1', 'q2', 'q3', 'p1', 'p2', 'p3']
            for i, power in enumerate(k):
                if power > 0:
                    vars_str.append(f"{var_names[i]}^{power}" if power > 1 else var_names[i])
            print(f"  {' '.join(vars_str)}: {coeff:.6e}")


def test_lie_expansion_accumulation(lie_test_setup):
    cm = lie_test_setup
    point = cm.point
    psi = cm.psi
    clmo = cm.clmo
    
    cm_point = np.array([0, 1e-3+1e-3j, 0.5e-3+0.2e-3j,
                   0, 1e-3-1e-3j, 0.5e-3-0.2e-3j])
    
    print("\nTesting transformation accumulation:")
    print("-" * 50)
    
    for max_deg in [3, 4, 5, 6]:
        H_phys = build_physical_hamiltonian(point, 6)
        H_rn = _local2realmodal(point, H_phys, 6, psi, clmo)
        H_cn = substitute_complex(H_rn, 6, psi, clmo)
        _, poly_G_total, _ = _lie_transform(point, H_cn, psi, clmo, max_deg)
        
        expansions = _lie_expansion(poly_G_total, max_deg, psi, clmo, 
                                   tol=1e-15, inverse=False)
        
        modal = _evaluate_transform(expansions, cm_point, clmo)
        
        print(f"\nDegree {max_deg}:")
        print(f"  q1 = {modal[0]:.6e}")
        print(f"  p1 = {modal[3]:.6e}")
        
        if max_deg < len(poly_G_total) and poly_G_total[max_deg] is not None:
            has_content = poly_G_total[max_deg].any()
            print(f"  G_{max_deg} has content: {has_content}")


def test_lie_expansion_application(lie_test_setup):
    cm = lie_test_setup
    psi = cm.psi
    clmo = cm.clmo
    max_degree = TEST_MAX_DEG

    poly_G_total = cm.cache_get(('generating_functions', TEST_MAX_DEG))
    
    # Test point on center manifold
    cm_point = np.array([0, 1e-3+1e-3j, 0.5e-3+0.2e-3j,
                   0, 1e-3-1e-3j, 0.5e-3-0.2e-3j])
    
    print("\nTesting Lie series convergence for each generator:")
    print("="*70)
    
    # Start with identity transformation
    identity_coords = []
    for i in range(6):
        poly = polynomial_zero_list(max_degree, psi)
        poly[1][i] = 1.0
        identity_coords.append(poly)
    
    # Apply each generator individually to see its effect
    for n in range(3, min(7, len(poly_G_total))):
        if poly_G_total[n] is None or not poly_G_total[n].any():
            continue
            
        print(f"\nApplying only G_{n}:")
        
        # Create polynomial for this generator only
        test_G = polynomial_zero_list(max_degree, psi)
        test_G[n] = poly_G_total[n].copy()
        
        # Apply to q1 coordinate (index 0)
        encode_dict_list = _create_encode_dict_from_clmo(clmo)
        transformed_q1 = _apply_coord_transform(
            identity_coords[0], test_G, max_degree, psi, clmo, encode_dict_list, 1e-15
        )
        
        # Evaluate the transformation at the test point
        q1_value = polynomial_evaluate(transformed_q1, cm_point, clmo)
        
        print(f"  q1 transformation: {q1_value:.6e}")
        print(f"  Change from identity: {abs(q1_value):.6e}")
        
        for d in range(min(len(transformed_q1), 6)):
            if transformed_q1[d].any():
                nonzero = np.count_nonzero(np.abs(transformed_q1[d]) > 1e-15)
                if nonzero > 0:
                    print(f"    Degree {d}: {nonzero} non-zero coefficients")

    cumulative_results = {}
    
    for max_deg in [3, 4, 5, 6]:
        expansions = _lie_expansion(poly_G_total, max_deg, psi, clmo, tol=1e-15, inverse=False)
        modal = _evaluate_transform(expansions, cm_point, clmo)
        cumulative_results[max_deg] = modal.copy()
    
    print("\nIncremental changes from adding each generator:")
    for deg in [4, 5, 6]:
        prev_q1 = cumulative_results[deg-1][0]
        curr_q1 = cumulative_results[deg][0]
        diff = curr_q1 - prev_q1
        
        print(f"\nG_{deg} contribution to q1:")
        print(f"  Previous (up to G_{deg-1}): {prev_q1:.6e}")
        print(f"  Current (up to G_{deg}):    {curr_q1:.6e}")
        print(f"  Difference:                 {diff:.6e}")
        print(f"  |Difference|:               {abs(diff):.6e}")
        
        if deg < len(poly_G_total) and poly_G_total[deg] is not None:
            G_mag = np.max(np.abs(poly_G_total[deg]))
            print(f"  Max |G_{deg}| coefficient:   {G_mag:.6e}")


def test_lie_expansion_degree_scaling(lie_test_setup):
    cm = lie_test_setup
    point = cm.point
    psi = cm.psi
    clmo = cm.clmo
    
    # Build and normalize Hamiltonian for different degrees
    H_phys = build_physical_hamiltonian(point, 6)  # Use max degree 6 for building
    H_rn = _local2realmodal(point, H_phys, 6, psi, clmo)
    H_cn = substitute_complex(H_rn, 6, psi, clmo)
    
    # Test point on center manifold
    cm_point = np.array([0, 1e-3+1e-3j, 0.5e-3+0.2e-3j,   # q1,q2,q3
                   0, 1e-3-1e-3j, 0.5e-3-0.2e-3j])  # p1,p2,p3  (complex conj)
    
    degrees_to_test = [4, 5, 6]
    errors = []
    
    print("\nTesting convergence with polynomial degree:")
    print("Degree | q1 leak | p1 leak | Max leak")
    print("-------|---------|---------|----------")
    
    for max_deg in degrees_to_test:
        _, poly_G_total, _ = _lie_transform(point, H_cn, psi, clmo, max_deg)
        
        expansions = _lie_expansion(poly_G_total, max_deg, psi, clmo, tol=1e-15, inverse=False)
        
        modal = _evaluate_transform(expansions, cm_point, clmo)
        
        q1_leak = abs(modal[0])
        p1_leak = abs(modal[3])
        max_leak = max(q1_leak, p1_leak)
        
        errors.append(max_leak)
        
        print(f"  {max_deg}    | {q1_leak:.2e} | {p1_leak:.2e} | {max_leak:.2e}")
    
    convergence_factor = 1.1
    
    for i in range(1, len(errors)):
        prev_error = errors[i-1]
        curr_error = errors[i]
        
        assert curr_error <= convergence_factor * prev_error, (
            f"Convergence failure: degree {degrees_to_test[i]} error {curr_error:.2e} "
            f"> {convergence_factor} x degree {degrees_to_test[i-1]} error {prev_error:.2e}"
        )


def test_lie_expansion_amplitude_scaling(lie_test_setup):
    cm = lie_test_setup
    poly_G_total = cm.cache_get(('generating_functions', TEST_MAX_DEG))
    psi = cm.psi
    clmo = cm.clmo
    max_degree = TEST_MAX_DEG
    
    # Generate the coordinate transformation expansions
    expansions = _lie_expansion(poly_G_total, max_degree, psi, clmo, tol=1e-15, inverse=False)
    
    # Test different amplitudes (scaling the base coordinates)
    base_coords = np.array([0, 1+1j, 0.5+0.2j,   # q1,q2,q3
                           0, 1-1j, 0.5-0.2j])   # p1,p2,p3  (complex conj)
    
    amplitudes = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    
    results = []
    
    print("\nTesting error scaling with center manifold amplitude:")
    print("Amplitude | Input norm | q1 leak | p1 leak | Max leak | leak/amp^2")
    print("----------|------------|---------|---------|----------|----------")
    
    for amp_scale in amplitudes:
        # Scale the test coordinates
        cm_point = amp_scale * base_coords
        
        # Compute input amplitude (only center manifold coordinates)
        input_amp = np.linalg.norm([cm_point[1], cm_point[2], cm_point[4], cm_point[5]])
        
        # Evaluate transformation
        modal = _evaluate_transform(expansions, cm_point, clmo)
        
        # Compute leak magnitudes
        q1_leak = abs(modal[0])
        p1_leak = abs(modal[3])
        max_leak = max(q1_leak, p1_leak)
        
        # Expected quadratic scaling
        leak_per_amp2 = max_leak / (input_amp**2) if input_amp > 0 else 0
        
        results.append({
            'amp_scale': amp_scale,
            'input_amp': input_amp,
            'q1_leak': q1_leak,
            'p1_leak': p1_leak,
            'max_leak': max_leak,
            'leak_per_amp2': leak_per_amp2
        })
        
        print(f"{amp_scale:.1e} | {input_amp:.2e} | {q1_leak:.2e} | {p1_leak:.2e} | {max_leak:.2e} | {leak_per_amp2:.2e}")
    
    leak_ratios = [r['leak_per_amp2'] for r in results]
    
    ratios_to_check = leak_ratios[1:]
    
    if len(ratios_to_check) > 1:
        mean_ratio = np.mean(ratios_to_check)
        std_ratio = np.std(ratios_to_check)
        
        max_variation = 0.5 * mean_ratio
        
        for i, ratio in enumerate(ratios_to_check):
            assert abs(ratio - mean_ratio) <= max_variation, (
                f"Quadratic scaling violation at amplitude {amplitudes[i+1]:.1e}: "
                f"leak/amp^2 = {ratio:.2e}, mean = {mean_ratio:.2e}, "
                f"deviation = {abs(ratio - mean_ratio):.2e} > tolerance {max_variation:.2e}"
            )
        
        print(f"\nQuadratic scaling verified:")
        print(f"Mean leak/amp^2 ratio: {mean_ratio:.2e}")
        print(f"Standard deviation: {std_ratio:.2e} ({100*std_ratio/mean_ratio:.1f}%)")
    
    for i in range(1, len(results)):
        prev_leak = results[i-1]['max_leak']
        curr_leak = results[i]['max_leak']
        
        assert curr_leak >= prev_leak, (
            f"Error should increase with amplitude: "
            f"amp {amplitudes[i]:.1e} leak {curr_leak:.2e} < "
            f"amp {amplitudes[i-1]:.1e} leak {prev_leak:.2e}"
        )


def test_lie_expansion_symplecticity(lie_test_setup):
    def poisson_matrix(expansions, clmo, psi, max_degree, test_point):
        """Return the 6x6 matrix M_ij = {Phi_i, Phi_j}(point)."""
        encode_dict_list = _create_encode_dict_from_clmo(clmo)
        n = 6
        M = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            for j in range(i+1, n):
                bracket = polynomial_poisson_bracket(
                    expansions[i], expansions[j], max_degree, psi, clmo, encode_dict_list
                )
                val = polynomial_evaluate(bracket, test_point, clmo)
                M[i, j] = val
                M[j, i] = -val  # antisymmetry
        return M

    def analyze_symplectic_error(M, Omega, description=""):
        """Analyze and return detailed symplectic error information."""
        error_matrix = M - Omega
        max_error = np.linalg.norm(error_matrix, np.inf)
        
        # Specific canonical relationships
        canonical_errors = {
            'q1_p1': abs(M[0,3] - 1.0),  # Should be +1
            'q2_p2': abs(M[1,4] - 1.0),  # Should be +1  
            'q3_p3': abs(M[2,5] - 1.0),  # Should be +1
            'q1_q2': abs(M[0,1]),        # Should be 0
            'p1_p2': abs(M[3,4]),        # Should be 0
            'q1_p2': abs(M[0,4]),        # Should be 0
        }
        
        return max_error, canonical_errors

    cm = lie_test_setup
    point = cm.point
    psi = cm.psi
    clmo = cm.clmo
    
    H_phys = build_physical_hamiltonian(point, 6)
    H_rn = _local2realmodal(point, H_phys, 6, psi, clmo)
    H_cn = substitute_complex(H_rn, 6, psi, clmo)
    
    Omega = np.block([[np.zeros((3,3)),  np.eye(3)],
                      [-np.eye(3), np.zeros((3,3))]])
    
    degrees_to_test = [4, 5, 6]
    base_amplitudes = [1e-4, 5e-4, 1e-3, 5e-3]
    
    test_point_templates = [
        np.array([0, 1+1j, 0.5+0.2j, 0, 1-1j, 0.5-0.2j]),      # Standard complex conjugate
        np.array([0, 1+0j, 0+1j, 0, 1+0j, 0-1j]),              # Real/imaginary separation  
        np.array([0, 1+0.5j, 0.3+0.8j, 0, 1-0.5j, 0.3-0.8j]), # Different conjugate pattern
        np.array([0, 0.7+0.7j, 0.1+0.1j, 0, 0.7-0.7j, 0.1-0.1j]), # Smaller, equal real/imag
    ]
    
    print(f"\nDirect Symplecticity Test")
    print("="*60)
    
    all_results = []
    overall_max_error = 0.0
    degree_max_errors = {}  # Track max error per degree
    
    for degree in degrees_to_test:
        print(f"\nDegree {degree}:")
        print("-" * 40)
        
        _, poly_G_total, _ = _lie_transform(point, H_cn, psi, clmo, degree)
        expansions = _lie_expansion(poly_G_total, degree, psi, clmo, 
                                   tol=1e-15, inverse=False, sign=1, restrict=False)
        
        degree_results = []
        
        for _, base_amp in enumerate(base_amplitudes):
            for point_idx, template in enumerate(test_point_templates):
                test_point = base_amp * template
                
                M = poisson_matrix(expansions, clmo, psi, degree, test_point)
                
                max_error, canonical_errors = analyze_symplectic_error(M, Omega)
                
                result = {
                    'degree': degree,
                    'amplitude': base_amp, 
                    'point_type': point_idx,
                    'max_error': max_error,
                    'canonical_errors': canonical_errors,
                    'test_point': test_point
                }
                
                degree_results.append(result)
                all_results.append(result)
                overall_max_error = max(overall_max_error, max_error)
                
        degree_max_error = max(r['max_error'] for r in degree_results)
        degree_avg_error = np.mean([r['max_error'] for r in degree_results])
        degree_max_errors[degree] = degree_max_error
        
        print(f"  Max error: {degree_max_error:.2e}")
        print(f"  Avg error: {degree_avg_error:.2e}")
        print(f"  Tests run: {len(degree_results)}")
    
    print(f"\nOverall Results:")
    print("="*30)
    print(f"Total tests: {len(all_results)}")
    print(f"Overall max error: {overall_max_error:.2e}")
    
    print(f"\nError vs Degree (avg over all amplitudes/points):")
    for degree in degrees_to_test:
        degree_results = [r for r in all_results if r['degree'] == degree]
        avg_error = np.mean([r['max_error'] for r in degree_results])
        print(f"  Degree {degree}: {avg_error:.2e}")
    
    highest_degree = max(degrees_to_test)
    print(f"\nError vs Amplitude (degree {highest_degree}, avg over all point types):")
    for amp in base_amplitudes:
        amp_results = [r for r in all_results 
                      if r['degree'] == highest_degree and r['amplitude'] == amp]
        if amp_results:
            avg_error = np.mean([r['max_error'] for r in amp_results])
            print(f"  Amplitude {amp:.1e}: {avg_error:.2e}")
    
    worst_result = max(all_results, key=lambda r: r['max_error'])
    print(f"\nWorst case:")
    print(f"  Degree: {worst_result['degree']}, Amplitude: {worst_result['amplitude']:.1e}")
    print(f"  Point type: {worst_result['point_type']}, Error: {worst_result['max_error']:.2e}")
    
    print(f"  Canonical relationship errors:")
    for name, error in worst_result['canonical_errors'].items():
        print(f"    {name}: {error:.2e}")
    
    print(f"\nDegree-specific tolerance verification:")
    all_passed = True
    
    for degree in degrees_to_test:
        max_amplitude_tested = max(base_amplitudes)
        
        degree_tolerance = 1e-15 * (10 ** (8 - degree)) * (max_amplitude_tested / 1e-4) ** (degree - 2)
        
        degree_error = degree_max_errors[degree]
        passed = degree_error < degree_tolerance
        all_passed = all_passed and passed
        
        print(f"  Degree {degree}: error={degree_error:.2e}, tolerance={degree_tolerance:.2e}, {'PASS' if passed else 'FAIL'}")
    
    error_ratios = []
    for i in range(1, len(degrees_to_test)):
        prev_degree = degrees_to_test[i-1]
        curr_degree = degrees_to_test[i]
        ratio = degree_max_errors[curr_degree] / degree_max_errors[prev_degree]
        error_ratios.append(ratio)
        print(f"\nError reduction {prev_degree}→{curr_degree}: {ratio:.2e} ({1/ratio:.1f}x improvement)")
    
    avg_reduction_factor = np.mean([1/r for r in error_ratios])
    print(f"\nAverage error reduction per degree: {avg_reduction_factor:.1f}x")
    
    assert all_passed, (
        f"Symplecticity test failed for some polynomial degrees. "
        f"See degree-specific results above."
    )
    
    assert avg_reduction_factor > 10, (
        f"Insufficient convergence rate: average error reduction {avg_reduction_factor:.1f}x < 10x per degree"
    )
    
    print(f"\nDirect symplecticity test passed!")
    print(f"All {len(all_results)} test combinations satisfy canonical relationships with appropriate tolerances.")
    print(f"Strong convergence verified: {avg_reduction_factor:.1f}x average error reduction per degree.")
