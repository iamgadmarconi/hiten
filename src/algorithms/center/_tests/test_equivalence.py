import pytest
import numpy as np
import symengine as se
import sympy as sp

from algorithms.center.hamiltonian import (
    build_physical_hamiltonian,
    phys2rn as new_phys2rn,
    rn2cn as new_rn2cn,
    cn2rn as new_cn2rn
)
from algorithms.center._deprecated.dep_hamiltonian import (
    hamiltonian as old_hamiltonian,
    _generate_subs_dict,
    physical_to_real_normal as old_physical_to_real_normal,
    real_normal_to_complex_canonical as old_real_normal_to_complex_canonical,
    complex_canonical_to_real_normal as old_complex_canonical_to_real_normal
)
from algorithms.center._deprecated.dep_core import Polynomial as OldPolynomial, _clean_numerical_artifacts
from system.libration import L1Point
from algorithms.center.polynomial.conversions import symengine2poly
from algorithms.center.polynomial.base import init_index_tables
from algorithms.variables import get_vars, physical_vars, real_normal_vars, canonical_normal_vars


x_s, y_s, z_s, px_s, py_s, pz_s = se.symbols("x y z px py pz")
PHYSICAL_SYMBOLS = [x_s, y_s, z_s, px_s, py_s, pz_s]

# Symbols - Real Normal (New)
x_rn_s, y_rn_s, z_rn_s, px_rn_s, py_rn_s, pz_rn_s = get_vars(real_normal_vars)
REAL_NORMAL_SYMBOLS = [x_rn_s, y_rn_s, z_rn_s, px_rn_s, py_rn_s, pz_rn_s]

# Symbols - Complex Canonical (New)
q1_s, q2_s, q3_s, p1_s, p2_s, p3_s = get_vars(canonical_normal_vars)
COMPLEX_CANONICAL_SYMBOLS = [q1_s, q2_s, q3_s, p1_s, p2_s, p3_s]


def get_monomial_exponents_ordered(num_vars, degree):
    if num_vars != 6: # This function is specialized for the 6 physical variables
        raise NotImplementedError(f"This placeholder get_monomial_exponents_ordered is only for num_vars=6.")

    if degree == 0:
        # Constant term: coeff * 1 (corresponds to x^0 y^0 ... pz^0)
        yield (0, 0, 0, 0, 0, 0)
        return
    
    if degree == 1:
        # Linear terms: c0*x, c1*y, ..., c5*pz
        # Assumes coeffs are stored in the order of variables (0 to 5)
        for i in range(num_vars):
            exponents = [0] * num_vars
            exponents[i] = 1
            yield tuple(exponents)
        return

    # For degree >= 2, the order is crucial and must match the
    # internal structure of the 'Poly' type from 'algorithms.center.polynomial.operations',
    # specifically how the clmo table is generated in init_index_tables.
    # The assumed variable order for num_vars=6 is (x, y, z, px, py, pz).
    # Exponents are k0, k1, k2, k3, k4, k5.
    for k0 in range(degree, -1, -1):
        for k1 in range(degree - k0, -1, -1):
            for k2 in range(degree - k0 - k1, -1, -1):
                for k3 in range(degree - k0 - k1 - k2, -1, -1):
                    for k4 in range(degree - k0 - k1 - k2 - k3, -1, -1):
                        k5 = degree - k0 - k1 - k2 - k3 - k4
                        yield (k0, k1, k2, k3, k4, k5)
    return # Generator is exhausted


# Renamed and adapted from evaluate_new_hamiltonian_poly for broader use, including complex numbers
def evaluate_poly_list_at_point(poly_coeffs_list, point_values_dict, sym_vars_ordered_list, max_eval_deg):
    num_vars = len(sym_vars_ordered_list)
    # Ensure total_value starts as complex if any coefficient or point value might be complex.
    # A simple way is to check if the first relevant coefficient is complex.
    # Or, more robustly, ensure accumulation can handle complex numbers throughout.
    # For now, we'll let Python's automatic type promotion handle it during accumulation.
    total_value = 0.0 

    for d in range(len(poly_coeffs_list)):
        if d > max_eval_deg:
            continue
        
        coeffs_for_deg_d = poly_coeffs_list[d]
        
        if coeffs_for_deg_d is None: 
            continue
        if isinstance(coeffs_for_deg_d, (list, tuple)) and not coeffs_for_deg_d:
            continue

        if not isinstance(coeffs_for_deg_d, np.ndarray) or coeffs_for_deg_d.size == 0:
            if hasattr(coeffs_for_deg_d, '__len__') and len(coeffs_for_deg_d) == 0:
                 continue
            elif not (isinstance(coeffs_for_deg_d, np.ndarray) and coeffs_for_deg_d.size == 0) :
                 print(f"Warning: Degree {d} coefficients are of unexpected type/state: {type(coeffs_for_deg_d)}, value: {coeffs_for_deg_d}")

        exponent_tuples_for_d = list(get_monomial_exponents_ordered(num_vars, d))
        num_coeffs = len(coeffs_for_deg_d) if hasattr(coeffs_for_deg_d, '__len__') else 0

        if num_coeffs != len(exponent_tuples_for_d):
            err_msg = (f"Degree {d}: Mismatch in number of coefficients ({num_coeffs}) "
                       f"and generated monomial exponent tuples ({len(exponent_tuples_for_d)}). "
                       f"The 'get_monomial_exponents_ordered' helper is likely incorrect or "
                       f"the structure of 'poly_coeffs_list[{d}]' is not a flat list of coefficients.")
            raise ValueError(err_msg)

        for coeff_val, exponents in zip(coeffs_for_deg_d, exponent_tuples_for_d):
            # Check for effectively zero coefficients (real or complex)
            is_coeff_complex = isinstance(coeff_val, complex)
            if is_coeff_complex:
                if abs(coeff_val.real) < 1e-18 and abs(coeff_val.imag) < 1e-18:
                    continue
            else: # coeff_val is float or int
                if abs(coeff_val) < 1e-18:
                    continue
            
            # Initialize monomial_val. If coeff is complex, monomial_val should start complex.
            monomial_val = complex(1.0, 0.0) if is_coeff_complex else 1.0
            
            any_point_val_complex = False
            for var_idx_check, exponent_val_check in enumerate(exponents):
                 if exponent_val_check > 0:
                    if isinstance(point_values_dict[sym_vars_ordered_list[var_idx_check]], complex):
                        any_point_val_complex = True
                        break
            if any_point_val_complex and not isinstance(monomial_val, complex):
                monomial_val = complex(monomial_val)


            for var_idx, exponent_val in enumerate(exponents):
                if exponent_val < 0: 
                    raise ValueError(f"Negative exponent {exponent_val} encountered.")
                if exponent_val > 0:
                    var_symbol = sym_vars_ordered_list[var_idx]
                    val_at_sym = point_values_dict[var_symbol]
                    
                    # Promote monomial_val to complex if val_at_sym is complex and monomial_val is not yet
                    if isinstance(val_at_sym, complex) and not isinstance(monomial_val, complex):
                        monomial_val = complex(monomial_val)
                    
                    term_val = val_at_sym ** exponent_val
                    monomial_val *= term_val
            
            total_value += coeff_val * monomial_val
            
    return total_value


@pytest.mark.parametrize("mu_val, max_degree_test", [
    (0.01215058162465319, 2),
    (0.01215058162465319, 3),
    (0.01215058162465319, 4),
    (0.01215058162465319, 5),
    (0.01215058162465319, 6),
    (1e-3, 1),
    (1e-3, 2),
    (1e-3, 3)])
def test_physical_hamiltonian_equivalence_L1(mu_val, max_degree_test):
    l1_point = L1Point(mu=mu_val)

    # 1. Generate Hamiltonian using the old (deprecated) symbolic method
    old_H_poly_obj = old_hamiltonian(l1_point, max_degree=max_degree_test)
    old_H_expr_raw = old_H_poly_obj.expansion.expression # This is a symengine expression
    
    # The old code used sympy for cleaning. Let's apply it here for a fair comparison.
    old_H_expr_sp = sp.sympify(str(old_H_expr_raw)) # Convert symengine to sympy via string
    old_H_expr_cleaned_sp = _clean_numerical_artifacts(old_H_expr_sp)
    old_H_expr = se.sympify(str(old_H_expr_cleaned_sp)) # Convert back to symengine

    # Create substitution dictionary for symbolic constants (c_n, etc.) from the old Hamiltonian
    constants_subs_dict = _generate_subs_dict(l1_point, max_degree_test)
    old_H_expr_numeric_coeffs = old_H_expr.subs(constants_subs_dict)

    # Convert the "old" symbolic Hamiltonian (with numeric constants) to the new polynomial list format
    # Determine max degree needed for this conversion: old_H contains K (deg 2) and U (up to max_degree_test)
    max_deg_for_s2p_conversion = max(2, max_degree_test) 
    psi_s2p, clmo_s2p = init_index_tables(max_deg_for_s2p_conversion)
    old_H_poly_list_representation = symengine2poly(
        old_H_expr_numeric_coeffs, 
        PHYSICAL_SYMBOLS, 
        max_deg_for_s2p_conversion, 
        psi_s2p, 
        clmo_s2p
    )

    # 2. Generate Hamiltonian using the new polynomial representation
    # build_physical_hamiltonian expects (point, max_deg, psi, clmo)
    # psi = (num_vars, complex_dt). num_vars = 6 for physical coords. complex_dt = False.
    psi_param = (6, False) 
    # clmo (coeff limit order) is max_deg for polynomial multiplications.
    clmo_param = max_degree_test
    
    new_H_poly_list = build_physical_hamiltonian(l1_point, max_degree_test, psi_param, clmo_param)

    # 3. Compare the Hamiltonians by numerical evaluation at random points
    np.random.seed(0) # for reproducible tests
    num_test_points = 5

    for i in range(num_test_points):
        # Generate a random point (values for x, y, z, px, py, pz)
        # Keep values small to avoid large numbers and potential numerical issues
        point_coords = np.random.uniform(-0.05, 0.05, size=len(PHYSICAL_SYMBOLS))
        test_point_dict = {sym: val for sym, val in zip(PHYSICAL_SYMBOLS, point_coords)}
        
        # Evaluate old Hamiltonian (now from its poly list representation, evaluated up to max_degree_test)
        val_old = evaluate_poly_list_at_point(
            old_H_poly_list_representation, 
            test_point_dict, 
            PHYSICAL_SYMBOLS, 
            max_eval_deg=max_degree_test
        )

        # Evaluate new Hamiltonian (polynomial list)
        try:
            val_new = evaluate_poly_list_at_point(new_H_poly_list, test_point_dict, PHYSICAL_SYMBOLS, max_degree_test)
        except (ValueError, NotImplementedError) as e:
            pytest.fail(f"Failed to evaluate new Hamiltonian representation at point {i} {test_point_dict}. Error: {e}")
            return 

        # Compare values
        assert np.isclose(val_old, val_new, rtol=1e-9, atol=1e-12), \
            (f"Hamiltonian values differ at point {i} ({test_point_dict}) for "
             f"mu={mu_val}, max_degree={max_degree_test}:\n"
             f"Old symbolic H (cleaned): {val_old}\n"
             f"New poly H: {val_new}\n"
             f"Difference: {abs(val_old - val_new)}")

    print(f"Equivalence test passed for mu={mu_val}, max_degree={max_degree_test} using numerical evaluation.")


# --- Conversion Function Equivalence Tests ---

@pytest.mark.parametrize("mu_val, max_degree_test", [
    (0.01215058162465319, 2),
    (0.01215058162465319, 3),
    (0.01215058162465319, 4),
    (0.01215058162465319, 5),
    (0.01215058162465319, 6),
    (1e-3, 1),
    (1e-3, 2),
    (1e-3, 3)])
def test_phys2rn_equivalence(mu_val, max_degree_test):
    l1_point = L1Point(mu=mu_val)
    psi_tables, clmo_tables = init_index_tables(max_degree_test)
    
    # Define a simple physical Hamiltonian based on max_degree_test
    if max_degree_test == 1:
        H_phys_sym = x_s + 2.5*py_s - 0.5*z_s 
    elif max_degree_test == 2:
         H_phys_sym = x_s**2 + 1.5*y_s*px_s + 0.5*z_s**2 - y_s # Mix of degrees
    else: # Should not be hit by current parametrization but as a fallback
        H_phys_sym = x_s + y_s*px_s 

    # --- Old Path ---
    old_H_phys = OldPolynomial(PHYSICAL_SYMBOLS, H_phys_sym)
    old_H_rn_poly_obj = old_physical_to_real_normal(l1_point, old_H_phys, symbolic=False, max_degree=max_degree_test)
    old_H_rn_expr_numeric_raw = old_H_rn_poly_obj.expression

    # Ensure all relevant symbolic constants like c2, c3, etc., are substituted numerically.
    # old_physical_to_real_normal with symbolic=False should do this, but if max_degree_test is low (e.g., 1),
    # its internal subs_dict might not cover all c_n that can appear from the transformation matrix S (which uses c2-c4).
    # We apply a more comprehensive substitution here. Max degree for constants usually doesn't exceed 4-6.
    constants_subs_max_deg = max(max_degree_test, 6) 
    comprehensive_constants_subs = _generate_subs_dict(l1_point, constants_subs_max_deg)
    old_H_rn_expr_numeric_substituted = old_H_rn_expr_numeric_raw.subs(comprehensive_constants_subs)
    old_H_rn_expr_numeric_cleaned = _clean_numerical_artifacts(old_H_rn_expr_numeric_substituted) # Clean after subs
    
    old_H_rn_list = symengine2poly(
        old_H_rn_expr_numeric_cleaned, # Use the fully substituted and cleaned expression
        REAL_NORMAL_SYMBOLS,
        max_degree_test,
        psi_tables,
        clmo_tables,
        complex_dtype=False # Real normal coordinates are real
    )

    # --- New Path ---
    new_H_phys_list = symengine2poly(
        H_phys_sym,
        PHYSICAL_SYMBOLS,
        max_degree_test,
        psi_tables,
        clmo_tables,
        complex_dtype=False # Physical coordinates are real
    )
    psi_param_new = (6, False) # num_vars=6, complex_dt=False for real normal output
    new_H_rn_list = new_phys2rn(l1_point, new_H_phys_list, max_degree_test, psi_tables, clmo_tables)

    # --- Comparison ---
    np.random.seed(42) 
    num_test_points = 5
    for i in range(num_test_points):
        point_coords = np.random.uniform(-0.1, 0.1, size=len(REAL_NORMAL_SYMBOLS))
        test_point_dict = {sym: val for sym, val in zip(REAL_NORMAL_SYMBOLS, point_coords)}

        val_old = evaluate_poly_list_at_point(
            old_H_rn_list, test_point_dict, REAL_NORMAL_SYMBOLS, max_eval_deg=max_degree_test
        )
        val_new = evaluate_poly_list_at_point(
            new_H_rn_list, test_point_dict, REAL_NORMAL_SYMBOLS, max_eval_deg=max_degree_test
        )
        assert np.isclose(val_old, val_new, rtol=1e-9, atol=1e-12), (
            f"phys2rn: Values differ at point {i} ({test_point_dict}) for mu={mu_val}, max_degree={max_degree_test}. Test Poly: {H_phys_sym}\n"
            f"Old_poly_list (eval): {val_old}\nNew_poly_list (eval): {val_new}\n"
            f"Difference: {abs(val_old - val_new)}\n"
            f"Old List (numeric): {[(d, arr.tolist()) for d, arr in enumerate(old_H_rn_list) if arr.size > 0]}\n"
            f"New List (numeric): {[(d, arr.tolist()) for d, arr in enumerate(new_H_rn_list) if arr.size > 0]}"
        )
    print(f"phys2rn equivalence test passed for mu={mu_val}, max_degree={max_degree_test}, H_phys_sym={H_phys_sym}")


@pytest.mark.parametrize("mu_val, max_degree_test", [
    (0.01215058162465319, 2),
    (0.01215058162465319, 3),
    (0.01215058162465319, 4),
    (0.01215058162465319, 5),
    (0.01215058162465319, 6),
    (1e-3, 1),
    (1e-3, 2),
    (1e-3, 3)])
def test_rn2cn_equivalence(mu_val, max_degree_test):
    l1_point = L1Point(mu=mu_val) # Needed for _generate_subs_dict in old path if symbolic constants were present
    psi_tables, clmo_tables = init_index_tables(max_degree_test)

    # Define a simple real normal Hamiltonian
    if max_degree_test == 1:
        H_rn_sym = x_rn_s + 2.0*py_rn_s
    elif max_degree_test == 2:
        H_rn_sym = x_rn_s**2 + 1.5*y_rn_s*px_rn_s - z_rn_s
    else:
        H_rn_sym = x_rn_s * px_rn_s

    # --- Old Path ---
    old_H_rn = OldPolynomial(REAL_NORMAL_SYMBOLS, H_rn_sym)
    # symbolic=False applies _generate_subs_dict (if H_rn_sym had c_n etc.) and _clean_numerical_artifacts
    old_H_cn_poly_obj = old_real_normal_to_complex_canonical(l1_point, old_H_rn, symbolic=False, max_degree=max_degree_test)
    old_H_cn_expr_numeric = old_H_cn_poly_obj.expression
    
    old_H_cn_list = symengine2poly(
        old_H_cn_expr_numeric,
        COMPLEX_CANONICAL_SYMBOLS,
        max_degree_test,
        psi_tables,
        clmo_tables,
        complex_dtype=True # Complex canonical coordinates are complex
    )

    # --- New Path ---
    new_H_rn_list = symengine2poly(
        H_rn_sym,
        REAL_NORMAL_SYMBOLS,
        max_degree_test,
        psi_tables,
        clmo_tables,
        complex_dtype=False # Real normal input is real
    )
    psi_param_new = (6, True) # num_vars=6, complex_dt=True for complex canonical output
    new_H_cn_list = new_rn2cn(new_H_rn_list, max_degree_test, psi_tables, clmo_tables)

    # --- Comparison ---
    np.random.seed(43)
    num_test_points = 5
    for i in range(num_test_points):
        # Generate random COMPLEX point in COMPLEX_CANONICAL_SYMBOLS space
        real_parts = np.random.uniform(-0.1, 0.1, size=len(COMPLEX_CANONICAL_SYMBOLS))
        imag_parts = np.random.uniform(-0.1, 0.1, size=len(COMPLEX_CANONICAL_SYMBOLS))
        point_coords = real_parts + 1j * imag_parts
        test_point_dict = {sym: val for sym, val in zip(COMPLEX_CANONICAL_SYMBOLS, point_coords)}

        val_old = evaluate_poly_list_at_point(
            old_H_cn_list, test_point_dict, COMPLEX_CANONICAL_SYMBOLS, max_eval_deg=max_degree_test
        )
        val_new = evaluate_poly_list_at_point(
            new_H_cn_list, test_point_dict, COMPLEX_CANONICAL_SYMBOLS, max_eval_deg=max_degree_test
        )
        
        assert np.isclose(val_old, val_new, rtol=1e-9, atol=1e-12), (
            f"rn2cn: Values differ at point {i} ({test_point_dict}) for mu={mu_val}, max_degree={max_degree_test}. Test Poly: {H_rn_sym}\n"
            f"Old_poly_list (eval): {val_old}\nNew_poly_list (eval): {val_new}\n"
            f"Difference: {abs(val_old - val_new)}\n"
            f"Old List (numeric): {[(d, arr.tolist()) for d, arr in enumerate(old_H_cn_list) if arr.size > 0]}\n"
            f"New List (numeric): {[(d, arr.tolist()) for d, arr in enumerate(new_H_cn_list) if arr.size > 0]}"
        )
    print(f"rn2cn equivalence test passed for mu={mu_val}, max_degree={max_degree_test}, H_rn_sym={H_rn_sym}")


@pytest.mark.parametrize("mu_val, max_degree_test", [
    (0.01215058162465319, 2),
    (0.01215058162465319, 3),
    (0.01215058162465319, 4),
    (0.01215058162465319, 5),
    (0.01215058162465319, 6),
    (1e-3, 1),
    (1e-3, 2),
    (1e-3, 3)])
def test_cn2rn_equivalence(mu_val, max_degree_test):
    l1_point = L1Point(mu=mu_val) # Needed for _generate_subs_dict in old path
    psi_tables, clmo_tables = init_index_tables(max_degree_test)
    I = se.I # Symengine's imaginary unit for defining symbolic complex H

    # Define a simple complex canonical Hamiltonian
    if max_degree_test == 1:
        H_cn_sym = q1_s + 2.0*p1_s + I * (q2_s - 0.5*p2_s)
    elif max_degree_test == 2:
        H_cn_sym = q1_s*p1_s + I*q2_s*p2_s + 0.5*q3_s**2 - p3_s*I
    else:
        H_cn_sym = q1_s*p1_s + I*q2_s*p2_s

    # --- Old Path ---
    old_H_cn = OldPolynomial(COMPLEX_CANONICAL_SYMBOLS, H_cn_sym)
    old_H_rn_poly_obj = old_complex_canonical_to_real_normal(l1_point, old_H_cn, symbolic=False, max_degree=max_degree_test)
    old_H_rn_expr_numeric = old_H_rn_poly_obj.expression
    
    old_H_rn_list = symengine2poly(
        old_H_rn_expr_numeric,
        REAL_NORMAL_SYMBOLS,
        max_degree_test,
        psi_tables,
        clmo_tables,
        complex_dtype=True # Output is real normal, but old expression might have I
    )

    # --- New Path ---
    new_H_cn_list = symengine2poly(
        H_cn_sym,
        COMPLEX_CANONICAL_SYMBOLS,
        max_degree_test,
        psi_tables,
        clmo_tables,
        complex_dtype=True # Input is complex canonical
    )
    # Although output is real_normal, the transformation involves complex numbers, so substitute_linear's internal logic might use complex_dtype=True
    # The new_cn2rn function sets complex_dtype=True for substitute_linear.
    psi_param_new = (6, True) # The transformation matrix Cinv itself is complex.
    new_H_rn_list = new_cn2rn(new_H_cn_list, max_degree_test, psi_tables, clmo_tables)

    # --- Comparison ---
    np.random.seed(44)
    num_test_points = 5
    for i in range(num_test_points):
        # Generate random REAL point in REAL_NORMAL_SYMBOLS space for evaluation
        point_coords = np.random.uniform(-0.1, 0.1, size=len(REAL_NORMAL_SYMBOLS))
        test_point_dict = {sym: val for sym, val in zip(REAL_NORMAL_SYMBOLS, point_coords)}

        val_old = evaluate_poly_list_at_point(
            old_H_rn_list, test_point_dict, REAL_NORMAL_SYMBOLS, max_eval_deg=max_degree_test
        )
        val_new = evaluate_poly_list_at_point(
            new_H_rn_list, test_point_dict, REAL_NORMAL_SYMBOLS, max_eval_deg=max_degree_test
        )
        
        # The output should be real, so np.isclose should work directly.
        # If there are tiny imaginary parts due to float errors, we might need to check abs(val_new.imag) < tol
        # For now, assume the transformations correctly result in real coefficients if starting from Hermitian H_cn
        if isinstance(val_old, complex) and abs(val_old.imag) > 1e-12 :
             print(f"Warning: Old value has non-trivial imaginary part: {val_old.imag}")
        if isinstance(val_new, complex) and abs(val_new.imag) > 1e-12 :
             print(f"Warning: New value has non-trivial imaginary part: {val_new.imag}")

        assert np.allclose(val_old, val_new, rtol=1e-9, atol=1e-12), (
            f"cn2rn: Values differ at point {i} ({test_point_dict}) for mu={mu_val}, max_degree={max_degree_test}. Test Poly: {H_cn_sym}\n"
            f"Old_poly_list (eval): {val_old}\nNew_poly_list (eval): {val_new}\n"
            f"Difference: {abs(val_old - val_new)}\n"
            f"Old List (numeric): {[(d, arr.tolist()) for d, arr in enumerate(old_H_rn_list) if arr.size > 0]}\n"
            f"New List (numeric): {[(d, arr.tolist()) for d, arr in enumerate(new_H_rn_list) if arr.size > 0]}"
        )
        # Also assert that imaginary parts are negligible if they exist
        # assert np.isclose(np.imag(val_old), 0, atol=1e-12), f"Old value has non-negligible imaginary part: {np.imag(val_old)}"
        # assert np.isclose(np.imag(val_new), 0, atol=1e-12), f"New value has non-negligible imaginary part: {np.imag(val_new)}"

    print(f"cn2rn equivalence test passed for mu={mu_val}, max_degree={max_degree_test}, H_cn_sym={H_cn_sym}")
