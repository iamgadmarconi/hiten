import pytest
import numpy as np
import symengine as se
import sympy as sp # Used by old symbolic Hamiltonian for cleaning

from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center._deprecated.dep_hamiltonian import hamiltonian as old_hamiltonian, _generate_subs_dict
from algorithms.center._deprecated.dep_core import Polynomial as OldPolynomial, _clean_numerical_artifacts
from system.libration import L1Point
from algorithms.center.polynomial.conversions import symengine2poly
from algorithms.center.polynomial.base import init_index_tables


x_s, y_s, z_s, px_s, py_s, pz_s = se.symbols("x y z px py pz")
PHYSICAL_SYMBOLS = [x_s, y_s, z_s, px_s, py_s, pz_s]


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


def evaluate_new_hamiltonian_poly(poly_coeffs_list, point_values_dict, sym_vars_ordered_list, max_eval_deg):
    num_vars = len(sym_vars_ordered_list)
    total_value = 0.0

    for d in range(len(poly_coeffs_list)):
        if d > max_eval_deg:
            continue
        
        coeffs_for_deg_d = poly_coeffs_list[d]
        
        # Handle cases where a degree might have no coefficients (e.g. None, empty list/array)
        if coeffs_for_deg_d is None: 
            continue
        if isinstance(coeffs_for_deg_d, (list, tuple)) and not coeffs_for_deg_d:
            continue

        if not isinstance(coeffs_for_deg_d, np.ndarray) or coeffs_for_deg_d.size == 0:
            # If it's some other empty sequence type that slips through, or an explicit empty ndarray
            if hasattr(coeffs_for_deg_d, '__len__') and len(coeffs_for_deg_d) == 0:
                 continue
            elif not (isinstance(coeffs_for_deg_d, np.ndarray) and coeffs_for_deg_d.size == 0) :
                 # If it's not an empty numpy array and not None/empty list/tuple, this is unexpected
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
            if abs(coeff_val) < 1e-18: # Tolerance for effectively zero coefficients
                continue
            
            monomial_val = 1.0
            for var_idx, exponent_val in enumerate(exponents):
                if exponent_val < 0: # Should not happen for standard polynomials
                    raise ValueError(f"Negative exponent {exponent_val} encountered.")
                if exponent_val > 0:
                    var_symbol = sym_vars_ordered_list[var_idx]
                    monomial_val *= (point_values_dict[var_symbol] ** exponent_val)
            
            total_value += coeff_val * monomial_val
            
    return total_value


@pytest.mark.parametrize("mu_val, max_degree_test", [
    (0.01215058162465319, 2),
    (0.01215058162465319, 3),
    (0.01215058162465319, 4),
    (0.01215058162465319, 5),
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
        val_old = evaluate_new_hamiltonian_poly(
            old_H_poly_list_representation, 
            test_point_dict, 
            PHYSICAL_SYMBOLS, 
            max_eval_deg=max_degree_test
        )

        # Evaluate new Hamiltonian (polynomial list)
        try:
            val_new = evaluate_new_hamiltonian_poly(new_H_poly_list, test_point_dict, PHYSICAL_SYMBOLS, max_degree_test)
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
